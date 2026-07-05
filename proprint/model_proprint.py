"""ProPRINT model components."""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAMBA_PATH = Path(os.environ.get("MAMBA_PATH", CODE_ROOT / "MambaOut-main"))
if str(DEFAULT_MAMBA_PATH) not in sys.path:
    sys.path.append(str(DEFAULT_MAMBA_PATH))
from models.mambaout import mambaout_small


class ProteinTower(nn.Module):
    """Protein representation tower."""

    def __init__(self, n_dim: int = 680, g: int = 12, d: int = 128, dropout: float = 0.1):
        super().__init__()
        self.n_dim = n_dim
        self.g = g
        self.d = d

        self.group_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_dim, d),
                nn.GELU(),
                nn.LayerNorm(d),
                nn.Dropout(dropout),
            )
            for _ in range(g)
        ])
        self.semantic_bottleneck = nn.Sequential(
            nn.Linear(g * d, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
        )
        self.skip = nn.Linear(n_dim, 128)
        self.out_norm = nn.LayerNorm(128)

    def forward(self, prot: torch.Tensor):
        tokens = torch.stack([head(prot) for head in self.group_heads], dim=1)
        sem = self.semantic_bottleneck(tokens.flatten(1))
        sem = self.out_norm(sem + self.skip(prot))
        return tokens, sem


class PrototypeMaskedRefiner(nn.Module):
    """Mask-aware VPR refinement module."""

    def __init__(
        self,
        img_dim: int = 576,
        semantic_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.mask_token = nn.Parameter(torch.zeros(1, semantic_dim))
        self.refiner = nn.Sequential(
            nn.Linear(img_dim + semantic_dim + semantic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, semantic_dim),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.3))

    def _random_mask_indices(self, batch_size: int, mask_ratio: float, device: torch.device):
        n_mask = max(1, int(round(self.semantic_dim * float(mask_ratio))))
        n_mask = min(self.semantic_dim, n_mask)
        return [torch.randperm(self.semantic_dim, device=device)[:n_mask] for _ in range(batch_size)]

    def forward(
        self,
        img_global: torch.Tensor,
        vpr_base: torch.Tensor,
        rpr_gt: torch.Tensor = None,
        mask_ratio: float = 0.5,
        training: bool = True,
    ):
        batch_size = img_global.shape[0]
        device = img_global.device

        if training and rpr_gt is not None:
            mask_indices = self._random_mask_indices(batch_size, mask_ratio, device)
            masked_rpr = rpr_gt.clone()
            for i, idx in enumerate(mask_indices):
                masked_rpr[i, idx] = self.mask_token[0, idx]
        else:
            masked_rpr = self.mask_token.expand(batch_size, -1)
            mask_indices = None

        residual = self.refiner(torch.cat([img_global, vpr_base, masked_rpr], dim=1))
        vpr = vpr_base + torch.sigmoid(self.res_scale) * residual
        rpr_recon = vpr
        return vpr, rpr_recon, mask_indices


class ImageToRepresentationPredictor(nn.Module):
    """Image-to-representation predictor."""

    def __init__(
        self,
        input_dim: int = 576,
        hidden_dim1: int = 384,
        hidden_dim2: int = 256,
        output_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        K_prototypes: int = 3,
        use_masked_refine: bool = True,
    ):
        super().__init__()
        self.K = K_prototypes
        self.output_dim = output_dim
        self.use_masked_refine = use_masked_refine
        self.assign_temperature = 1.0
        self.prototypes_initialized = True

        self.token_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim2,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim2)
        self.to_rep = nn.Sequential(
            nn.LayerNorm(hidden_dim2),
            nn.Linear(hidden_dim2, output_dim),
        )

        self.prototypes = nn.Parameter(F.normalize(torch.randn(K_prototypes, output_dim), p=2, dim=1))
        if use_masked_refine:
            self.masked_refiner = PrototypeMaskedRefiner(
                img_dim=input_dim,
                semantic_dim=output_dim,
                hidden_dim=512,
                dropout=dropout,
            )

    def set_prototypes(self, prototypes: np.ndarray):
        assert prototypes.shape == (self.K, self.output_dim), (
            f"prototype shape mismatch: {prototypes.shape} vs {(self.K, self.output_dim)}"
        )
        proto = torch.tensor(prototypes, dtype=torch.float32, device=self.prototypes.device)
        self.prototypes.data.copy_(F.normalize(proto, p=2, dim=1))
        self.prototypes_initialized = True
        print(f"[ImageToRepresentationPredictor] prototypes set: K={self.K}, dim={self.output_dim}")

    def set_assignment_temperature(self, tau: float):
        self.assign_temperature = float(max(1e-4, tau))
        print(f"[ImageToRepresentationPredictor] assignment temperature: {self.assign_temperature:.3f}")

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        img_global: torch.Tensor = None,
        rpr_gt: torch.Tensor = None,
        return_assignment: bool = False,
        mask_ratio: float = 0.5,
        training: bool = True,
    ):
        if spatial_tokens.dim() == 2:
            if img_global is None:
                img_global = spatial_tokens
            spatial_tokens = spatial_tokens.unsqueeze(1)
        elif img_global is None:
            img_global = spatial_tokens.mean(dim=1)

        x = self.token_proj(spatial_tokens)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.attn_norm(x + attn_out)
        visual_rep = self.to_rep(x.mean(dim=1))

        visual_rep_n = F.normalize(visual_rep, p=2, dim=1)
        proto_n = F.normalize(self.prototypes, p=2, dim=1)
        assignment_logits = torch.matmul(visual_rep_n, proto_n.t())
        assignment_weights = F.softmax(assignment_logits / self.assign_temperature, dim=1)
        vpr_base = torch.matmul(assignment_weights, self.prototypes)

        if self.use_masked_refine:
            vpr, rpr_recon, mask_indices = self.masked_refiner(
                img_global=img_global,
                vpr_base=vpr_base,
                rpr_gt=rpr_gt,
                mask_ratio=mask_ratio,
                training=training,
            )
        else:
            vpr, rpr_recon, mask_indices = vpr_base, None, None

        if return_assignment:
            return vpr, assignment_weights, assignment_logits, rpr_recon, mask_indices
        return vpr


class ProPRINTModel(nn.Module):
    """ProPRINT dual-path model."""

    def __init__(
        self,
        protein_dim: int = 680,
        g: int = 12,
        d: int = 128,
        drop_path_rate: float = 0.3,
        num_classes: int = 2,
        load_backbone_pretrained: bool = True,
        pretrained_path: str = None,
        kappa: float = 0.7,
        K_prototypes: int = 3,
        use_masked_refine: bool = True,
    ):
        super().__init__()
        self.kappa = kappa
        self.protein_dim = protein_dim

        self.backbone = mambaout_small(
            pretrained=False,
            num_classes=num_classes,
            in_chans=3,
            drop_path_rate=drop_path_rate,
        )

        if pretrained_path is None:
            pretrained_path = str(DEFAULT_MAMBA_PATH / "models" / "mambaout_small.pth")
        if load_backbone_pretrained and os.path.isfile(pretrained_path):
            state = torch.load(pretrained_path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            clean_state = {k: v for k, v in state.items() if not k.startswith("head.")}
            self.backbone.load_state_dict(clean_state, strict=False)

        self.tower = ProteinTower(n_dim=protein_dim, g=g, d=d)
        self.img2rep = ImageToRepresentationPredictor(
            input_dim=576,
            hidden_dim1=384,
            hidden_dim2=256,
            output_dim=128,
            n_heads=4,
            dropout=0.1,
            K_prototypes=K_prototypes,
            use_masked_refine=use_masked_refine,
        )

        self.gamma_fc = nn.Linear(128, 576)
        self.beta_fc = nn.Linear(128, 576)
        self.gate_fc = nn.Linear(128, 576)
        self.mod_drop = nn.Dropout(0.1)
        self.fused_ln = nn.LayerNorm(576)

        self.fc1 = nn.Linear(576, 512)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

        self.tap_norms = nn.ModuleList([
            nn.LayerNorm(96),
            nn.LayerNorm(192),
            nn.LayerNorm(384),
            nn.LayerNorm(576),
        ])

    def _forward_backbone(self, x_img: torch.Tensor):
        x = x_img
        taps = []
        for i in range(self.backbone.num_stage):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)
            pooled = x.mean([1, 2])
            taps.append(self.tap_norms[i](pooled))

        spatial = self.backbone.norm(x)
        img_global = spatial.mean([1, 2])
        spatial_tokens = spatial.flatten(1, 2)
        return img_global, spatial_tokens, taps

    def _classifier(self, fused: torch.Tensor):
        x = self.fc1(fused)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)

    def _film_se(self, img_vec: torch.Tensor, prot_sem: torch.Tensor, kappa: float = None):
        if kappa is None:
            kappa = self.kappa
        gamma = torch.tanh(self.gamma_fc(prot_sem))
        beta = self.beta_fc(prot_sem)
        gate = torch.sigmoid(self.gate_fc(prot_sem)) * kappa
        mod = gate * ((1.0 + gamma) * img_vec + beta)
        return self.fused_ln(img_vec + self.mod_drop(mod))

    def forward_real_path(self, x_img: torch.Tensor, prot: torch.Tensor):
        img_global, spatial_tokens, taps = self._forward_backbone(x_img)
        prot_tokens, prot_sem = self.tower(prot)
        fused_cls = self._film_se(img_global, prot_sem)
        logits_real = self._classifier(fused_cls)
        return logits_real, fused_cls, img_global, prot_sem, taps

    def forward_virtual_path(
        self,
        x_img: torch.Tensor = None,
        rpr_gt: torch.Tensor = None,
        img_global: torch.Tensor = None,
        spatial_tokens: torch.Tensor = None,
        taps: list = None,
        mask_ratio: float = 0.5,
        training: bool = True,
    ):
        if img_global is None or spatial_tokens is None:
            img_global, spatial_tokens, taps = self._forward_backbone(x_img)

        vpr, assign_w, assign_logits, rpr_recon, mask_indices = self.img2rep(
            spatial_tokens,
            img_global=img_global,
            rpr_gt=rpr_gt,
            return_assignment=True,
            mask_ratio=mask_ratio,
            training=training,
        )
        fused_hat = self._film_se(img_global, vpr)
        logits_virtual = self._classifier(fused_hat)
        return logits_virtual, fused_hat, vpr, taps, rpr_recon, mask_indices, assign_w, assign_logits

    def forward(
        self,
        x_img: torch.Tensor,
        prot: torch.Tensor = None,
        use_virtual_only: bool = False,
        mask_ratio: float = 0.5,
        training: bool = True,
    ):
        if use_virtual_only:
            logits_virtual, fused_hat, vpr, taps, rpr_recon, mask_indices, _, _ = self.forward_virtual_path(
                x_img=x_img,
                rpr_gt=None,
                mask_ratio=mask_ratio,
                training=training,
            )
            return logits_virtual, fused_hat, vpr, taps, rpr_recon, mask_indices

        img_global, spatial_tokens, taps = self._forward_backbone(x_img)
        prot_tokens, prot_sem = self.tower(prot)
        fused_cls = self._film_se(img_global, prot_sem)
        logits_real = self._classifier(fused_cls)

        logits_virtual, fused_hat, vpr, _, rpr_recon, mask_indices, assign_w, assign_logits = self.forward_virtual_path(
            rpr_gt=prot_sem,
            img_global=img_global,
            spatial_tokens=spatial_tokens,
            taps=taps,
            mask_ratio=mask_ratio,
            training=training,
        )

        aux = {
            "rpr_recon_masked": rpr_recon,
            "mask_indices": mask_indices,
            "assignment_weights": assign_w,
            "assignment_logits": assign_logits,
        }
        return logits_real, logits_virtual, fused_cls, fused_hat, prot_sem, vpr, img_global, taps, aux

    def set_kappa(self, kappa: float):
        self.kappa = float(kappa)

    def freeze_for_stage2(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.tower.parameters():
            param.requires_grad = False
        for param in self.gamma_fc.parameters():
            param.requires_grad = False
        for param in self.beta_fc.parameters():
            param.requires_grad = False
        for param in self.gate_fc.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = True
        for param in self.fc2.parameters():
            param.requires_grad = True
        for param in self.img2rep.parameters():
            param.requires_grad = True

    def progressive_unfreeze(self, epoch: int, total_epochs: int):
        progress = epoch / max(total_epochs - 1, 1)
        if progress >= 0.5:
            for param in self.backbone.stages[2].parameters():
                param.requires_grad = True
            for param in self.backbone.downsample_layers[2].parameters():
                param.requires_grad = True
            for param in self.gamma_fc.parameters():
                param.requires_grad = True
            for param in self.beta_fc.parameters():
                param.requires_grad = True


def build_proprint_model(
    device=None,
    protein_dim: int = 680,
    g: int = 12,
    d: int = 128,
    drop_path_rate: float = 0.3,
    kappa: float = 0.7,
    load_backbone_pretrained: bool = True,
    K_prototypes: int = 3,
    use_masked_refine: bool = True,
):
    set_seed(42)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ProPRINTModel(
        protein_dim=protein_dim,
        g=g,
        d=d,
        drop_path_rate=drop_path_rate,
        kappa=kappa,
        load_backbone_pretrained=load_backbone_pretrained,
        K_prototypes=K_prototypes,
        use_masked_refine=use_masked_refine,
    )
    model.to(device)
    return model


__all__ = [
    "ProteinTower",
    "PrototypeMaskedRefiner",
    "ImageToRepresentationPredictor",
    "ProPRINTModel",
    "build_proprint_model",
]
