"""ProPRINT Stage B fine-tuning script."""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import average_precision_score


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = CODE_ROOT
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "proprint"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "proprint"
STAGEB_SEED = 42
STAGEB_FOLD = 0
STAGEB_HIGHER_LR_MULT = 1.0
STAGEB_FL_GAMMA = 2.0
STAGEB_FL_ALPHA_POS = 1.2
STAGEB_LABEL_SMOOTHING = 0.1
STAGEB_DROP_PATH = 0.2
STAGEB_GRAD_CLIP_MAX_NORM = 1.0
STAGEB_KAPPA_INIT = 0.4
STAGEB_KAPPA_MID = 0.5
STAGEB_KAPPA_FINAL = 0.6
STAGEB_TAU_INIT = 1.8
STAGEB_TAU_FINAL = 1.3
STAGEB_ANCHOR_START_PROGRESS = 0.3
STAGEB_ANCHOR_MAX_WEIGHT = 0.03
STAGEB_TTA = 4
STAGEB_AGGREGATION = "max"
if str(CODE_ROOT) in sys.path:
    sys.path.remove(str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT))

from utils.dataset import ThyroidDataset, build_transforms
from proprint.model_proprint import build_proprint_model


def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def binary_roc_auc(scores, labels):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).astype(np.float64)
    equal = (pos[:, None] == neg[None, :]).astype(np.float64)
    return float((greater.sum() + 0.5 * equal.sum()) / (pos.size * neg.size))


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        log_prob = F.log_softmax(logits, dim=1)
        prob = torch.exp(log_prob)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()

        if self.label_smoothing > 0.0:
            classes = logits.shape[1]
            eps = self.label_smoothing
            targets_one_hot = targets_one_hot * (1.0 - eps) + eps / float(classes)

        pt = (prob * targets_one_hot).sum(dim=1)
        loss = -((1 - pt) ** self.gamma) * (targets_one_hot * log_prob).sum(dim=1)

        if self.alpha is not None:
            alpha_t = targets_one_hot * self.alpha
            loss = alpha_t.sum(dim=1) * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def set_dataset_df(dataset: ThyroidDataset, df: pd.DataFrame):
    if hasattr(dataset, "set_dataframe"):
        dataset.set_dataframe(df)
    else:
        dataset.df = df.reset_index(drop=True)


def assert_disjoint_ids(named_frames):
    seen = {}
    for split_name, df in named_frames:
        ids = set(df["ID"].astype(str).str.strip())
        overlap = ids.intersection(seen.keys())
        if overlap:
            examples = sorted(overlap)[:10]
            previous = sorted({seen[pid] for pid in overlap})
            raise ValueError(
                f"Patient ID overlap detected between {split_name} and {previous}: {examples}. "
                "Stage B train/internal/external tables must be patient-disjoint."
            )
        for pid in ids:
            seen[pid] = split_name


def build_sampler_weights(dataset):
    df = dataset.df
    label_counts = df["Label"].value_counts()
    total_n = len(df)
    class_weights = {label: total_n / count for label, count in label_counts.items()}
    return torch.tensor([class_weights[row["Label"]] for _, row in df.iterrows()], dtype=torch.float32)


def aggregate_probs(prob_list, aggregation: str):
    if len(prob_list) == 0:
        return float("nan")
    if aggregation == "max":
        return float(max(prob_list))
    if aggregation == "top3":
        top3 = sorted(prob_list, reverse=True)[:min(3, len(prob_list))]
        return float(np.mean(top3))
    return float(np.mean(prob_list))


def predict_with_tta(model, loader, device, tta_views=4, aggregation="max"):
    model.eval()
    patient_probs = {}
    patient_labels = {}
    image_count = 0

    with torch.no_grad():
        for images, _proteins, labels, pids, _proto_ids in loader:
            images = images.to(device)
            image_count += int(images.shape[0])

            with autocast(enabled=True):
                logits_sum, _, _, _, _, _ = model(images, prot=None, use_virtual_only=True, training=False)
                count = 1

                if tta_views >= 2:
                    logits_h, _, _, _, _, _ = model(torch.flip(images, dims=[3]), prot=None, use_virtual_only=True, training=False)
                    logits_sum = logits_sum + logits_h
                    count += 1
                if tta_views >= 3:
                    logits_v, _, _, _, _, _ = model(torch.flip(images, dims=[2]), prot=None, use_virtual_only=True, training=False)
                    logits_sum = logits_sum + logits_v
                    count += 1
                if tta_views >= 4:
                    logits_hv, _, _, _, _, _ = model(
                        torch.flip(torch.flip(images, dims=[3]), dims=[2]),
                        prot=None,
                        use_virtual_only=True,
                        training=False,
                    )
                    logits_sum = logits_sum + logits_hv
                    count += 1

                logits = logits_sum / float(count)
                probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

            labels_np = labels.detach().cpu().numpy()
            for i, pid in enumerate([str(x) for x in pids]):
                patient_probs.setdefault(pid, []).append(float(probs[i]))
                patient_labels[pid] = int(labels_np[i])

    patient_ids = sorted(patient_probs.keys())
    probs = [aggregate_probs(patient_probs[pid], aggregation) for pid in patient_ids]
    labels = [patient_labels[pid] for pid in patient_ids]
    return probs, labels, patient_ids, image_count


def build_eval_dataset(data_dir, clinical_file, df, transform, fold, image_mode, image_subdir):
    dataset = ThyroidDataset(
        data_dir=str(data_dir),
        clinical_file=clinical_file,
        protein_file="stageA_protein.xlsx",
        transform=transform,
        training=False,
        fold=fold,
        n_folds=5,
        image_mode=image_mode,
        load_protein=False,
        image_subdir=image_subdir,
    )
    set_dataset_df(dataset, df)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="ProPRINT Stage B fine-tuning")

    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="auto", help="Device, e.g. auto/cuda:0/cpu")

    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Directory containing Stage B data files")
    parser.add_argument("--image_subdir", type=str, default="stageB_ultrasound", help="Stage B image folder under data_dir")
    parser.add_argument("--init_weight", type=str, default=str(WEIGHTS_DIR / "proprint_stageA.pth"), help="Stage A checkpoint")
    parser.add_argument("--prototype_path", type=str, default=str(WEIGHTS_DIR / "prototypes.npz"), help="Prototype npz file")
    parser.add_argument("--save_dir", type=str, default=str(WEIGHTS_DIR), help="Checkpoint output directory")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR), help="Prediction output directory")
    parser.add_argument("--protein_dim", type=int, default=680, help="Protein feature dimension used by the Stage A checkpoint")

    parser.add_argument("--lr", type=float, default=1e-5, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=4e-4, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    args = parser.parse_args()
    set_seed(STAGEB_SEED)
    device = resolve_device(args.device)

    data_dir = Path(args.data_dir)
    train_xlsx = data_dir / "stageB_train.xlsx"
    val_xlsx = data_dir / "stageB_internal_val.xlsx"
    ext_xlsx = data_dir / "stageB_external_val.xlsx"
    for path in [train_xlsx, val_xlsx, ext_xlsx]:
        if not path.is_file():
            raise FileNotFoundError(f"Required Stage B data file not found: {path}")

    train_df = pd.read_excel(train_xlsx)
    val_df = pd.read_excel(val_xlsx)
    external_df = pd.read_excel(ext_xlsx)
    for df in [train_df, val_df, external_df]:
        df["ID"] = df["ID"].astype(str).str.strip()
    assert_disjoint_ids([
        ("stageB_train", train_df),
        ("stageB_internal_val", val_df),
        ("stageB_external_val", external_df),
    ])

    image_mode = "L"
    image_subdir = args.image_subdir
    train_transform = build_transforms(use_grayscale=True, training=True)
    val_transform = build_transforms(use_grayscale=True, training=False)

    train_dataset = ThyroidDataset(
        data_dir=str(data_dir),
        clinical_file="stageB_train.xlsx",
        protein_file="stageA_protein.xlsx",
        transform=train_transform,
        training=True,
        fold=STAGEB_FOLD,
        n_folds=5,
        image_mode=image_mode,
        load_protein=False,
        image_subdir=image_subdir,
    )
    set_dataset_df(train_dataset, train_df)
    val_dataset = build_eval_dataset(data_dir, "stageB_internal_val.xlsx", val_df, val_transform, STAGEB_FOLD, image_mode, image_subdir)
    external_dataset = build_eval_dataset(data_dir, "stageB_external_val.xlsx", external_df, val_transform, STAGEB_FOLD, image_mode, image_subdir)

    print("\n" + "=" * 80)
    print("ProPRINT - Stage B fine-tuning")
    print("=" * 80)
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size}")
    print(f"Device: {device} | TTA views: {STAGEB_TTA} | aggregation: {STAGEB_AGGREGATION}")
    print(f"Data directory: {data_dir}")
    print(f"[Data] train patients={len(train_df)} | train records={len(train_dataset)}")
    print(f"[Data] internal val patients={len(val_df)} | internal val images={len(val_dataset)}")
    print(f"[Data] external patients={len(external_df)} | external images={len(external_dataset)}")
    print(f"[Schedule] tau: {STAGEB_TAU_INIT:.2f}->{STAGEB_TAU_FINAL:.2f} | kappa: {STAGEB_KAPPA_INIT:.2f}->{STAGEB_KAPPA_FINAL:.2f}")
    print("=" * 80 + "\n")

    weights = build_sampler_weights(train_dataset)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    external_loader = torch.utils.data.DataLoader(
        external_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if not os.path.isfile(args.prototype_path):
        raise FileNotFoundError(f"Prototype file not found: {args.prototype_path}")
    prototypes = np.load(args.prototype_path)
    proto_benign = torch.tensor(prototypes["benign"], dtype=torch.float32, device=device)
    proto_malignant = torch.tensor(prototypes["malignant"], dtype=torch.float32, device=device)
    if proto_benign.dim() == 1:
        proto_benign = proto_benign.unsqueeze(0)
    if proto_malignant.dim() == 1:
        proto_malignant = proto_malignant.unsqueeze(0)
    if proto_benign.shape[0] != 1 or proto_malignant.shape[0] != 2:
        raise ValueError(
            f"ProPRINT public configuration expects 1 benign + 2 malignant prototypes, "
            f"got benign={tuple(proto_benign.shape)}, malignant={tuple(proto_malignant.shape)}"
        )
    anchor_proto_benign = F.normalize(proto_benign, p=2, dim=1)
    anchor_proto_malignant = F.normalize(proto_malignant, p=2, dim=1)
    K_prototypes = 3

    model = build_proprint_model(
        device=device,
        protein_dim=args.protein_dim,
        drop_path_rate=STAGEB_DROP_PATH,
        kappa=STAGEB_KAPPA_INIT,
        load_backbone_pretrained=False,
        K_prototypes=K_prototypes,
        use_masked_refine=True,
    )

    if not os.path.isfile(args.init_weight):
        raise FileNotFoundError(f"Stage A checkpoint not found: {args.init_weight}")
    state_dict = torch.load(args.init_weight, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.img2rep.set_prototypes(torch.cat([anchor_proto_benign, anchor_proto_malignant], dim=0).detach().cpu().numpy())
    print(f"[Model] loaded Stage A checkpoint: {args.init_weight}")
    print(f"[Model] loaded prototypes: {args.prototype_path}\n")

    model.freeze_for_stage2()
    for module in [model.backbone.stages[3], model.backbone.downsample_layers[3], model.backbone.stages[2], model.backbone.downsample_layers[2]]:
        for param in module.parameters():
            param.requires_grad = True
    for module in [model.gamma_fc, model.beta_fc, model.gate_fc]:
        for param in module.parameters():
            param.requires_grad = True

    focal_loss = FocalLoss(
        gamma=STAGEB_FL_GAMMA,
        alpha=torch.tensor([1.0, STAGEB_FL_ALPHA_POS], device=device),
        reduction="mean",
        label_smoothing=STAGEB_LABEL_SMOOTHING,
    )

    def build_optimizer():
        group1_params = []
        group2_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "fc1" in name or "fc2" in name or "img2rep" in name:
                group1_params.append(param)
            else:
                group2_params.append(param)
        param_groups = []
        if group1_params:
            param_groups.append({"params": group1_params, "lr": args.lr, "name": "classifier+img2rep"})
        if group2_params:
            param_groups.append({"params": group2_params, "lr": args.lr * STAGEB_HIGHER_LR_MULT, "name": "backbone+film"})
        return optim.AdamW(param_groups, weight_decay=args.weight_decay)

    optimizer = build_optimizer()
    scheduler = None
    scaler = GradScaler()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "proprint_stageB_best.pth")
    best_auc_val = -float("inf")
    best_epoch = -1
    patience_counter = 0

    print("[Training] start\n")
    for epoch in range(args.epochs):
        progress = epoch / max(args.epochs - 1, 1)
        tau = STAGEB_TAU_INIT - (STAGEB_TAU_INIT - STAGEB_TAU_FINAL) * progress
        model.img2rep.set_assignment_temperature(tau)

        if progress < 0.3:
            current_kappa = STAGEB_KAPPA_INIT
        elif progress < 0.7:
            current_kappa = STAGEB_KAPPA_INIT + (STAGEB_KAPPA_MID - STAGEB_KAPPA_INIT) * (progress - 0.3) / 0.4
        else:
            current_kappa = STAGEB_KAPPA_MID + (STAGEB_KAPPA_FINAL - STAGEB_KAPPA_MID) * (progress - 0.7) / 0.3
        model.set_kappa(current_kappa)

        if progress < STAGEB_ANCHOR_START_PROGRESS:
            current_anchor_weight = 0.0
        else:
            denom = max(1e-6, 1.0 - STAGEB_ANCHOR_START_PROGRESS)
            current_anchor_weight = STAGEB_ANCHOR_MAX_WEIGHT * (progress - STAGEB_ANCHOR_START_PROGRESS) / denom
            current_anchor_weight = min(STAGEB_ANCHOR_MAX_WEIGHT, max(0.0, current_anchor_weight))

        if epoch < args.warmup_epochs:
            warmup_factor = 0.3 + 0.7 * ((epoch + 1) / max(1, args.warmup_epochs)) ** 2
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warmup_factor
        elif scheduler is None:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - args.warmup_epochs),
                eta_min=args.lr * 0.1,
            )

        model.train()
        running_loss = 0.0
        running_focal = 0.0
        running_anchor = 0.0
        for images, _proteins, labels, _pids, _proto_ids in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with autocast(enabled=True):
                logits_virtual, _fused_hat, vpr, _taps, _rpr_recon, _mask_indices = model(
                    images,
                    prot=None,
                    use_virtual_only=True,
                )
                loss_focal = focal_loss(logits_virtual, labels)

                vpr_norm = F.normalize(vpr, p=2, dim=1)
                anchor_loss_terms = []
                for i, label in enumerate(labels):
                    if int(label.item()) == 0:
                        prototype = anchor_proto_benign
                    else:
                        sims = F.cosine_similarity(
                            vpr_norm[i:i + 1].expand(anchor_proto_malignant.shape[0], -1),
                            anchor_proto_malignant,
                            dim=1,
                        )
                        prototype = anchor_proto_malignant[sims.argmax():sims.argmax() + 1]
                    anchor_loss_terms.append(1.0 - F.cosine_similarity(vpr_norm[i:i + 1], prototype, dim=1))
                loss_anchor = torch.cat(anchor_loss_terms).mean()
                loss = loss_focal + current_anchor_weight * loss_anchor

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=STAGEB_GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            running_focal += float(loss_focal.item())
            running_anchor += float(loss_anchor.item())

        if scheduler is not None:
            scheduler.step()

        val_probs, val_labels, _val_ids, val_image_count = predict_with_tta(
            model,
            val_loader,
            device,
            tta_views=STAGEB_TTA,
            aggregation=STAGEB_AGGREGATION,
        )
        val_auc = binary_roc_auc(val_probs, val_labels)
        val_ap = average_precision_score(val_labels, val_probs) if len(set(val_labels)) > 1 else float("nan")

        avg_loss = running_loss / max(1, len(train_loader))
        avg_focal = running_focal / max(1, len(train_loader))
        avg_anchor = running_anchor / max(1, len(train_loader))
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"[StageB] Epoch {epoch + 1}/{args.epochs} ({progress * 100:.1f}%) | LR={current_lr:.6f}")
        print(f"  dynamic: kappa={current_kappa:.3f} | tau={tau:.3f} | anchor_w={current_anchor_weight:.3f}")
        print(f"  loss: total={avg_loss:.4f} | focal={avg_focal:.4f} | anchor={avg_anchor:.4f}")
        print(f"  internal val: patients={len(val_labels)} | images={val_image_count} | AUC={val_auc:.4f} | AP={val_ap:.4f}")

        if val_auc > best_auc_val:
            best_auc_val = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  saved best checkpoint by internal validation AUC: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[INFO] early stopping after {args.patience} epochs without improvement")
                break
        print()

    print("\n" + "=" * 80)
    print("Stage B training finished")
    print(f"Best internal validation AUC: {best_auc_val:.4f} (Epoch {best_epoch + 1})")
    print(f"Checkpoint: {best_model_path}")
    print("=" * 80 + "\n")

    if not os.path.exists(best_model_path):
        print("[WARNING] no best checkpoint was saved; skip final evaluation and prediction export")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_probs, val_labels, val_ids, val_image_count = predict_with_tta(
        model,
        val_loader,
        device,
        tta_views=STAGEB_TTA,
        aggregation=STAGEB_AGGREGATION,
    )
    ext_probs, ext_labels, ext_ids, ext_image_count = predict_with_tta(
        model,
        external_loader,
        device,
        tta_views=STAGEB_TTA,
        aggregation=STAGEB_AGGREGATION,
    )
    val_auc = binary_roc_auc(val_probs, val_labels)
    ext_auc = binary_roc_auc(ext_probs, ext_labels)
    val_ap = average_precision_score(val_labels, val_probs) if len(set(val_labels)) > 1 else float("nan")
    ext_ap = average_precision_score(ext_labels, ext_probs) if len(set(ext_labels)) > 1 else float("nan")
    print(f"[Final internal] patients={len(val_labels)} | images={val_image_count} | AUC={val_auc:.4f} | AP={val_ap:.4f}")
    print(f"[Final external] patients={len(ext_labels)} | images={ext_image_count} | AUC={ext_auc:.4f} | AP={ext_ap:.4f}")

    finetune_full_df = pd.concat([train_df, val_df], ignore_index=True)
    finetune_full_dataset = build_eval_dataset(
        data_dir,
        "stageB_train.xlsx",
        finetune_full_df,
        val_transform,
        STAGEB_FOLD,
        image_mode,
        image_subdir,
    )
    finetune_full_loader = torch.utils.data.DataLoader(
        finetune_full_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    ft_probs, _ft_labels, ft_ids, _ft_image_count = predict_with_tta(
        model,
        finetune_full_loader,
        device,
        tta_views=STAGEB_TTA,
        aggregation=STAGEB_AGGREGATION,
    )

    prob_by_id = {pid: prob for pid, prob in zip(ft_ids, ft_probs)}
    prob_by_id.update({pid: prob for pid, prob in zip(ext_ids, ext_probs)})

    train_out = train_df.copy()
    train_out["dataset"] = 0
    val_out = val_df.copy()
    val_out["dataset"] = 1
    ext_out = external_df.copy()
    ext_out["dataset"] = 2
    union_df = pd.concat([train_out, val_out, ext_out], ignore_index=True)
    union_df["ID"] = union_df["ID"].astype(str).str.strip()
    union_df["proprint_prob_malignant"] = union_df["ID"].map(prob_by_id)

    output_excel_path = os.path.join(args.output_dir, "proprint_stageB_predictions.xlsx")
    union_df.to_excel(output_excel_path, index=False)
    print(f"[Output] prediction probabilities saved to: {output_excel_path}")


if __name__ == "__main__":
    main()
