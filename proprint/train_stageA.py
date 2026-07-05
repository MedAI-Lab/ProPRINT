"""ProPRINT Stage A training script."""

import os
import sys
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = CODE_ROOT
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "proprint"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "proprint"
STAGEA_SEED = 42
STAGEA_FOLD = 0
STAGEA_DROP_PATH = 0.3
STAGEA_P_NOISE_PROTEIN = 0.3
STAGEA_NOISE_SIGMA = 0.2
if str(CODE_ROOT) in sys.path:
    sys.path.remove(str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT))

from utils.dataset import ThyroidDataModule
from proprint.model_proprint import build_proprint_model


def set_seed(seed: int = 42):
    import random
    import numpy as np
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
        return float('nan')
    greater = (pos[:, None] > neg[None, :]).astype(np.float64)
    equal = (pos[:, None] == neg[None, :]).astype(np.float64)
    auc = (greater.sum() + 0.5 * equal.sum()) / (pos.size * neg.size)
    return float(auc)


def kl_divergence(logits_teacher, logits_student, T: float = 2.0):
    p_t = F.softmax(logits_teacher / T, dim=1)
    p_s = F.log_softmax(logits_student / T, dim=1)
    kd = F.kl_div(p_s, p_t, reduction='batchmean') * (T ** 2)
    return kd


def compute_cka(X, Y):
    X = X.detach().float()
    Y = Y.detach().float()
    
    K = torch.mm(X, X.t())  # B×B
    L = torch.mm(Y, Y.t())  # B×B
    
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
    K_c = torch.mm(torch.mm(H, K), H)
    L_c = torch.mm(torch.mm(H, L), H)
    
    hsic_kl = (K_c * L_c).sum()
    hsic_kk = (K_c * K_c).sum()
    hsic_ll = (L_c * L_c).sum()
    
    cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-8)
    return cka.item()


def compute_per_dim_correlation(X, Y):
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    
    D = X.shape[1]
    correlations = []
    for i in range(D):
        x_i = X[:, i]
        y_i = Y[:, i]
        sx = np.std(x_i)
        sy = np.std(y_i)
        if sx < 1e-4 or sy < 1e-4:
            correlations.append(0.0)
            continue
        x_mean = x_i.mean()
        y_mean = y_i.mean()
        x_c = x_i - x_mean
        y_c = y_i - y_mean
        cov = (x_c * y_c).mean()
        r = cov / (sx * sy + 1e-8)
        r = np.clip(r, -1.0, 1.0)
        correlations.append(r)
    
    return np.array(correlations)


def prototype_anchor_loss(prot_sem: torch.Tensor, prot_sem_hat: torch.Tensor, labels: torch.Tensor,
                          proto_benign: torch.Tensor, proto_malignant_all: torch.Tensor):
    del labels
    rpr = F.normalize(prot_sem.detach().float(), p=2, dim=1)
    vpr = F.normalize(prot_sem_hat.float(), p=2, dim=1)

    proto_b = proto_benign.float()
    if proto_b.dim() == 1:
        proto_b = proto_b.unsqueeze(0)
    proto_m = proto_malignant_all.float()
    if proto_m.dim() == 1:
        proto_m = proto_m.unsqueeze(0)
    prototypes = F.normalize(torch.cat([proto_b, proto_m], dim=0), p=2, dim=1)

    rpr_sim = torch.matmul(rpr, prototypes.t())
    vpr_sim = torch.matmul(vpr, prototypes.t())
    return F.mse_loss(vpr_sim, rpr_sim)
    


def get_dynamic_weights_smart(epoch, total_epochs, distill_started, last_auc_real, distill_start_epoch):
    min_teacher_epochs = max(10, int(0.20 * total_epochs))
    auc_threshold = 0.92
    if not distill_started:
        if epoch < min_teacher_epochs or (last_auc_real is None) or (last_auc_real < auc_threshold):
            return {
                'stage': 'teacher_build',
                'λv': 0.0, 'λc': 0.0, 'λd': 0.0, 'λr': 0.0, 'λmmd': 0.0,
                'λanchor': 0.0, 'λfused': 0.0, 'λent': 0.0, 'λnce': 0.0,
            }
        return {
            'stage': 'semantic_distill_early',
            'λv': 0.6, 'λc': 0.0, 'λd': 1.0, 'λr': 0.0, 'λmmd': 0.0,
            'λanchor': 0.5, 'λfused': 0.8, 'λent': 0.02, 'λnce': 0.0,
        }
    rel = (epoch - distill_start_epoch) / max(1, (total_epochs - distill_start_epoch - 1))
    if rel < 0.80:
        return {
            'stage': 'semantic_distill_early',
            'λv': 0.6, 'λc': 0.0, 'λd': 1.0, 'λr': 0.0, 'λmmd': 0.0,
            'λanchor': 0.5, 'λfused': 0.8, 'λent': 0.02, 'λnce': 0.0,
        }
    return {
        'stage': 'semantic_distill_late',
        'λv': 1.2, 'λc': 0.0, 'λd': 0.7, 'λr': 0.0, 'λmmd': 0.0,
        'λanchor': 0.3, 'λfused': 0.8, 'λent': 0.02, 'λnce': 0.0,
    }


def get_temperature_schedules(epoch, total_epochs):
    progress = epoch / max(total_epochs - 1, 1)
    T_kd = max(1.0, 2.5 - 1.5 * progress)
    tau_assign = max(0.8, 1.5 - 0.8 * progress)
    return {'T_kd': T_kd, 'tau_assign': tau_assign}


def compute_prototypes_from_data(model, train_loader, assign_df, device, Kp):
    model.eval()
    patient_sem = {}
    patient_lbl = {}
    
    with torch.no_grad():
        for images, proteins, labels, pids, proto_ids in train_loader:
            images = images.to(device)
            proteins = proteins.to(device)
            logits_real, fused_cls, img_global, prot_sem, taps = model.forward_real_path(images, proteins)
            sem_np = prot_sem.detach().cpu().numpy()
            lbl_np = labels.detach().cpu().numpy()
            pids_np = [str(pid) for pid in pids]
            for i, pid in enumerate(pids_np):
                if pid not in patient_sem:
                    patient_sem[pid] = []
                patient_sem[pid].append(sem_np[i])
                patient_lbl[pid] = int(lbl_np[i])
    
    patient_sem_mean = {pid: np.mean(np.stack(vecs, axis=0), axis=0) for pid, vecs in patient_sem.items()}
    
    clusters = sorted(assign_df['cluster'].unique().tolist())
    proto_m_list = []
    for cid in clusters:
        ids_c = assign_df.loc[assign_df['cluster'] == cid, 'ID'].tolist()
        vecs = [patient_sem_mean[pid] for pid in ids_c if pid in patient_sem_mean]
        if len(vecs) == 0:
            continue
        v = np.stack(vecs, axis=0)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        c_mean = v.mean(axis=0)
        c_mean = c_mean / (np.linalg.norm(c_mean) + 1e-8)
        proto_m_list.append(c_mean.astype(np.float32))
    
    prototypes_malignant = np.stack(proto_m_list, axis=0) if len(proto_m_list) > 0 else np.zeros((Kp-1, 128), dtype=np.float32)
    
    benign_ids = [pid for pid, lab in patient_lbl.items() if lab == 0]
    vecs_b = [patient_sem_mean[pid] for pid in benign_ids if pid in patient_sem_mean]
    if len(vecs_b) > 0:
        vb = np.stack(vecs_b, axis=0)
        vb = vb / (np.linalg.norm(vb, axis=1, keepdims=True) + 1e-8)
        proto_b = vb.mean(axis=0)
        proto_b = proto_b / (np.linalg.norm(proto_b) + 1e-8)
    else:
        proto_b = np.zeros(128, dtype=np.float32)
    
    prototypes_benign = proto_b.reshape(1, -1).astype(np.float32)
    combined = np.vstack([prototypes_benign, prototypes_malignant]).astype(np.float32)
    
    return combined, prototypes_benign, prototypes_malignant


def add_gaussian_noise(prot, sigma=0.2, p=0.3):
    if np.random.rand() < p:
        noise = torch.randn_like(prot) * sigma
        return prot + noise
    return prot


def assert_disjoint_stagea_ids(train_dataset, val_dataset):
    train_ids = set(train_dataset.patient_df['ID'].astype(str).str.strip())
    val_ids = set(val_dataset.patient_df['ID'].astype(str).str.strip())
    overlap = train_ids.intersection(val_ids)
    if overlap:
        examples = sorted(overlap)[:10]
        raise ValueError(
            f"Stage A train/validation patient ID overlap detected: {examples}. "
            "Stage A split must be patient-disjoint."
        )


def train_stageA(args):
    set_seed(STAGEA_SEED)
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    weights_dir = Path(args.save_dir)
    output_dir = Path(args.output_dir)
    cluster_dir = data_dir / "stageA_cluster_data"
    
    proto_assign_path = cluster_dir / 'stageA_prototype_labels_all.csv'
    if not os.path.isfile(proto_assign_path):
        missing_proto_assign_path = proto_assign_path
        proto_assign_path = None
        print(f"[WARNING] Prototype assignment file not found: {missing_proto_assign_path}; supervision disabled")
    else:
        print(f"[Prototype supervision] loaded: {proto_assign_path}")
    
    dm = ThyroidDataModule(
        data_dir=str(data_dir),
        clinical_file='stageA_clinical.xlsx',
        protein_file='stageA_protein.xlsx',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fold=STAGEA_FOLD,
        n_folds=5,
        load_protein=True,
        use_grayscale=True,
        proto_assign_file=proto_assign_path,
        image_subdir=args.image_subdir,
    )
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    assert_disjoint_stagea_ids(dm.train_dataset, dm.val_dataset)
    print(f"[Data] train patients={len(dm.train_dataset.patient_df)} | train records={len(dm.train_dataset)}")
    print(f"[Data] internal val patients={len(dm.val_dataset.patient_df)} | internal val images={len(dm.val_dataset)}")
    
    default_assign = cluster_dir / 'stageA_prototype_labels_malignant.csv'
    assign_path = Path(args.cluster_assign_path) if args.cluster_assign_path else default_assign
    if not os.path.isfile(assign_path):
        raise FileNotFoundError(f"Cluster assignment file not found: {assign_path}")
    assign_df_meta = pd.read_csv(assign_path)
    uniq_c = sorted(assign_df_meta['cluster'].astype(int).unique().tolist())
    if uniq_c != [0, 1]:
        raise ValueError(f"ProPRINT public configuration expects malignant clusters [0, 1], got {uniq_c}")
    Kp = 3
    protein_dim = args.protein_dim if args.protein_dim is not None else dm.train_dataset.protein_dim
    if protein_dim != dm.train_dataset.protein_dim:
        raise ValueError(
            f"protein_dim={protein_dim} does not match the loaded protein matrix "
            f"dimension {dm.train_dataset.protein_dim}"
        )
    model = build_proprint_model(
        device=device,
        protein_dim=protein_dim,
        drop_path_rate=STAGEA_DROP_PATH,
        kappa=0.7,
        load_backbone_pretrained=True,
        K_prototypes=Kp
    )
    
    assign_df = pd.read_csv(assign_path)
    assign_df['ID'] = assign_df['ID'].astype(str).str.strip()
    assign_df['cluster'] = assign_df['cluster'].astype(int)
    
    combined, prototypes_benign, prototypes_malignant = compute_prototypes_from_data(
        model, train_loader, assign_df, device, Kp
    )
    model.img2rep.set_prototypes(combined)
    model.train()
    
    prototype_path = os.path.join(weights_dir, 'prototypes.npz')
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    np.savez(prototype_path, benign=prototypes_benign, malignant=prototypes_malignant)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    base_lr = args.lr
    
    scheduler = None
    
    scaler = GradScaler()
    
    out_dir = weights_dir
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'proprint_stageA.pth')
    
    best_auc = 0.0
    best_epoch = -1

    best_stageB_score = -1.0
    best_stageB_epoch = -1

    distill_started = False
    distill_start_epoch = None
    last_auc_real = None

    print(f"[StageA] start | epochs={args.epochs} | batch_size={args.batch_size}")
    print(f"[StageA] device={device} | seed={STAGEA_SEED} | checkpoint={save_path}")
    
    for epoch in range(args.epochs):
        weights = get_dynamic_weights_smart(epoch, args.epochs, distill_started, last_auc_real, distill_start_epoch)
        λv, λc, λd, λr = weights['λv'], weights['λc'], weights['λd'], weights['λr']
        λmmd = weights.get('λmmd', 0.0)
        λanchor = weights.get('λanchor', 0.0)
        λfused = weights.get('λfused', 0.0)
        λent = weights.get('λent', 0.0)
        λnce = weights.get('λnce', 0.0)
        stage_name = weights['stage']
        temp_sched = get_temperature_schedules(epoch, args.epochs)
        T_kd, tau_assign = temp_sched['T_kd'], temp_sched['tau_assign']
        model.img2rep.set_assignment_temperature(tau_assign)
        
        if weights['stage'] == 'teacher_build':
            for p in model.img2rep.parameters():
                p.requires_grad = False
        else:
            for p in model.img2rep.parameters():
                p.requires_grad = True
            if (not distill_started) and (weights['stage'] == 'semantic_distill_early'):
                distill_started = True
                distill_start_epoch = epoch
                prototypes_init, _, _ = compute_prototypes_from_data(
                    model, train_loader, assign_df, device, Kp
                )
                model.img2rep.set_prototypes(prototypes_init)
                model.train()
            elif distill_started and (epoch > distill_start_epoch) and ((epoch - distill_start_epoch) % 5 == 0):
                prototypes_new, _, _ = compute_prototypes_from_data(
                    model, train_loader, assign_df, device, Kp
                )
                with torch.no_grad():
                    beta = 0.7
                    new_t = torch.tensor(prototypes_new, dtype=torch.float32, device=device)
                    ema = beta * model.img2rep.prototypes.data + (1.0 - beta) * new_t
                    model.img2rep.prototypes.data = F.normalize(ema, p=2, dim=1)
                model.train()
        if epoch < args.warmup_epochs:
            warmup_factor = float(epoch + 1) / float(max(1, args.warmup_epochs))
            for g in optimizer.param_groups:
                g['lr'] = base_lr * warmup_factor
        else:
            if scheduler is None:
                for g in optimizer.param_groups:
                    g['lr'] = base_lr
                t_max = max(1, args.epochs - args.warmup_epochs)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=t_max, eta_min=base_lr * 0.1
                )
        
        model.train()
        running_loss = {'ce_real': 0.0, 'ce_virtual': 0.0,
                       'kl': 0.0, 'anchor': 0.0, 'fused': 0.0,
                       'entropy': 0.0, 'masked_recon': 0.0,
                       'proto_assign': 0.0, 'total': 0.0}
        for batch_idx, (images, proteins, labels, pids, proto_ids) in enumerate(train_loader):
            images = images.to(device)
            proteins = proteins.to(device)
            labels = labels.to(device)
            proto_ids = proto_ids.to(device)
            
            proteins = add_gaussian_noise(
                proteins,
                sigma=STAGEA_NOISE_SIGMA,
                p=STAGEA_P_NOISE_PROTEIN,
            )
            
            optimizer.zero_grad()
            
            with autocast(enabled=True):
                if stage_name == 'teacher_build':
                    logits_real, fused_cls, img_global, prot_sem, taps = model.forward_real_path(images, proteins)
                    loss_ce_real = F.cross_entropy(logits_real, labels)
                    loss_ce_virtual = torch.tensor(0.0, device=device)
                    loss_kl = torch.tensor(0.0, device=device)
                    loss_anchor = torch.tensor(0.0, device=device)
                    loss_fused = torch.tensor(0.0, device=device)
                    loss_entropy = torch.tensor(0.0, device=device)
                    loss_masked_recon = torch.tensor(0.0, device=device)
                    loss_proto_assign = torch.tensor(0.0, device=device)
                    loss = loss_ce_real
                else:
                    progress = (epoch + 1) / max(1, args.epochs)
                    mask_ratio_dyn = 0.5 + 0.4 * progress
                    full_mask_ratio = 0.0 + 0.7 * progress
                    use_full_mask = (torch.rand([]).item() < full_mask_ratio)
                    if use_full_mask:
                        mask_ratio_dyn = 1.0
                    logits_real, logits_virtual, fused_cls, fused_hat, prot_sem, prot_sem_hat, img_global, taps, aux = \
                        model(images, proteins, use_virtual_only=False, mask_ratio=mask_ratio_dyn, training=True)
                    
                    loss_ce_real = F.cross_entropy(logits_real, labels)
                    loss_ce_virtual = F.cross_entropy(logits_virtual, labels)
                    loss_kl = kl_divergence(logits_real.detach(), logits_virtual, T=T_kd)
                    
                    proto_b = model.img2rep.prototypes[0]
                    proto_m_all = model.img2rep.prototypes[1:]
                    proto_b = F.normalize(proto_b, p=2, dim=0)
                    proto_m_all = F.normalize(proto_m_all, p=2, dim=1)
                    loss_anchor = prototype_anchor_loss(
                        prot_sem.detach(), prot_sem_hat,
                        labels,
                        proto_benign=proto_b,
                        proto_malignant_all=proto_m_all,
                    )

                    fused_cos = F.cosine_similarity(fused_hat, fused_cls.detach(), dim=1)

                    benign_mask = (labels == 0)
                    malignant_mask = (labels == 1)

                    if benign_mask.any() and malignant_mask.any():
                        loss_fused_inst_b = (1.0 - fused_cos[benign_mask]).mean()
                        loss_fused_inst_m = (1.0 - fused_cos[malignant_mask]).mean()
                        gamma_m = 1.5
                        loss_fused_instance = (loss_fused_inst_b + gamma_m * loss_fused_inst_m) / (1.0 + gamma_m)
                    else:
                        loss_fused_instance = (1.0 - fused_cos).mean()

                    if benign_mask.any() and malignant_mask.any():
                        fused_real_b = fused_cls.detach()[benign_mask].mean(dim=0)
                        fused_real_m = fused_cls.detach()[malignant_mask].mean(dim=0)
                        fused_hat_b = fused_hat[benign_mask].mean(dim=0)
                        fused_hat_m = fused_hat[malignant_mask].mean(dim=0)

                        fused_real_b = F.normalize(fused_real_b, p=2, dim=0)
                        fused_real_m = F.normalize(fused_real_m, p=2, dim=0)
                        fused_hat_b = F.normalize(fused_hat_b, p=2, dim=0)
                        fused_hat_m = F.normalize(fused_hat_m, p=2, dim=0)

                        cos_b = F.cosine_similarity(fused_hat_b.unsqueeze(0), fused_real_b.unsqueeze(0)).mean()
                        cos_m = F.cosine_similarity(fused_hat_m.unsqueeze(0), fused_real_m.unsqueeze(0)).mean()

                        alpha_m = 0.6
                        alpha_b = 1.0 - alpha_m
                        loss_fused_class = 1.0 - (alpha_b * cos_b + alpha_m * cos_m)

                        loss_fused = 0.5 * loss_fused_instance + 0.5 * loss_fused_class
                    else:
                        loss_fused = loss_fused_instance
                    
                    assign_w = aux.get('assignment_weights')
                    assign_logits = aux.get('assignment_logits')
                    if assign_w is not None:
                        entropy = -(assign_w * (assign_w.clamp(min=1e-8).log())).sum(dim=1).mean()
                        loss_entropy = -entropy
                    else:
                        loss_entropy = torch.tensor(0.0, device=device)
                    
                    valid_proto_mask = (proto_ids >= 0)
                    if valid_proto_mask.any() and assign_logits is not None:
                        loss_proto_assign = F.cross_entropy(
                            assign_logits[valid_proto_mask], 
                            proto_ids[valid_proto_mask],
                            reduction='mean'
                        )
                    else:
                        loss_proto_assign = torch.tensor(0.0, device=device)
                    
                    rpr_recon_masked = aux.get('rpr_recon_masked')
                    mask_indices = aux.get('mask_indices')
                    
                    if rpr_recon_masked is not None and mask_indices is not None:
                        loss_masked_recon = 0.0
                        B = prot_sem.shape[0]
                        for i in range(B):
                            if len(mask_indices[i]) > 0:
                                masked_gt = prot_sem.detach()[i, mask_indices[i]]
                                masked_pred = rpr_recon_masked[i, mask_indices[i]]
                                loss_masked_recon += F.mse_loss(masked_pred, masked_gt)
                        loss_masked_recon = loss_masked_recon / B if B > 0 else torch.tensor(0.0, device=device)
                    else:
                        loss_masked_recon = torch.tensor(0.0, device=device)
                    
                    λmask = 3.0
                    λproto = 0.5
                    
                    loss = 1.0 * loss_ce_real + λv * loss_ce_virtual + λd * loss_kl + \
                           λanchor * loss_anchor + λfused * loss_fused + λent * loss_entropy + \
                           λmask * loss_masked_recon + λproto * loss_proto_assign
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss['ce_real'] += loss_ce_real.item()
            running_loss['ce_virtual'] += loss_ce_virtual.item()
            running_loss['kl'] += loss_kl.item()
            running_loss['anchor'] += float(loss_anchor.item())
            running_loss['fused'] += float(loss_fused.item())
            running_loss['entropy'] += float(loss_entropy.item())
            running_loss['masked_recon'] += float(loss_masked_recon.item())
            running_loss['proto_assign'] += float(loss_proto_assign.item())
            running_loss['total'] += loss.item()        
        if scheduler is not None:
            scheduler.step()
        
        n_batches = len(train_loader)
        avg_loss = {k: v / n_batches for k, v in running_loss.items()}
        
        model.eval()
        val_probs_real_by_pid = {}
        val_probs_virt_by_pid = {}
        val_label_by_pid = {}
        
        cka_sum = 0.0
        per_dim_r_all = []
        n_val_batches = 0
        ent_val_sum = 0.0
        
        with torch.no_grad():
            for images, proteins, labels, pids, proto_ids in val_loader:
                images = images.to(device)
                proteins = proteins.to(device)
                
                with autocast(enabled=True):
                    logits_real, logits_virtual, fused_cls, fused_hat, \
                    prot_sem, prot_sem_hat, img_global, taps, aux = \
                        model(images, proteins, use_virtual_only=False, training=False)
                
                probs_real = F.softmax(logits_real, dim=1)[:, 1].detach().cpu().numpy()
                
                probs_virt = F.softmax(logits_virtual, dim=1)[:, 1].detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                for i, pid in enumerate([str(x) for x in pids]):
                    val_probs_real_by_pid.setdefault(pid, []).append(float(probs_real[i]))
                    val_probs_virt_by_pid.setdefault(pid, []).append(float(probs_virt[i]))
                    val_label_by_pid[pid] = int(labels_np[i])
                
                cka = compute_cka(prot_sem, prot_sem_hat)
                cka_sum += cka
                
                per_dim_r = compute_per_dim_correlation(prot_sem, prot_sem_hat)
                per_dim_r_all.append(per_dim_r)
                
                val_assign_w = aux.get('assignment_weights')
                if val_assign_w is not None:
                    entropy_val = -(val_assign_w * (val_assign_w.clamp(min=1e-8).log())).sum(dim=1).mean().item()
                    ent_val_sum += entropy_val
                
                n_val_batches += 1
        
        patient_ids_val = sorted(val_label_by_pid.keys())
        val_probs_real = [max(val_probs_real_by_pid[pid]) for pid in patient_ids_val]
        val_probs_virt = [max(val_probs_virt_by_pid[pid]) for pid in patient_ids_val]
        val_labels = [val_label_by_pid[pid] for pid in patient_ids_val]

        auc_real = binary_roc_auc(val_probs_real, val_labels)
        auc_virt = binary_roc_auc(val_probs_virt, val_labels)
        auc = auc_real
        stageB_score = None
        if (stage_name != 'teacher_build') and (auc_real >= 0.92):
            geom_score = 1.0 / (1.0 + max(avg_loss['fused'], 0.0) + max(avg_loss['anchor'], 0.0))
            stageB_score = auc_virt + geom_score
        last_auc_real = auc_real
        
        avg_cka = cka_sum / max(1, n_val_batches)
        per_dim_r_concat = np.concatenate(per_dim_r_all, axis=0) if per_dim_r_all else np.array([])
        per_dim_r_mean = per_dim_r_concat.mean() if len(per_dim_r_concat) > 0 else 0.0
        per_dim_r_std = per_dim_r_concat.std() if len(per_dim_r_concat) > 0 else 0.0
        per_dim_r_min = per_dim_r_concat.min() if len(per_dim_r_concat) > 0 else 0.0
        per_dim_r_max = per_dim_r_concat.max() if len(per_dim_r_concat) > 0 else 0.0
        ent_val_mean = ent_val_sum / max(1, n_val_batches)
        
        current_lr = optimizer.param_groups[0]['lr']
        with torch.no_grad():
            proto = model.img2rep.prototypes.detach()
            proto_norm_mean = proto.norm(dim=1).mean().item()
            proto_n = F.normalize(proto, p=2, dim=1)
            Kp = proto.shape[0]
            pairwise = torch.matmul(proto_n, proto_n.t())
            mask = torch.ones(Kp, Kp, device=proto.device) - torch.eye(Kp, device=proto.device)
            min_offdiag = (pairwise * mask).masked_select(mask.bool()).min().item()
        train_assign_entropy_mean = -avg_loss['entropy']
        print(f"[StageA] Epoch {epoch+1}/{args.epochs} | Stage: {stage_name} | LR: {current_lr:.6f}")
        print(f"  Loss: total={avg_loss['total']:.4f} | CE_real={avg_loss['ce_real']:.4f} | "
              f"CE_virt={avg_loss['ce_virtual']:.4f} | KL={avg_loss['kl']:.4f}")
        print(f"        Anchor={avg_loss['anchor']:.4f} | Fused={avg_loss['fused']:.4f} | Ent={avg_loss['entropy']:.4f}")
        print(f"        MaskedRPRRecon={avg_loss['masked_recon']:.4f} | ProtoAssign={avg_loss['proto_assign']:.4f}")
        print(f"  Weights: λv={λv:.3f} | λd={λd:.3f} | λanchor={λanchor:.2f} | λfused={λfused:.2f} | λent={λent:.3f}")
        print(f"  Temp: T_kd={T_kd:.2f} | τ_assign={tau_assign:.2f}")
        print(f"  Val AUC: Real={auc_real:.4f} | Virt={auc_virt:.4f} | CKA={avg_cka:.4f} (monitor only)")
        print(f"  Per-dim corr: mean={per_dim_r_mean:.4f} | std={per_dim_r_std:.4f} | "
              f"range=[{per_dim_r_min:.4f}, {per_dim_r_max:.4f}]")
        print(f"  Proto: norm_mean={proto_norm_mean:.3f} | pairwise_cos_min(offdiag)={min_offdiag:.3f}")
        print(f"  Assign Ent: train={train_assign_entropy_mean:.3f} | val={ent_val_mean:.3f}")
        if stageB_score is not None:
            print(f"  StageB_score={stageB_score:.4f} (AUC_virt+geom)")

        if auc_real > best_auc:
            best_auc = auc_real
            best_epoch = epoch
            if best_stageB_epoch < 0:
                torch.save(model.state_dict(), save_path)
                print(f"  Saved Stage A checkpoint | Epoch={epoch+1} | AUC_real={auc_real:.4f}")
        
        if stageB_score is not None and stageB_score > best_stageB_score:
            best_stageB_score = stageB_score
            best_stageB_epoch = epoch
            torch.save(model.state_dict(), save_path)
            print(f"  Saved Stage-B initialization checkpoint | Epoch={epoch+1} | AUC_real={auc_real:.4f} | AUC_virt={auc_virt:.4f}")
    
    if not os.path.isfile(save_path):
        torch.save(model.state_dict(), save_path)
        print(f"[StageA] saved fallback checkpoint: {save_path}")
    
    print(f"\n[StageA] finished | Best StageB Epoch: {best_stageB_epoch+1 if best_stageB_epoch>=0 else -1} | Best StageB Score: {best_stageB_score:.4f}")
    print("[StageA] saving semantic prototypes...")

    with torch.no_grad():
        proto = model.img2rep.prototypes.detach().cpu().numpy()
    prototypes_benign = proto[:1]
    prototypes_malignant = proto[1:]
    prototype_path = os.path.join(out_dir, 'prototypes.npz')
    np.savez(prototype_path,
             benign=prototypes_benign,
             malignant=prototypes_malignant)

    print(f"[StageA] prototypes saved: {prototype_path}")
    print(f"  benign prototype shape: {prototypes_benign.shape} | malignant prototype shape: {prototypes_malignant.shape}")

    return {
        'best_auc': best_auc,
        'best_epoch': best_epoch,
        'best_stageB_score': best_stageB_score,
        'best_stageB_epoch': best_stageB_epoch,
        'save_path': save_path,
        'prototype_path': prototype_path
    }


def main():
    parser = argparse.ArgumentParser(description='ProPRINT Stage A training')
    
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='auto', help='Device, e.g. auto/cuda:0/cpu')

    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR), help='Directory containing Stage A data files')
    parser.add_argument('--image_subdir', type=str, default='stageA_ultrasound', help='Stage A image folder under data_dir')
    parser.add_argument('--cluster_assign_path', type=str, default=None, help='Malignant prototype label CSV')
    parser.add_argument('--save_dir', type=str, default=str(WEIGHTS_DIR), help='Checkpoint output directory')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUTS_DIR), help='Output directory')
    parser.add_argument('--protein_dim', type=int, default=None, help='Protein feature dimension; inferred from data when omitted')
    
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs')

    args = parser.parse_args()
    
    print("=" * 80)
    print("ProPRINT - Stage A training")
    print("=" * 80)
    print(f"Device: {args.device} | seed: {STAGEA_SEED}")
    print(f"Data: {Path(args.data_dir)}")
    print(f"Arguments: {vars(args)}")
    print("=" * 80)
    
    result = train_stageA(args)
    
    print("\n" + "=" * 80)
    print("Training finished.")
    print(f"Best validation AUC: {result['best_auc']:.4f} (Epoch {result['best_epoch']+1})" if result['best_epoch']>=0 else "Best validation AUC: N/A")
    if 'best_stageB_epoch' in result and result['best_stageB_epoch']>=0:
        print(f"Best Stage B initialization epoch: {result['best_stageB_epoch']+1} | Score={result['best_stageB_score']:.4f}")
    print(f"Checkpoint: {result['save_path']}")
    print(f"Prototypes: {result['prototype_path']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
