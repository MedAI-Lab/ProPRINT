"""ProPRINT inference runner."""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = CODE_ROOT
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "proprint"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "proprint"
if str(CODE_ROOT) in sys.path:
    sys.path.remove(str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT))

from proprint.model_proprint import build_proprint_model
from proprint.utils_eval import probability_metrics, predict_probabilities_with_tta
from utils.dataset import build_transforms


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


class ProPRINTInferenceDataset(Dataset):
    """Image-level inference dataset with patient IDs retained for aggregation."""

    def __init__(
        self,
        data_dir: Path,
        clinical_file: str,
        image_subdir: str,
        transform,
        id_col: str = "ID",
        label_col: str = "Label",
        image_exts: str = "jpg,jpeg,png,bmp,tif,tiff",
        protein_dim: int = 680,
    ):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / image_subdir
        self.transform = transform
        self.id_col = id_col
        self.label_col = label_col
        self.protein_dim = int(protein_dim)
        self.image_exts = [x.strip().lower().lstrip(".") for x in image_exts.split(",") if x.strip()]

        clinical_path = self.data_dir / clinical_file
        if not clinical_path.is_file():
            raise FileNotFoundError(f"Clinical file not found: {clinical_path}")
        if clinical_path.suffix.lower() == ".csv":
            self.patient_df = pd.read_csv(clinical_path)
        else:
            self.patient_df = pd.read_excel(clinical_path)
        if id_col not in self.patient_df.columns:
            raise ValueError(f"Clinical file must contain ID column: {id_col}")
        self.patient_df[id_col] = self.patient_df[id_col].astype(str).str.strip()

        self.has_labels = label_col in self.patient_df.columns
        self.patient_image_count = {}
        self.records = []
        for _, row in self.patient_df.iterrows():
            pid = str(row[id_col]).strip()
            label = -1
            if self.has_labels and not pd.isna(row[label_col]):
                label = int(row[label_col])
            paths = self._get_image_paths(pid)
            self.patient_image_count[pid] = len(paths)
            if not paths:
                self.records.append((None, pid, label))
                continue
            for path in paths:
                self.records.append((path, pid, label))
        self.image_file_count = int(sum(self.patient_image_count.values()))

    def _get_image_paths(self, patient_id: str):
        paths = []
        for ext in self.image_exts:
            paths.extend(glob.glob(str(self.image_dir / f"{patient_id}_*.{ext}")))
            paths.extend(glob.glob(str(self.image_dir / f"{patient_id}*.{ext}")))
        return sorted(set(paths))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        image_path, patient_id, label = self.records[idx]
        if image_path is None:
            image = torch.zeros(3, 224, 224)
        else:
            image = Image.open(image_path).convert("L")
            image = self.transform(image)
        protein = torch.zeros(self.protein_dim, dtype=torch.float32)
        return image, protein, int(label), patient_id


def load_proprint_model(args, device):
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if not os.path.isfile(args.prototype_path):
        raise FileNotFoundError(f"Prototype file not found: {args.prototype_path}")

    prototypes = np.load(args.prototype_path)
    proto_benign = prototypes["benign"]
    proto_malignant = prototypes["malignant"]
    if proto_benign.ndim == 1:
        proto_benign = proto_benign.reshape(1, -1)
    if proto_malignant.ndim == 1:
        proto_malignant = proto_malignant.reshape(1, -1)
    if proto_benign.shape[0] != 1 or proto_malignant.shape[0] != 2:
        raise ValueError(
            "ProPRINT public configuration expects 1 benign + 2 malignant prototypes, "
            f"got benign={proto_benign.shape}, malignant={proto_malignant.shape}"
        )

    model = build_proprint_model(
        device=device,
        protein_dim=args.protein_dim,
        drop_path_rate=args.drop_path,
        kappa=args.kappa_final,
        load_backbone_pretrained=False,
        K_prototypes=3,
        use_masked_refine=True,
    )
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.img2rep.set_prototypes(np.vstack([proto_benign, proto_malignant]).astype(np.float32))
    model.img2rep.set_assignment_temperature(args.tau_final)
    model.set_kappa(args.kappa_final)
    model.eval()
    return model


def export_inference_outputs(result: dict, dataset: ProPRINTInferenceDataset, args):
    os.makedirs(args.output_dir, exist_ok=True)

    prob_by_id = {pid: prob for pid, prob in zip(result["patient_ids"], result["patient_probs"])}
    rows = []
    for _, row in dataset.patient_df.iterrows():
        pid = str(row[args.id_col]).strip()
        out = row.to_dict()
        out["proprint_prob_malignant"] = prob_by_id.get(pid, float("nan"))
        out["image_count"] = int(dataset.patient_image_count.get(pid, 0))
        rows.append(out)

    out_df = pd.DataFrame(rows)
    output_base = Path(args.output_dir) / args.output_prefix
    prob_path = str(output_base.with_suffix(".xlsx"))
    out_df.to_excel(prob_path, index=False)

    labeled = [
        (prob, label, pid)
        for prob, label, pid in zip(result["patient_probs"], result["patient_labels"], result["patient_ids"])
        if int(label) >= 0 and np.isfinite(float(prob))
    ]
    metrics_payload = {
        "model_path": str(args.model_path),
        "prototype_path": str(args.prototype_path),
        "data_dir": str(args.data_dir),
        "clinical_file": args.clinical_file,
        "image_subdir": args.image_subdir,
        "tta_views": int(args.tta),
        "aggregation": args.aggregation,
        "patient_count": int(len(dataset.patient_df)),
        "image_file_count": int(dataset.image_file_count),
        "inference_record_count": int(result["image_count"]),
        "probability_file": prob_path,
    }
    if labeled:
        labeled_probs, labeled_labels, _labeled_ids = zip(*labeled)
        metrics_payload.update(probability_metrics(labeled_probs, labeled_labels))
        metrics_payload["labeled_patient_count"] = int(len(labeled_labels))
    else:
        metrics_payload.update({"auc": float("nan"), "ap": float("nan"), "labeled_patient_count": 0})

    metrics_path = str(output_base.with_name(output_base.name + "_metrics").with_suffix(".json"))
    pd.Series(metrics_payload).to_json(metrics_path, indent=2, force_ascii=False)
    return prob_path, metrics_path, metrics_payload


def main():
    parser = argparse.ArgumentParser(description="ProPRINT locked-checkpoint inference")
    parser.add_argument("--device", type=str, default="auto", help="Device, e.g. auto/cuda:0/cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")

    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Directory containing clinical file and images")
    parser.add_argument("--clinical_file", type=str, default="clinical_prospective.xlsx", help="Clinical CSV/XLSX file")
    parser.add_argument("--image_subdir", type=str, default="inference_ultrasound", help="Image folder under data_dir")
    parser.add_argument("--id_col", type=str, default="ID", help="Patient ID column")
    parser.add_argument("--label_col", type=str, default="Label", help="Optional label column")
    parser.add_argument("--image_exts", type=str, default="jpg,jpeg,png,bmp,tif,tiff", help="Comma-separated image extensions")

    parser.add_argument("--model_path", type=str, default=str(WEIGHTS_DIR / "proprint_stageB_best.pth"), help="Stage B checkpoint")
    parser.add_argument("--prototype_path", type=str, default=str(WEIGHTS_DIR / "prototypes.npz"), help="Prototype npz file")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR), help="Output directory")
    parser.add_argument("--output_prefix", type=str, default="proprint_inference_predictions", help="Output file prefix")

    parser.add_argument("--drop_path", type=float, default=0.2, help="DropPath rate used by the checkpoint")
    parser.add_argument("--protein_dim", type=int, default=680, help="Protein feature dimension used by the checkpoint")
    parser.add_argument("--kappa_final", type=float, default=0.6, help="Final FiLM gate scale")
    parser.add_argument("--tau_final", type=float, default=1.3, help="Final prototype assignment temperature")
    parser.add_argument("--tta", type=int, default=4, help="Number of TTA views")
    parser.add_argument("--aggregation", type=str, default="max", choices=["mean", "max", "top3"], help="Patient-level aggregation")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)

    transform = build_transforms(
        use_grayscale=True,
        training=False,
    )
    dataset = ProPRINTInferenceDataset(
        data_dir=Path(args.data_dir),
        clinical_file=args.clinical_file,
        image_subdir=args.image_subdir,
        transform=transform,
        id_col=args.id_col,
        label_col=args.label_col,
        image_exts=args.image_exts,
        protein_dim=args.protein_dim,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("=" * 80)
    print("ProPRINT locked-checkpoint inference")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Prototype: {args.prototype_path}")
    print(f"Data: {Path(args.data_dir) / args.clinical_file}")
    print(f"Images: {Path(args.data_dir) / args.image_subdir}")
    print(f"Patients: {len(dataset.patient_df)} | Image files: {dataset.image_file_count} | Records: {len(dataset)}")
    print(f"TTA: {args.tta} | Aggregation: {args.aggregation} | Device: {device}")
    print("=" * 80)

    model = load_proprint_model(args, device)
    result = predict_probabilities_with_tta(
        model,
        loader,
        device,
        tta_views=args.tta,
        aggregation=args.aggregation,
    )

    prob_path, metrics_path, metrics_payload = export_inference_outputs(result, dataset, args)
    print(f"[Metrics] patients={metrics_payload['patient_count']} | image_files={metrics_payload['image_file_count']} | "
          f"records={metrics_payload['inference_record_count']} | "
          f"labeled={metrics_payload['labeled_patient_count']} | AUC={metrics_payload['auc']:.4f} | AP={metrics_payload['ap']:.4f}")
    print(f"[Output] probabilities: {prob_path}")
    print(f"[Output] metrics: {metrics_path}")


if __name__ == "__main__":
    main()
