"""Evaluation helpers for ProPRINT probability outputs."""

import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.cuda.amp import autocast


def binary_roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).astype(np.float64)
    equal = (pos[:, None] == neg[None, :]).astype(np.float64)
    return float((greater.sum() + 0.5 * equal.sum()) / (pos.size * neg.size))


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    labels_arr = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels_arr)) < 2:
        return float("nan")
    return float(average_precision_score(labels_arr, np.asarray(scores, dtype=np.float64)))


def probability_metrics(scores: Sequence[float], labels: Sequence[int]) -> dict:
    return {
        "auc": binary_roc_auc(scores, labels),
        "ap": average_precision(scores, labels),
    }


def aggregate_probs(prob_list: Sequence[float], aggregation: str = "max") -> float:
    if len(prob_list) == 0:
        return float("nan")
    if aggregation == "max":
        return float(np.max(prob_list))
    if aggregation == "top3":
        top3 = sorted(prob_list, reverse=True)[:min(3, len(prob_list))]
        return float(np.mean(top3))
    if aggregation == "mean":
        return float(np.mean(prob_list))
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def patient_level_aggregation(
    image_probs: Sequence[float],
    image_labels: Sequence[int],
    image_pids: Sequence[str],
    aggregation: str = "max",
) -> Tuple[List[float], List[int], List[str]]:
    patient_dict = {}
    for prob, label, pid in zip(image_probs, image_labels, image_pids):
        pid = str(pid)
        patient_dict.setdefault(pid, {"probs": [], "label": int(label)})
        patient_dict[pid]["probs"].append(float(prob))

    patient_ids = sorted(patient_dict.keys())
    patient_probs = [aggregate_probs(patient_dict[pid]["probs"], aggregation) for pid in patient_ids]
    patient_labels = [patient_dict[pid]["label"] for pid in patient_ids]
    return patient_probs, patient_labels, patient_ids


def _iter_tta_views(images: torch.Tensor, tta_views: int):
    yield images
    if tta_views >= 2:
        yield torch.flip(images, dims=[3])
    if tta_views >= 3:
        yield torch.flip(images, dims=[2])
    if tta_views >= 4:
        yield torch.flip(torch.flip(images, dims=[3]), dims=[2])


def predict_probabilities_with_tta(
    model,
    loader: Iterable,
    device: torch.device,
    tta_views: int = 4,
    aggregation: str = "max",
):
    model.eval()
    image_probs = []
    image_labels = []
    image_pids = []
    image_count = 0

    with torch.no_grad():
        for batch in loader:
            images, _proteins, labels, pids = batch[:4]
            images = images.to(device)
            image_count += int(images.shape[0])

            with autocast(enabled=True):
                logits_sum = None
                count = 0
                for view in _iter_tta_views(images, tta_views):
                    logits, _, _, _, _, _ = model(view, prot=None, use_virtual_only=True, training=False)
                    logits_sum = logits if logits_sum is None else logits_sum + logits
                    count += 1
                logits = logits_sum / float(count)
                probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

            image_probs.extend([float(x) for x in probs])
            image_labels.extend([int(x) for x in labels.detach().cpu().numpy()])
            image_pids.extend([str(pid) for pid in pids])

    patient_probs, patient_labels, patient_ids = patient_level_aggregation(
        image_probs,
        image_labels,
        image_pids,
        aggregation=aggregation,
    )
    return {
        "patient_probs": patient_probs,
        "patient_labels": patient_labels,
        "patient_ids": patient_ids,
        "image_probs": image_probs,
        "image_labels": image_labels,
        "image_pids": image_pids,
        "image_count": image_count,
        "patient_count": len(patient_ids),
        "metrics": probability_metrics(patient_probs, patient_labels),
    }


def export_probability_table(
    patient_ids: Sequence[str],
    patient_probs: Sequence[float],
    save_path: str,
    labels: Sequence[int] = None,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame({
        "ID": [str(pid) for pid in patient_ids],
        "proprint_prob_malignant": [float(prob) for prob in patient_probs],
    })
    if labels is not None:
        df["Label"] = [int(label) for label in labels]
    if save_path.lower().endswith((".xlsx", ".xls")):
        df.to_excel(save_path, index=False)
    else:
        df.to_csv(save_path, index=False)
    return df
