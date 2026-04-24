"""Experiment 1: classification metrics on the stratified test split.

Outputs under ``benchmark/report/``:

- ``metrics.json``              : top-1, top-3, macro/weighted F1, counts.
- ``per_class_metrics.csv``     : precision / recall / F1 / support per class.
- ``confusion_matrix.png``      : counts + row-normalised heatmaps.
- ``confidence_hist.png``       : max-softmax histogram, split by correct/wrong.
- ``predictions.npz``           : cached probs / labels / paths, reused
                                    by threshold_analysis.py to avoid a
                                    second inference pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark.common import (
    DEFAULT_MODEL_PATH,
    ManifestDataset,
    SPLITS_DIR,
    ensure_report_dir,
    load_model,
    read_manifest,
)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_paths: List[str] = []
    for batch in tqdm(loader, desc="inference"):
        tensors, labels, paths = batch
        tensors = tensors.to(device, non_blocking=True)
        logits = model(tensors)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(np.asarray(labels))
        all_paths.extend(paths)
    return np.concatenate(all_probs), np.concatenate(all_labels), all_paths


def plot_confusion(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm_norm = cm.astype(np.float64)
    row_sum = cm_norm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    cm_norm = cm_norm / row_sum

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=axes[0],
        cbar=False,
    )
    axes[0].set_title("Confusion matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
        xticklabels=class_names, yticklabels=class_names, ax=axes[1],
        cbar=True,
    )
    axes[1].set_title("Confusion matrix (row-normalised)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confidence_hist(
    max_probs: np.ndarray,
    correct: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0.0, 1.0, 41)
    ax.hist(max_probs[correct], bins=bins, alpha=0.65, label="correct", color="#2a9d8f")
    ax.hist(max_probs[~correct], bins=bins, alpha=0.65, label="wrong", color="#e76f51")
    ax.axvline(0.5, linestyle="--", color="gray", label="backend threshold (0.5)")
    ax.set_xlabel("Max softmax confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence distribution on test split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SPLITS_DIR / "test_manifest.csv")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
    )

    device = torch.device(args.device) if args.device else None
    model, class_names, device = load_model(args.model, device=device)
    print(f"[evaluate_cv] device={device}, classes={class_names}")

    rows = read_manifest(args.manifest)
    ds = ManifestDataset(rows)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    probs, labels, paths = run_inference(model, loader, device)
    preds = probs.argmax(axis=1)
    top1 = float((preds == labels).mean())

    top3_preds = np.argsort(-probs, axis=1)[:, :3]
    top3 = float(np.mean([labels[i] in top3_preds[i] for i in range(len(labels))]))

    macro_f1 = float(f1_score(labels, preds, average="macro"))
    weighted_f1 = float(f1_score(labels, preds, average="weighted"))

    report_dir = ensure_report_dir()
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    plot_confusion(cm, class_names, report_dir / "confusion_matrix.png")

    max_probs = probs.max(axis=1)
    correct = preds == labels
    plot_confidence_hist(max_probs, correct, report_dir / "confidence_hist.png")

    report_dict = classification_report(
        labels, preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    # Flatten per-class metrics to CSV.
    per_class_path = report_dir / "per_class_metrics.csv"
    with per_class_path.open("w", encoding="utf-8") as f:
        f.write("class,precision,recall,f1,support\n")
        for cls in class_names:
            d = report_dict[cls]
            f.write(f"{cls},{d['precision']:.4f},{d['recall']:.4f},"
                    f"{d['f1-score']:.4f},{int(d['support'])}\n")

    metrics = {
        "n_samples": int(len(labels)),
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "class_names": class_names,
        "per_class": {
            cls: {
                "precision": float(report_dict[cls]["precision"]),
                "recall": float(report_dict[cls]["recall"]),
                "f1": float(report_dict[cls]["f1-score"]),
                "support": int(report_dict[cls]["support"]),
            }
            for cls in class_names
        },
        "confusion_matrix": cm.tolist(),
        "backend_threshold_default": 0.5,
        "confidence_stats": {
            "mean": float(max_probs.mean()),
            "p05": float(np.percentile(max_probs, 5)),
            "p50": float(np.percentile(max_probs, 50)),
            "p95": float(np.percentile(max_probs, 95)),
        },
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Cache probabilities for downstream experiments.
    np.savez(
        report_dir / "predictions.npz",
        probs=probs,
        labels=labels,
        paths=np.array(paths),
        class_names=np.array(class_names),
    )

    print(f"[evaluate_cv] top1={top1:.4f} top3={top3:.4f} "
          f"macro_f1={macro_f1:.4f} weighted_f1={weighted_f1:.4f}")
    print(f"[evaluate_cv] report dir: {report_dir}")


if __name__ == "__main__":
    main()
