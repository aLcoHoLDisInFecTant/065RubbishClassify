"""Experiment 2: confidence threshold sensitivity.

Reuses cached probabilities from ``evaluate_cv.py`` when available, so
you don't need a second inference pass.

Outputs under ``benchmark/report/``:

- ``threshold_table.csv``       : coverage / selective accuracy / risk per threshold.
- ``threshold_tradeoff.png``    : coverage vs selective-accuracy curve.
- ``risk_coverage.png``         : risk-coverage curve (AURC).
- ``roc_pr_curves.png``         : one-vs-rest ROC + PR curves per class.
- ``threshold_summary.json``    : key numbers (AURC, recommended threshold, AUCs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmark.common import (
    DEFAULT_MODEL_PATH,
    ManifestDataset,
    SPLITS_DIR,
    ensure_report_dir,
    load_model,
    read_manifest,
)


def load_or_infer(
    cache_path: Path,
    manifest: Path,
    model_path: Path,
    batch_size: int,
    num_workers: int,
    device_str: str | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        print(f"[threshold] reusing cached predictions from {cache_path}")
        return data["probs"], data["labels"], list(data["class_names"])

    print("[threshold] no cached predictions found, running inference")
    device = torch.device(device_str) if device_str else None
    model, class_names, device = load_model(model_path, device=device)
    rows = read_manifest(manifest)
    ds = ManifestDataset(rows)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=(device.type == "cuda"))
    probs, labels = [], []
    from tqdm import tqdm
    with torch.no_grad():
        for tensors, y, _paths in tqdm(loader, desc="inference"):
            tensors = tensors.to(device, non_blocking=True)
            p = torch.softmax(model(tensors), dim=1).cpu().numpy()
            probs.append(p)
            labels.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(labels), class_names


def sweep_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: List[float],
) -> list[dict]:
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    n = len(labels)
    rows = []
    for t in thresholds:
        accepted = conf >= t
        n_acc = int(accepted.sum())
        coverage = n_acc / n if n else 0.0
        if n_acc:
            sel_acc = float((preds[accepted] == labels[accepted]).mean())
        else:
            sel_acc = float("nan")
        overall_acc = float(
            ((preds == labels) & accepted).sum() / n
        ) if n else 0.0
        rejection_rate = 1.0 - coverage
        rows.append({
            "threshold": t,
            "coverage": coverage,
            "selective_accuracy": sel_acc,
            "risk": (1.0 - sel_acc) if n_acc else 0.0,
            "overall_accuracy_including_rejected": overall_acc,
            "rejection_rate": rejection_rate,
            "n_accepted": n_acc,
        })
    return rows


def risk_coverage_curve(probs: np.ndarray, labels: np.ndarray):
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    order = np.argsort(-conf)  # most-confident first
    correct = (preds == labels)[order]
    cum_correct = np.cumsum(correct)
    k = np.arange(1, len(labels) + 1)
    coverage = k / len(labels)
    risk = 1.0 - cum_correct / k
    aurc = float(np.trapz(risk, coverage))
    return coverage, risk, aurc


def recommend_threshold(table: list[dict], target_sel_acc: float) -> float | None:
    """Smallest threshold whose selective accuracy >= target."""
    candidates = [r for r in table
                  if not np.isnan(r["selective_accuracy"])
                  and r["selective_accuracy"] >= target_sel_acc]
    if not candidates:
        return None
    return min(r["threshold"] for r in candidates)


def plot_threshold_tradeoff(table: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    ts = [r["threshold"] for r in table]
    cov = [r["coverage"] for r in table]
    sel = [r["selective_accuracy"] for r in table]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(ts, cov, "o-", color="#264653", label="Coverage")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Coverage", color="#264653")
    ax1.tick_params(axis="y", labelcolor="#264653")
    ax1.set_ylim(0, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(ts, sel, "s--", color="#e76f51", label="Selective accuracy")
    ax2.set_ylabel("Selective accuracy", color="#e76f51")
    ax2.tick_params(axis="y", labelcolor="#e76f51")
    ax2.set_ylim(0, 1.02)
    ax1.axvline(0.5, linestyle=":", color="gray")
    ax1.set_title("Threshold trade-off: coverage vs selective accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_risk_coverage(coverage, risk, aurc: float, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(coverage, risk, color="#2a9d8f")
    ax.fill_between(coverage, risk, alpha=0.2, color="#2a9d8f")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk (1 - selective accuracy)")
    ax.set_title(f"Risk-Coverage curve  (AURC = {aurc:.4f})")
    ax.set_ylim(0, max(0.05, float(risk.max()) * 1.1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_pr(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> dict:
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    aucs, aps = {}, {}
    for i, cls in enumerate(class_names):
        y_true = (labels == i).astype(int)
        y_score = probs[:, i]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            aucs[cls] = float("nan")
            aps[cls] = float("nan")
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = float(roc_auc_score(y_true, y_score))
        axes[0].plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})")
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = float(average_precision_score(y_true, y_score))
        axes[1].plot(recall, precision, label=f"{cls} (AP={ap:.3f})")
        aucs[cls] = auc
        aps[cls] = ap

    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC (one-vs-rest)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall (one-vs-rest)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"roc_auc": aucs, "average_precision": aps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SPLITS_DIR / "test_manifest.csv")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--target-sel-acc", type=float, default=0.97)
    args = parser.parse_args()

    report_dir = ensure_report_dir()
    cache_path = report_dir / "predictions.npz"
    probs, labels, class_names = load_or_infer(
        cache_path, args.manifest, args.model,
        args.batch_size, args.num_workers, args.device,
    )

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    table = sweep_thresholds(probs, labels, thresholds)

    csv_path = report_dir / "threshold_table.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("threshold,coverage,selective_accuracy,risk,"
                "overall_accuracy_including_rejected,rejection_rate,n_accepted\n")
        for r in table:
            f.write(
                f"{r['threshold']:.2f},{r['coverage']:.4f},"
                f"{r['selective_accuracy']:.4f},{r['risk']:.4f},"
                f"{r['overall_accuracy_including_rejected']:.4f},"
                f"{r['rejection_rate']:.4f},{r['n_accepted']}\n"
            )

    plot_threshold_tradeoff(table, report_dir / "threshold_tradeoff.png")

    cov_arr, risk_arr, aurc = risk_coverage_curve(probs, labels)
    plot_risk_coverage(cov_arr, risk_arr, aurc, report_dir / "risk_coverage.png")

    auc_info = plot_roc_pr(probs, labels, class_names, report_dir / "roc_pr_curves.png")

    recommended = recommend_threshold(table, args.target_sel_acc)

    summary = {
        "thresholds": thresholds,
        "table": table,
        "aurc": aurc,
        "target_selective_accuracy": args.target_sel_acc,
        "recommended_threshold": recommended,
        "per_class_auc": auc_info,
    }
    (report_dir / "threshold_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[threshold] AURC={aurc:.4f}  recommended_threshold(@{args.target_sel_acc})="
          f"{recommended}")


if __name__ == "__main__":
    main()
