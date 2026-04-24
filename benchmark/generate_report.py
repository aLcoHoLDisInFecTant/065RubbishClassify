"""Aggregate all experiment outputs into ``benchmark/report/REPORT.md``.

With ``--run-all`` this also runs the four experiments end-to-end. By
default it only consumes whatever JSON/CSV/PNG files already exist in
``benchmark/report/``; each experiment is reported independently, so
missing ones produce a "not available" note.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from benchmark.common import SPLITS_DIR, ensure_report_dir

ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[report] failed to parse {path}: {e}")
        return None


def fmt_pct(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x * 100:.2f}%"


def fmt_float(x: Optional[float], digits: int = 4) -> str:
    return "N/A" if x is None else f"{x:.{digits}f}"


def section_overview(report_dir: Path, metrics: Optional[dict],
                     latency: Optional[dict]) -> str:
    env = (latency or {}).get("env") if latency else None
    lines = ["## 1. Overview", ""]
    lines.append(f"- Report generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Report directory: `{report_dir}`")
    if metrics:
        lines.append(f"- Test samples: **{metrics.get('n_samples', 'N/A')}**")
        lines.append(f"- Classes: `{metrics.get('class_names')}`")
    if env:
        lines.append(f"- Python: `{env.get('python')}`  |  Platform: `{env.get('platform')}`")
        lines.append(f"- PyTorch: `{env.get('torch')}`  |  CUDA: `{env.get('cuda_available')}`"
                     f"  |  GPU: `{env.get('cuda_device')}`")
    lines.append("")
    lines.append("**Data caveat.** The original training script used `random_split` without "
                 "a fixed seed, so the exact validation split is unrecoverable. The evaluation "
                 "set in this report is a fresh stratified sample (`seed=2026`, 15%) and may "
                 "overlap with training data; numbers therefore represent an upper-bound "
                 "estimate of in-distribution performance rather than a true held-out test. "
                 "Consider adding an independently collected field-test set for a stricter "
                 "evaluation.")
    lines.append("")
    return "\n".join(lines)


def section_accuracy(metrics: Optional[dict]) -> str:
    if not metrics:
        return "## 2. Classification accuracy\n\n_Run `evaluate_cv.py` first._\n"
    lines = [
        "## 2. Classification accuracy",
        "",
        f"- Top-1 accuracy: **{fmt_pct(metrics['top1_accuracy'])}**",
        f"- Top-3 accuracy: **{fmt_pct(metrics['top3_accuracy'])}**",
        f"- Macro-F1: **{fmt_float(metrics['macro_f1'])}**",
        f"- Weighted-F1: **{fmt_float(metrics['weighted_f1'])}**",
        "",
        "### Per-class metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for cls in metrics["class_names"]:
        pc = metrics["per_class"][cls]
        lines.append(
            f"| {cls} | {pc['precision']:.4f} | {pc['recall']:.4f} | "
            f"{pc['f1']:.4f} | {pc['support']} |"
        )
    # Identify weakest class by F1.
    weakest = min(metrics["class_names"],
                  key=lambda c: metrics["per_class"][c]["f1"])
    lines.append("")
    lines.append(f"- Weakest class by F1: **`{weakest}`** "
                 f"(F1={metrics['per_class'][weakest]['f1']:.4f})")
    lines.append("")
    lines.append("![Confusion matrix](confusion_matrix.png)")
    lines.append("")
    lines.append("![Confidence histogram](confidence_hist.png)")
    lines.append("")
    return "\n".join(lines)


def section_threshold(thr: Optional[dict]) -> str:
    if not thr:
        return "## 3. Confidence threshold\n\n_Run `threshold_analysis.py` first._\n"
    lines = [
        "## 3. Confidence threshold sensitivity",
        "",
        f"- AURC (Area under Risk-Coverage): **{fmt_float(thr['aurc'])}** (lower is better)",
        f"- Target selective accuracy: {fmt_pct(thr['target_selective_accuracy'])}",
        f"- Recommended threshold: **{thr.get('recommended_threshold')}**",
        "",
        "### Threshold sweep",
        "",
        "| Threshold | Coverage | Selective acc. | Rejection rate | N accepted |",
        "|---|---|---|---|---|",
    ]
    for r in thr["table"]:
        sel = r["selective_accuracy"]
        sel_str = "N/A" if sel is None or (isinstance(sel, float) and sel != sel) \
            else f"{sel * 100:.2f}%"
        lines.append(
            f"| {r['threshold']:.2f} | {r['coverage'] * 100:.2f}% | "
            f"{sel_str} | {r['rejection_rate'] * 100:.2f}% | {r['n_accepted']} |"
        )
    lines.append("")
    lines.append("### Per-class one-vs-rest AUC")
    lines.append("")
    lines.append("| Class | ROC-AUC | Average Precision |")
    lines.append("|---|---|---|")
    aucs = thr["per_class_auc"]["roc_auc"]
    aps = thr["per_class_auc"]["average_precision"]
    for cls in aucs:
        lines.append(f"| {cls} | {fmt_float(aucs[cls])} | {fmt_float(aps[cls])} |")
    lines.append("")
    lines.append("![Threshold trade-off](threshold_tradeoff.png)")
    lines.append("")
    lines.append("![Risk-Coverage](risk_coverage.png)")
    lines.append("")
    lines.append("![ROC / PR](roc_pr_curves.png)")
    lines.append("")
    return "\n".join(lines)


def section_robustness(rob: Optional[dict]) -> str:
    if not rob:
        return "## 4. Robustness\n\n_Run `robustness_cv.py` first._\n"
    lines = [
        "## 4. Robustness to image perturbations",
        "",
        f"- Baseline accuracy on robust subset: **{fmt_pct(rob['baseline_accuracy'])}**",
        f"- Most sensitive perturbation: **`{rob['most_sensitive']}`**",
        f"- Least sensitive perturbation: **`{rob['least_sensitive']}`**",
        "",
        "### Mean accuracy per perturbation (averaged over 3 severities)",
        "",
        "| Perturbation | Mean accuracy | Drop vs baseline |",
        "|---|---|---|",
    ]
    base = rob["baseline_accuracy"]
    for name, acc in sorted(rob["per_perturbation_mean_accuracy"].items(),
                            key=lambda kv: kv[1]):
        drop = base - acc
        lines.append(f"| {name} | {acc * 100:.2f}% | {drop * 100:+.2f} pp |")
    lines.append("")
    lines.append("![Robustness curves](robustness_curves.png)")
    lines.append("")
    return "\n".join(lines)


def section_latency(lat: Optional[dict]) -> str:
    if not lat:
        return "## 5. Latency & throughput\n\n_Run `benchmark_latency.py` first._\n"
    lines = [
        "## 5. Inference latency & throughput",
        "",
        "### Preprocessing (PIL -> 224x224 normalised tensor)",
        "",
        f"- P50: **{lat['preprocess_ms']['p50_ms']:.2f} ms**,  "
        f"P95: {lat['preprocess_ms']['p95_ms']:.2f} ms,  "
        f"mean: {lat['preprocess_ms']['mean_ms']:.2f} ms",
        "",
        "### Model forward (warmed up, random input)",
        "",
        "| Device | Batch | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (img/s) | Peak GPU mem (MB) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(lat["rows"], key=lambda r: (r["device"], r["batch_size"])):
        mem = "-" if r["peak_memory_mb"] is None else f"{r['peak_memory_mb']:.1f}"
        lines.append(
            f"| `{r['device']}` | {r['batch_size']} | "
            f"{r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} | "
            f"{r['throughput_img_per_s']:.1f} | {mem} |"
        )
    lines.append("")
    # Pick best (lowest P50) for single-image latency.
    single = [r for r in lat["rows"] if r["batch_size"] == 1]
    if single:
        best = min(single, key=lambda r: r["p50_ms"])
        lines.append(f"- Best single-image P50 latency: **{best['p50_ms']:.2f} ms** "
                     f"on `{best['device']}`")
    top_thru = max(lat["rows"], key=lambda r: r["throughput_img_per_s"])
    lines.append(f"- Peak throughput: **{top_thru['throughput_img_per_s']:.1f} img/s** "
                 f"(`{top_thru['device']}`, batch={top_thru['batch_size']})")
    lines.append("")
    lines.append("![Latency distribution](latency_distribution.png)")
    lines.append("")
    lines.append("![Throughput vs batch](throughput_vs_batch.png)")
    lines.append("")
    return "\n".join(lines)


def section_conclusions(metrics, thr, rob, lat) -> str:
    lines = ["## 6. Conclusions & limitations", ""]
    bullets: list[str] = []
    if metrics:
        weakest = min(metrics["class_names"],
                      key=lambda c: metrics["per_class"][c]["f1"])
        bullets.append(
            f"The classifier reaches **Top-1 = {metrics['top1_accuracy'] * 100:.2f}%** "
            f"and **Macro-F1 = {metrics['macro_f1']:.3f}** on the stratified test split; "
            f"`{weakest}` remains the hardest class."
        )
    if thr:
        bullets.append(
            f"Under the backend's default threshold 0.5, selective accuracy is "
            f"{next((r['selective_accuracy'] * 100 for r in thr['table'] if abs(r['threshold'] - 0.5) < 1e-6), 0):.2f}% "
            f"with coverage "
            f"{next((r['coverage'] * 100 for r in thr['table'] if abs(r['threshold'] - 0.5) < 1e-6), 0):.2f}%. "
            f"AURC = {thr['aurc']:.4f}; recommended threshold for "
            f">={thr['target_selective_accuracy'] * 100:.0f}% selective accuracy: "
            f"**{thr.get('recommended_threshold')}**."
        )
    if rob:
        bullets.append(
            f"The model is most fragile under **{rob['most_sensitive']}** and most "
            f"robust under **{rob['least_sensitive']}**; consider adding these "
            f"augmentations in future training rounds."
        )
    if lat:
        single = [r for r in lat["rows"] if r["batch_size"] == 1]
        if single:
            best = min(single, key=lambda r: r["p50_ms"])
            bullets.append(
                f"Single-image P50 latency is **{best['p50_ms']:.2f} ms** on "
                f"`{best['device']}`, well within a <200 ms interactive budget; "
                f"throughput scales to "
                f"{max(lat['rows'], key=lambda r: r['throughput_img_per_s'])['throughput_img_per_s']:.0f} img/s "
                f"at the best batch size."
            )
    for b in bullets:
        lines.append(f"- {b}")
    lines.append("")
    lines.append("### Limitations")
    lines.append("")
    lines.append("- Evaluation set is drawn from the same pool as training data; "
                 "true generalisation on unseen conditions is likely lower.")
    lines.append("- Robustness perturbations cover lens / lighting / compression effects "
                 "but do not emulate background clutter, multi-object scenes, or non-target items.")
    lines.append("- Latency measurements use random input tensors after warm-up; real "
                 "serving adds preprocessing + network transit which are reported separately.")
    lines.append("")
    return "\n".join(lines)


def run_all(manifest_split: Path, model_path: Path) -> None:
    cmds = [
        [sys.executable, "-m", "benchmark.make_splits"],
        [sys.executable, "-m", "benchmark.evaluate_cv",
         "--manifest", str(manifest_split / "test_manifest.csv"),
         "--model", str(model_path)],
        [sys.executable, "-m", "benchmark.threshold_analysis",
         "--manifest", str(manifest_split / "test_manifest.csv"),
         "--model", str(model_path)],
        [sys.executable, "-m", "benchmark.robustness_cv",
         "--manifest", str(manifest_split / "robust_manifest.csv"),
         "--model", str(model_path)],
        [sys.executable, "-m", "benchmark.benchmark_latency",
         "--manifest", str(manifest_split / "test_manifest.csv"),
         "--model", str(model_path)],
    ]
    for cmd in cmds:
        print(f"[report] running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-all", action="store_true",
                        help="Run the four experiment scripts first, then build the report.")
    parser.add_argument("--model", type=Path,
                        default=Path(ROOT.parent) / "best_model.pth")
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    args = parser.parse_args()

    if args.run_all:
        run_all(args.splits_dir, args.model)

    report_dir = ensure_report_dir()
    metrics = load_json(report_dir / "metrics.json")
    thr = load_json(report_dir / "threshold_summary.json")
    rob = load_json(report_dir / "robustness_summary.json")
    lat = load_json(report_dir / "latency_summary.json")

    parts = [
        "# ResNet50 Garbage Classifier - Performance Report",
        "",
        section_overview(report_dir, metrics, lat),
        section_accuracy(metrics),
        section_threshold(thr),
        section_robustness(rob),
        section_latency(lat),
        section_conclusions(metrics, thr, rob, lat),
    ]
    out = "\n".join(parts)
    (report_dir / "REPORT.md").write_text(out, encoding="utf-8")
    print(f"[report] wrote {report_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
