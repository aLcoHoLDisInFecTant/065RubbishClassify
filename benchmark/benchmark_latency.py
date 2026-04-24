"""Experiment 4: inference latency and throughput.

Measures for each (device, batch_size) combination:
  - preprocessing time (PIL -> normalised tensor)
  - model forward time (warmed up, 200 iterations)
  - end-to-end latency
  - throughput (images / second)
  - peak GPU memory (if CUDA)

Outputs under ``benchmark/report/``:

- ``latency_table.csv``             : aggregate stats per (device, batch).
- ``latency_distribution.png``      : batch=1 forward-latency histogram.
- ``throughput_vs_batch.png``       : throughput curve per device.
- ``latency_summary.json``          : raw percentiles + environment info.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from benchmark.common import (
    DEFAULT_MODEL_PATH,
    ManifestDataset,
    SPLITS_DIR,
    ensure_report_dir,
    get_eval_transform,
    load_model,
    read_manifest,
)


def percentiles(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "std_ms": float(values.std()),
    }


@torch.no_grad()
def time_forward(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    warmup: int,
    iters: int,
) -> Dict[str, object]:
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    use_cuda = device.type == "cuda"

    # Warmup
    for _ in range(warmup):
        _ = model(x)
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    times_ms = np.empty(iters, dtype=np.float64)

    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for i in range(iters):
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            times_ms[i] = starter.elapsed_time(ender)
    else:
        for i in range(iters):
            t0 = time.perf_counter()
            _ = model(x)
            times_ms[i] = (time.perf_counter() - t0) * 1000.0

    result = {"per_iter_ms": times_ms, **percentiles(times_ms)}
    if use_cuda:
        result["peak_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    else:
        result["peak_memory_mb"] = None
    return result


def time_preprocess(
    real_image_paths: List[Path],
    iters: int,
) -> Dict[str, float]:
    from PIL import Image
    tf = get_eval_transform()
    times_ms = np.empty(iters, dtype=np.float64)
    n = len(real_image_paths)
    for i in range(iters):
        path = real_image_paths[i % n]
        t0 = time.perf_counter()
        img = Image.open(path).convert("RGB")
        _ = tf(img).unsqueeze(0)
        times_ms[i] = (time.perf_counter() - t0) * 1000.0
    return percentiles(times_ms)


def plot_distribution(per_iter_ms: np.ndarray, label: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(per_iter_ms, bins=40, color="#2a9d8f", alpha=0.85)
    for q, color in zip([50, 95, 99], ["#264653", "#e76f51", "#9c2a5f"]):
        v = float(np.percentile(per_iter_ms, q))
        ax.axvline(v, linestyle="--", color=color, label=f"P{q}={v:.2f}ms")
    ax.set_xlabel("Forward latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"Forward latency distribution ({label}, batch=1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_throughput(rows: List[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    by_device: Dict[str, List[dict]] = {}
    for r in rows:
        by_device.setdefault(r["device"], []).append(r)
    for dev, entries in by_device.items():
        entries = sorted(entries, key=lambda r: r["batch_size"])
        xs = [r["batch_size"] for r in entries]
        ys = [r["throughput_img_per_s"] for r in entries]
        ax.plot(xs, ys, "o-", label=dev)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (images / s)")
    ax.set_title("Throughput vs batch size")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SPLITS_DIR / "test_manifest.csv")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--devices", type=str, nargs="+", default=None,
                        help="Override auto-selection, e.g. --devices cuda cpu")
    parser.add_argument("--preprocess-iters", type=int, default=100)
    args = parser.parse_args()

    report_dir = ensure_report_dir()

    if args.devices:
        devices = [torch.device(d) for d in args.devices]
    else:
        devices = [torch.device("cuda:0")] if torch.cuda.is_available() else []
        devices.append(torch.device("cpu"))

    manifest_rows = read_manifest(args.manifest)
    preprocess_stats = time_preprocess(
        [r.path for r in manifest_rows[: min(100, len(manifest_rows))]],
        iters=args.preprocess_iters,
    )

    rows: List[dict] = []
    dist_to_plot = None

    for dev in devices:
        print(f"[latency] loading model to {dev}")
        model, _names, dev = load_model(args.model, device=dev)
        for bs in args.batches:
            try:
                res = time_forward(model, dev, bs, args.warmup, args.iters)
            except RuntimeError as e:
                print(f"[latency] batch={bs} device={dev} failed: {e}")
                continue
            per_iter = res.pop("per_iter_ms")
            throughput = bs / (res["mean_ms"] / 1000.0)
            row = {
                "device": str(dev),
                "batch_size": bs,
                "warmup": args.warmup,
                "iters": args.iters,
                "throughput_img_per_s": float(throughput),
                **res,
            }
            rows.append(row)
            print(f"[latency] {dev} bs={bs:3d}  "
                  f"P50={res['p50_ms']:.2f}ms  P95={res['p95_ms']:.2f}ms  "
                  f"throughput={throughput:.1f} img/s")
            if bs == 1 and dist_to_plot is None:
                dist_to_plot = (str(dev), per_iter)

        # Free memory before next device
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = report_dir / "latency_table.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("device,batch_size,mean_ms,p50_ms,p90_ms,p95_ms,p99_ms,"
                "min_ms,max_ms,std_ms,throughput_img_per_s,peak_memory_mb\n")
        for r in rows:
            mem = "" if r["peak_memory_mb"] is None else f"{r['peak_memory_mb']:.2f}"
            f.write(
                f"{r['device']},{r['batch_size']},{r['mean_ms']:.3f},"
                f"{r['p50_ms']:.3f},{r['p90_ms']:.3f},{r['p95_ms']:.3f},"
                f"{r['p99_ms']:.3f},{r['min_ms']:.3f},{r['max_ms']:.3f},"
                f"{r['std_ms']:.3f},{r['throughput_img_per_s']:.2f},{mem}\n"
            )

    if dist_to_plot is not None:
        plot_distribution(dist_to_plot[1], dist_to_plot[0],
                          report_dir / "latency_distribution.png")
    plot_throughput(rows, report_dir / "throughput_vs_batch.png")

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (report_dir / "latency_summary.json").write_text(
        json.dumps({
            "env": env,
            "preprocess_ms": preprocess_stats,
            "rows": rows,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[latency] done. preprocess P50={preprocess_stats['p50_ms']:.2f}ms")


if __name__ == "__main__":
    main()
