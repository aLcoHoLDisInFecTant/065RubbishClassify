"""Create reproducible evaluation splits from unified_dataset/.

Generates two CSV manifests under benchmark/splits/:

- test_manifest.csv   : 15% stratified subset used for accuracy &
                         threshold experiments.
- robust_manifest.csv : a disjoint 5% stratified subset used for the
                         (slower) robustness experiments.

Both manifests share columns: ``path, label_name, label_idx``.

Notes
-----
The original training script used ``random_split`` without a fixed seed,
so we cannot reconstruct its validation split. We therefore sample a
fresh stratified subset with a deterministic seed and document in the
final report that train/test overlap is possible.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "unified_dataset"
DEFAULT_CLASS_NAMES = PROJECT_ROOT / "class_names.json"
SPLITS_DIR = Path(__file__).resolve().parent / "splits"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_files(data_dir: Path, class_names: List[str]) -> Dict[str, List[Path]]:
    per_class: Dict[str, List[Path]] = {c: [] for c in class_names}
    for cls in class_names:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Class directory missing: {cls_dir}")
        files = [
            p for p in sorted(cls_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if not files:
            raise RuntimeError(f"No images found under {cls_dir}")
        per_class[cls] = files
    return per_class


def stratified_split(
    per_class: Dict[str, List[Path]],
    test_ratio: float,
    robust_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    rng = random.Random(seed)
    test_rows: List[Tuple[Path, str]] = []
    robust_rows: List[Tuple[Path, str]] = []
    for cls, files in per_class.items():
        shuffled = files.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, int(round(n * test_ratio)))
        n_robust = max(1, int(round(n * robust_ratio)))
        test_part = shuffled[:n_test]
        robust_part = shuffled[n_test:n_test + n_robust]
        test_rows.extend((p, cls) for p in test_part)
        robust_rows.extend((p, cls) for p in robust_part)
    return test_rows, robust_rows


def write_manifest(
    rows: List[Tuple[Path, str]],
    class_to_idx: Dict[str, int],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label_name", "label_idx"])
        for path, cls in rows:
            writer.writerow([str(path), cls, class_to_idx[cls]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--class-names", type=Path, default=DEFAULT_CLASS_NAMES)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--robust-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=Path, default=SPLITS_DIR)
    args = parser.parse_args()

    with args.class_names.open("r", encoding="utf-8") as f:
        class_names: List[str] = json.load(f)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    per_class = collect_files(args.data_dir, class_names)
    totals = {c: len(v) for c, v in per_class.items()}
    print(f"[make_splits] per-class totals: {totals}")

    test_rows, robust_rows = stratified_split(
        per_class,
        test_ratio=args.test_ratio,
        robust_ratio=args.robust_ratio,
        seed=args.seed,
    )

    test_path = args.out_dir / "test_manifest.csv"
    robust_path = args.out_dir / "robust_manifest.csv"
    write_manifest(test_rows, class_to_idx, test_path)
    write_manifest(robust_rows, class_to_idx, robust_path)

    print(f"[make_splits] seed={args.seed}")
    print(f"[make_splits] test  -> {test_path} ({len(test_rows)} rows)")
    print(f"[make_splits] robust-> {robust_path} ({len(robust_rows)} rows)")


if __name__ == "__main__":
    main()
