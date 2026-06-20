"""Cache backbone features for every video in a labels CSV.

Reads:  data/labels.csv                     (columns: video_path, leaf_id)
Writes: data/features.pt                    {features, leaf_ids, video_paths, ...}

Run once after labeling, then iterate quickly with scripts/train_classifier.py.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

# Make project modules importable when running as `uv run python scripts/build_features.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import VideoClassifier   # noqa: E402


def _read_labels(csv_path: Path) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or {"video_path", "leaf_id"} - set(reader.fieldnames):
            raise ValueError(
                f"{csv_path}: missing required columns. Header must include "
                f"'video_path' and 'leaf_id'; got {reader.fieldnames}"
            )
        for r in reader:
            vp = (r["video_path"] or "").strip()
            lid = (r["leaf_id"] or "").strip()
            if not vp or not lid:
                continue
            rows.append((Path(vp), lid))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    p.add_argument("--out", type=Path, default=Path("data/features.pt"))
    p.add_argument("--taxonomy", type=Path, default=Path("content_taxonomy_3.1.tsv"))
    p.add_argument("--sample-fps", type=float, default=1.0)
    p.add_argument(
        "--cache-root", type=Path, default=Path("data/cache"),
        help="per-stage cache directory; pass empty string to disable cache",
    )
    args = p.parse_args()

    if not args.labels.exists():
        sys.exit(f"labels CSV not found: {args.labels}")

    rows = _read_labels(args.labels)
    if not rows:
        sys.exit(f"{args.labels}: no rows")

    # Don't load weights at this stage — we only need the backbone, and the
    # warning would be misleading during pre-training feature extraction.
    cache_root = args.cache_root if str(args.cache_root) else None
    clf = VideoClassifier(
        taxonomy_path=args.taxonomy,
        sample_fps=args.sample_fps,
        warn_untrained=False,
        router_weights_dir=None,
        cache_root=cache_root,
    )

    # Validate everything up-front so we fail fast before doing minutes of work
    missing_videos = [vp for vp, _ in rows if not vp.exists()]
    if missing_videos:
        sys.exit(f"missing video files:\n  " + "\n  ".join(str(v) for v in missing_videos))

    bad_labels = [lid for _, lid in rows if lid not in clf.taxonomy.nodes]
    if bad_labels:
        sys.exit(f"leaf_ids not in taxonomy: {sorted(set(bad_labels))[:10]}")

    feats: list[torch.Tensor] = []
    leaf_ids: list[str] = []
    video_paths: list[str] = []
    skipped: list[tuple[str, str]] = []

    started = time.time()
    for i, (vp, lid) in enumerate(rows, 1):
        try:
            ext = clf.extract_features(vp)
        except Exception as e:
            print(f"[{i}/{len(rows)}] SKIP {vp}: {e}", file=sys.stderr)
            skipped.append((str(vp), str(e)))
            continue
        feats.append(ext.feature.squeeze(0).detach().cpu())
        leaf_ids.append(lid)
        video_paths.append(str(vp))
        elapsed = time.time() - started
        rate = i / elapsed if elapsed > 0 else 0.0
        print(f"[{i}/{len(rows)}] {vp.name} -> {lid}  ({rate:.2f} videos/s)")

    if not feats:
        sys.exit("no features extracted; nothing to write")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "features": torch.stack(feats),                # (N, in_dim)
        "leaf_ids": leaf_ids,
        "video_paths": video_paths,
        "feature_dim": clf.in_dim,
        "taxonomy_path": str(args.taxonomy),
        "skipped": skipped,
    }, args.out)
    print(f"\nwrote {len(feats)} features ({clf.in_dim}-dim) → {args.out}")
    if skipped:
        print(f"skipped {len(skipped)} videos (see file for details)")


if __name__ == "__main__":
    main()
