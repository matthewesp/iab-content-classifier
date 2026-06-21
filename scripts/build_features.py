"""Cache backbone features for every video in a labels CSV.

Reads:  data/labels.csv                     (columns: video_path, leaf_id)
Writes: data/features.pt                    {features, leaf_ids, video_paths, ...}

Run once after labeling, then iterate quickly with scripts/train_classifier.py.

Feature extraction is dominated by OCR + video decode (CPU/GPU detection),
not the GPU backbones — see data/profile_*.json. So the throughput win on a
big GPU (e.g. Colab H100) comes from running many videos CONCURRENTLY so OCR
and decode overlap and the GPU stays fed. Use --workers N for that. The
per-video stage cache makes concurrent runs safe and resumable: each video
keys on its own file hash, so workers never collide.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import torch

# Make project modules importable when running as `uv run python scripts/build_features.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxonomy import load_taxonomy   # noqa: E402


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


# --- worker process state ---------------------------------------------------
# One VideoClassifier per worker process, built once in the initializer and
# reused across all videos that worker handles (loading the backbone per video
# would dwarf the actual work).
_WORKER: dict[str, object] = {}


def _init_worker(taxonomy_path: str, sample_fps: float, cache_root: str | None,
                 threads: int) -> None:
    # Limit intra-op threads per worker so N processes don't oversubscribe the
    # CPU and slow each other's OCR/decode. Set before torch spins up its pools.
    if threads > 0:
        torch.set_num_threads(threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    from pipeline import VideoClassifier
    _WORKER["clf"] = VideoClassifier(
        taxonomy_path=Path(taxonomy_path),
        sample_fps=sample_fps,
        warn_untrained=False,
        router_weights_dir=None,
        cache_root=Path(cache_root) if cache_root else None,
    )


def _process_one(task: tuple[int, str, str]):
    """Returns (idx, feature_cpu | None, leaf_id, video_path, error | None)."""
    idx, vp, lid = task
    clf = _WORKER["clf"]
    try:
        ext = clf.extract_features(Path(vp))           # type: ignore[attr-defined]
    except Exception as e:                              # noqa: BLE001
        return (idx, None, lid, vp, str(e))
    return (idx, ext.feature.squeeze(0).detach().cpu(), lid, vp, None)


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
    p.add_argument(
        "--workers", type=int, default=1,
        help="concurrent video workers. >1 overlaps OCR/decode and keeps the "
             "GPU fed — the main throughput lever on a big GPU. Each worker "
             "loads its own backbone copy (small models, fine on H100).",
    )
    args = p.parse_args()

    if not args.labels.exists():
        sys.exit(f"labels CSV not found: {args.labels}")

    rows = _read_labels(args.labels)
    if not rows:
        sys.exit(f"{args.labels}: no rows")

    cache_root = str(args.cache_root) if str(args.cache_root) else None

    # Validate up-front WITHOUT loading the backbone (taxonomy only) so we fail
    # fast and don't hold GPU memory in the parent while workers run.
    tax = load_taxonomy(args.taxonomy)
    missing_videos = [vp for vp, _ in rows if not vp.exists()]
    if missing_videos:
        sys.exit("missing video files:\n  " + "\n  ".join(str(v) for v in missing_videos))
    bad_labels = [lid for _, lid in rows if lid not in tax.nodes]
    if bad_labels:
        sys.exit(f"leaf_ids not in taxonomy: {sorted(set(bad_labels))[:10]}")

    n = len(rows)
    workers = max(1, args.workers)
    # Divide CPU across workers (min 1) to avoid thread oversubscription.
    per_worker_threads = max(1, (os.cpu_count() or workers) // workers)

    results: list[tuple] = [None] * n  # type: ignore[assignment]
    started = time.time()
    done = 0

    def _record(res: tuple) -> None:
        nonlocal done
        idx, feat, lid, vp, err = res
        results[idx] = res
        done += 1
        elapsed = time.time() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        name = Path(vp).name
        if err is None:
            print(f"[{done}/{n}] {name} -> {lid}  ({rate:.2f} videos/s)")
        else:
            print(f"[{done}/{n}] SKIP {vp}: {err}", file=sys.stderr)

    tasks = [(i, str(vp), lid) for i, (vp, lid) in enumerate(rows)]

    if workers == 1:
        # Serial path — no multiprocessing overhead; identical results.
        _init_worker(str(args.taxonomy), args.sample_fps, cache_root, 0)
        for t in tasks:
            _record(_process_one(t))
    else:
        # CUDA + fork is unsafe; spawn fresh interpreters for each worker.
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        print(f"extracting with {workers} workers "
              f"({per_worker_threads} CPU threads each)…")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(str(args.taxonomy), args.sample_fps, cache_root, per_worker_threads),
        ) as ex:
            futures = [ex.submit(_process_one, t) for t in tasks]
            for fut in as_completed(futures):
                _record(fut.result())

    feats: list[torch.Tensor] = []
    leaf_ids: list[str] = []
    video_paths: list[str] = []
    skipped: list[tuple[str, str]] = []
    for res in results:
        idx, feat, lid, vp, err = res
        if err is not None:
            skipped.append((vp, err))
            continue
        feats.append(feat)
        leaf_ids.append(lid)
        video_paths.append(vp)

    if not feats:
        sys.exit("no features extracted; nothing to write")

    feature_dim = int(feats[0].shape[-1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "features": torch.stack(feats),                # (N, in_dim)
        "leaf_ids": leaf_ids,
        "video_paths": video_paths,
        "feature_dim": feature_dim,
        "taxonomy_path": str(args.taxonomy),
        "skipped": skipped,
    }, args.out)
    total = time.time() - started
    print(f"\nwrote {len(feats)} features ({feature_dim}-dim) → {args.out} "
          f"in {total:.0f}s ({len(feats)/total:.2f} videos/s)")
    if skipped:
        print(f"skipped {len(skipped)} videos (see file for details)")


if __name__ == "__main__":
    main()
