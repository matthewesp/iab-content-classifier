"""Profile per-stage wall-clock for VideoClassifier.extract_features.

Run BEFORE optimizing anything — establishes the baseline so later phases can
prove their wins in real numbers.

Capture MPS fallbacks (silent CPU round-trips) by setting these env vars in the
shell before invoking:

    PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_LOG_FALLBACKS=1 \\
        uv run python scripts/profile_pipeline.py 2>&1 | tee profile.log

Lines mentioning "fallback" in profile.log are per-call CPU round-trips; each
one is a likely optimization target.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Permissive MPS fallback by default so we can actually run; logging is opt-in
# via the env var above (must be set BEFORE the python process starts to be
# picked up by libtorch on import).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import VideoClassifier  # noqa: E402


def _summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "total_s": sum(values),
        "mean_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "min_s": min(values),
        "max_s": max(values),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "videos", nargs="*", type=Path,
        help="video files to profile (defaults to data/videos/*.mp4)",
    )
    p.add_argument("--out", type=Path, default=Path("data/profile_baseline.json"))
    p.add_argument("--taxonomy", type=Path, default=Path("content_taxonomy_3.1.tsv"))
    p.add_argument("--sample-fps", type=float, default=1.0)
    p.add_argument(
        "--use-cache", action="store_true",
        help="enable per-stage cache (default off — profiling needs cold runs)",
    )
    p.add_argument(
        "--cache-root", type=Path, default=Path("data/cache"),
        help="cache directory used when --use-cache is set",
    )
    args = p.parse_args()

    videos = list(args.videos) or sorted(Path("data/videos").glob("*.mp4"))
    if not videos:
        sys.exit("no videos to profile (pass paths or populate data/videos/)")

    print(f"profiling {len(videos)} video(s) with sample_fps={args.sample_fps}")
    clf = VideoClassifier(
        taxonomy_path=args.taxonomy,
        sample_fps=args.sample_fps,
        warn_untrained=False,
        router_weights_dir=None,
        cache_root=args.cache_root if args.use_cache else None,
    )
    print(f"device: {clf.device}")

    per_video: list[dict] = []
    started = time.perf_counter()
    for i, vp in enumerate(videos, 1):
        if not vp.exists():
            print(f"[{i}/{len(videos)}] missing: {vp}", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        ext = clf.extract_features(vp, profile=True, use_cache=args.use_cache)
        wall = time.perf_counter() - t0
        ext.stage_timings["wall_s"] = wall
        per_video.append({
            "video": str(vp),
            "frames_sampled": ext.frames_sampled,
            **ext.stage_timings,
        })
        print(
            f"[{i}/{len(videos)}] {vp.name}  wall={wall:.2f}s  "
            f"frames={ext.frames_sampled}  "
            f"ocr_calls={int(ext.stage_timings.get('ocr_n_calls', 0))}"
        )

    if not per_video:
        sys.exit("no videos profiled")

    timing_keys = sorted({
        k for r in per_video for k in r
        if k.endswith("_s") and k != "wall_s"
    })
    aggregates = {
        k: _summarize([r[k] for r in per_video if k in r])
        for k in timing_keys + ["wall_s"]
    }

    print("\nstage                     total_s     mean_s   median_s   n")
    print("-" * 65)
    for k in sorted(timing_keys, key=lambda x: -aggregates[x]["total_s"]):
        a = aggregates[k]
        print(
            f"{k:<24} {a['total_s']:>8.2f}  {a['mean_s']:>8.3f}  "
            f"{a['median_s']:>8.3f}  {a['n']:>3}"
        )
    a = aggregates["wall_s"]
    print(
        f"{'wall_s (sum)':<24} {a['total_s']:>8.2f}  {a['mean_s']:>8.3f}  "
        f"{a['median_s']:>8.3f}  {a['n']:>3}"
    )

    elapsed = time.perf_counter() - started
    print(f"\ntotal elapsed: {elapsed:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "device": clf.device,
        "sample_fps": args.sample_fps,
        "use_cache": args.use_cache,
        "elapsed_s": elapsed,
        "per_video": per_video,
        "aggregates": aggregates,
    }, indent=2))
    print(f"baseline → {args.out}")


if __name__ == "__main__":
    main()
