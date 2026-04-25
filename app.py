import argparse
import json
from pathlib import Path

from pipeline import VideoClassifier


def main() -> None:
    p = argparse.ArgumentParser(
        description="Classify mp4 video content into IAB v3.1 categories.",
    )
    p.add_argument("video", type=Path, help="path to the input mp4")
    p.add_argument("--taxonomy", type=Path, default=Path("content_taxonomy_3.1.tsv"))
    p.add_argument("--threshold", type=float, default=0.8,
                   help="confidence threshold below which the cascade escalates")
    p.add_argument("--sample-fps", type=float, default=1.0,
                   help="frames per second sampled for OCR + video encoder")
    p.add_argument("--max-active-routers", type=int, default=16,
                   help="LRU cap on Tier 2+ routers held on MPS at once")
    p.add_argument("--no-caption-filter", action="store_true",
                   help="disable the OCR caption filter; keep every OCR hit (noisier text input)")
    args = p.parse_args()

    clf = VideoClassifier(
        taxonomy_path=args.taxonomy,
        confidence_threshold=args.threshold,
        sample_fps=args.sample_fps,
        max_active_routers=args.max_active_routers,
        caption_only=not args.no_caption_filter,
    )
    print(json.dumps(clf.classify(args.video), indent=2))


if __name__ == "__main__":
    main()
