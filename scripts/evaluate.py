"""Evaluate a trained CascadingClassifier against a labels CSV.

Reads (video_path, leaf_id) rows, classifies each video, compares the
predicted IAB taxonomy path against the truth path. Reports:

- end-to-end leaf accuracy (final predicted ID == labeled leaf ID)
- per-tier accuracy (does the prediction agree with truth at depth k?)
- per-router accuracy on samples whose truth path passes through that
  parent — answers "does router X pick the right child when given a
  sample that belongs in its subtree?"
- text confusion matrix (truth leaf -> predicted leaf counts)

Designed for held-out evaluation. The cache is read-write by default so
re-runs are cheap.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import VideoClassifier   # noqa: E402
from taxonomy import load_taxonomy     # noqa: E402


def _read_labels(csv_path: Path) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or {"video_path", "leaf_id"} - set(reader.fieldnames):
            raise ValueError(
                f"{csv_path}: needs columns 'video_path' and 'leaf_id'; "
                f"got {reader.fieldnames}"
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
    p.add_argument("--labels", type=Path, required=True,
                   help="CSV of (video_path, leaf_id) — typically your held-out test set")
    p.add_argument("--router-weights-dir", type=Path, default=Path("models"))
    p.add_argument("--taxonomy", type=Path, default=Path("content_taxonomy_3.1.tsv"))
    p.add_argument("--threshold", type=float, default=0.8,
                   help="cascade confidence threshold (matches production default)")
    p.add_argument("--force-leaf", action="store_true",
                   help="always descend to the deepest available router, ignore confidence gate")
    p.add_argument("--cache-root", type=Path, default=Path("data/cache"),
                   help="per-stage cache (re-runs are cheap when warm)")
    p.add_argument("--out", type=Path, default=None,
                   help="optional JSON dump of per-sample results + aggregates")
    args = p.parse_args()

    rows = _read_labels(args.labels)
    if not rows:
        sys.exit(f"{args.labels}: no rows")

    tax = load_taxonomy(args.taxonomy)
    bad = [lid for _, lid in rows if lid not in tax.nodes]
    if bad:
        sys.exit(f"unknown leaf_ids in labels: {sorted(set(bad))[:10]}")

    missing_videos = [vp for vp, _ in rows if not vp.exists()]
    if missing_videos:
        sys.exit("missing video files:\n  " + "\n  ".join(str(v) for v in missing_videos))

    clf = VideoClassifier(
        taxonomy_path=args.taxonomy,
        confidence_threshold=args.threshold,
        force_leaf=args.force_leaf,
        router_weights_dir=args.router_weights_dir,
        cache_root=args.cache_root,
        warn_untrained=False,
    )

    print(f"evaluating {len(rows)} samples  (threshold={args.threshold}, "
          f"force_leaf={args.force_leaf})")

    per_sample: list[dict] = []
    leaf_correct = 0
    cascade_reached_leaf = 0
    # tier_correct[k] = (correct, total) at depth k of the truth path
    tier_correct: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    # router_acc[parent_id] = (correct, total) over samples whose truth route
    # passes through parent_id AND the cascade actually visited that parent
    router_acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    # confusion: dict[truth_leaf_id, dict[pred_leaf_id, count]]
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for i, (vp, truth_leaf) in enumerate(rows, 1):
        truth_path = tax.path_to(truth_leaf)
        result = clf.classify(vp)
        steps = result["classification"]["steps"]
        pred_path = [s["id"] for s in steps]
        pred_leaf = pred_path[-1] if pred_path else None

        is_leaf = pred_leaf in tax.nodes and pred_leaf not in tax.children
        if is_leaf:
            cascade_reached_leaf += 1
        if pred_leaf == truth_leaf:
            leaf_correct += 1

        # Per-tier — only score depths the cascade actually reached.
        for depth, truth_id in enumerate(truth_path):
            if depth >= len(pred_path):
                break
            tier_correct[depth][1] += 1
            if pred_path[depth] == truth_id:
                tier_correct[depth][0] += 1

        # Per-router — was the right child predicted at each visited parent
        # along the truth path?
        for s in steps:
            parent_id = s["parent_id"]
            if parent_id is None:
                # root step: parent is the implicit Tier-0; the "right answer"
                # is truth_path[0]
                target = truth_path[0]
                router_acc["__root__"][1] += 1
                if s["id"] == target:
                    router_acc["__root__"][0] += 1
            else:
                # truth's expected child at this parent (if the truth path
                # passes through this parent at all)
                if parent_id in truth_path:
                    pos = truth_path.index(parent_id)
                    if pos + 1 < len(truth_path):
                        target = truth_path[pos + 1]
                        router_acc[parent_id][1] += 1
                        if s["id"] == target:
                            router_acc[parent_id][0] += 1

        confusion[truth_leaf][pred_leaf or "(none)"] += 1

        per_sample.append({
            "video": str(vp),
            "truth_leaf_id": truth_leaf,
            "truth_leaf_name": tax.nodes[truth_leaf].name,
            "truth_path": truth_path,
            "pred_path": pred_path,
            "pred_leaf_id": pred_leaf,
            "pred_leaf_name": tax.nodes[pred_leaf].name if pred_leaf in tax.nodes else None,
            "final_confidence": result["classification"]["final_confidence"],
            "leaf_correct": pred_leaf == truth_leaf,
        })
        ok = "✓" if pred_leaf == truth_leaf else "✗"
        print(f"  [{i}/{len(rows)}] {ok}  {vp.name[:55]:<55}  "
              f"truth={tax.nodes[truth_leaf].name[:30]:<30}  "
              f"pred={(tax.nodes[pred_leaf].name if pred_leaf in tax.nodes else pred_leaf)[:30]}")

    # ---------- Aggregates ----------
    n = len(rows)
    print(f"\n=== overall ===")
    print(f"  end-to-end leaf accuracy:  {leaf_correct}/{n} = {leaf_correct/n:.1%}")
    print(f"  cascade reached a leaf:    {cascade_reached_leaf}/{n}")

    print(f"\n=== per-tier accuracy (truth depth) ===")
    for d in sorted(tier_correct):
        c, t = tier_correct[d]
        print(f"  Tier {d + 1}: {c}/{t} = {c/t:.1%}")

    print(f"\n=== per-router accuracy (samples whose truth path includes that parent) ===")
    for pid in sorted(router_acc):
        c, t = router_acc[pid]
        if pid == "__root__":
            label = "__root__"
        else:
            name = tax.nodes[pid].name if pid in tax.nodes else "?"
            label = f"{pid} ({name})"
        print(f"  {label:<40}  {c}/{t} = {c/t:.1%}")

    print(f"\n=== confusion (truth -> pred counts) ===")
    for truth_id in sorted(confusion):
        truth_name = tax.nodes[truth_id].name
        items = sorted(confusion[truth_id].items(), key=lambda kv: -kv[1])
        bits = ", ".join(
            f"{(tax.nodes[pid].name if pid in tax.nodes else pid)[:25]}({n})"
            for pid, n in items
        )
        print(f"  {truth_name[:40]:<40} -> {bits}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "labels_csv": str(args.labels),
            "n": n,
            "threshold": args.threshold,
            "force_leaf": args.force_leaf,
            "leaf_accuracy": leaf_correct / n,
            "cascade_reached_leaf": cascade_reached_leaf,
            "tier_accuracy": {d: {"correct": c, "total": t}
                              for d, (c, t) in sorted(tier_correct.items())},
            "router_accuracy": {pid: {"correct": c, "total": t}
                                for pid, (c, t) in sorted(router_acc.items())},
            "confusion": {k: dict(v) for k, v in confusion.items()},
            "per_sample": per_sample,
        }, indent=2))
        print(f"\nresults → {args.out}")


if __name__ == "__main__":
    main()
