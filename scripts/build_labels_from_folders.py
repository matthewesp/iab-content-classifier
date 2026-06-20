#!/usr/bin/env python3
"""Derive a labels.csv from the tiered iab/ folder tree.

Videos live under ``iab/<Tier1>/<Tier2>/.../<Leaf>/*.mp4`` where each folder is
a taxonomy node Name (see build_taxonomy_folders.py). This walks the tree, maps
each video's *deepest containing folder* back to its taxonomy Unique ID by
matching the full root->leaf Name path, and writes ``video_path,leaf_id`` rows
that build_features.py consumes.

``_sort_data/`` (the unsorted staging area) is skipped.

Test runs: ``--per-leaf N`` takes the first N videos (sorted by name) from each
leaf — a representative slice across categories. ``--limit M`` caps the total.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxonomy import load_taxonomy


def _sanitize(name: str) -> str:
    """Must match build_taxonomy_folders.py exactly so folder names round-trip."""
    return name.replace("/", "-").replace("\\", "-").strip() or "_unnamed_"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iab", type=Path, default=Path("iab"),
                   help="root iab/ folder containing the tiered tree (default: ./iab)")
    p.add_argument("--taxonomy", type=Path, default=Path("content_taxonomy_3.1.tsv"))
    p.add_argument("--out", type=Path, default=Path("data/labels.csv"))
    p.add_argument("--per-leaf", type=int, default=None,
                   help="take only the first N videos (name-sorted) per leaf folder")
    p.add_argument("--limit", type=int, default=None,
                   help="cap total rows written (applied after --per-leaf)")
    p.add_argument("--abs", action="store_true",
                   help="write absolute video paths (use for Colab/Drive runs)")
    args = p.parse_args()

    tax = load_taxonomy(args.taxonomy)

    # full sanitized Name path (root..node) -> Unique ID. Unique by construction:
    # sibling names are unique per parent, so the whole path disambiguates.
    path_to_uid: dict[tuple[str, ...], str] = {}
    for uid in tax.nodes:
        key = tuple(_sanitize(tax.nodes[n].name) for n in tax.path_to(uid))
        path_to_uid[key] = uid

    iab = args.iab.resolve()
    rows: list[tuple[str, str]] = []
    unmatched: set[tuple[str, ...]] = set()
    skipped_sort = 0

    # Group videos by their containing folder so --per-leaf is deterministic.
    by_folder: dict[Path, list[Path]] = {}
    for v in sorted(iab.rglob("*.mp4")):
        rel = v.relative_to(iab)
        if rel.parts and rel.parts[0] == "_sort_data":
            skipped_sort += 1
            continue
        by_folder.setdefault(v.parent, []).append(v)

    for folder in sorted(by_folder):
        vids = sorted(by_folder[folder])
        if args.per_leaf is not None:
            vids = vids[: args.per_leaf]
        key = tuple(folder.relative_to(iab).parts)
        uid = path_to_uid.get(key)
        if uid is None:
            unmatched.add(key)
            continue
        for v in vids:
            if args.abs:
                path = str(v)
            elif v.is_relative_to(Path.cwd()):
                path = str(v.relative_to(Path.cwd()))
            else:
                path = str(v)  # outside CWD → fall back to absolute
            rows.append((path, uid))

    if args.limit is not None:
        rows = rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_path", "leaf_id"])
        w.writerows(rows)

    leaves = len({r[1] for r in rows})
    print(f"Wrote {len(rows)} labels across {leaves} leaves -> {args.out}", file=sys.stderr)
    if skipped_sort:
        print(f"Skipped {skipped_sort} videos under _sort_data/", file=sys.stderr)
    if unmatched:
        print(f"WARNING: {len(unmatched)} folders had videos but no taxonomy match:",
              file=sys.stderr)
        for k in sorted(unmatched)[:20]:
            print("  " + "/".join(k), file=sys.stderr)


if __name__ == "__main__":
    main()
