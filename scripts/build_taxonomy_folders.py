#!/usr/bin/env python3
"""Materialize the content taxonomy as a tiered folder tree.

Each node becomes a directory nested under its parents, mirroring the
relational ID hierarchy:

    iab/Attractions/Amusement and Theme Parks/
    iab/Automotive/Auto Body Styles/Coupe/
    iab/Automotive/Auto Technology/Auto Infotainment Technologies/

Folder names use node Names (not Unique IDs). Characters illegal in a path
component (notably "/") are sanitized so e.g. "Bars & Restaurants" stays one
folder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/build_taxonomy_folders.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taxonomy import load_taxonomy


def _sanitize(name: str) -> str:
    """Make a node Name safe to use as a single path component."""
    # "/" and "\" would create unintended nesting; strip control/edge chars.
    cleaned = name.replace("/", "-").replace("\\", "-").strip()
    return cleaned or "_unnamed_"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("taxonomy", type=Path,
                   help="taxonomy CSV/TSV (e.g. content_taxonomy_3.1.tsv)")
    p.add_argument("-o", "--out", type=Path, default=Path("iab"),
                   help="root output folder (default: ./iab)")
    p.add_argument("--delimiter", default=None,
                   help="override auto-detect ('\\t' for TSV, ',' for CSV)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the folders that would be created without creating them")
    args = p.parse_args()

    delim = args.delimiter.encode().decode("unicode_escape") if args.delimiter else None
    tax = load_taxonomy(args.taxonomy, delimiter=delim)

    created = 0
    for uid in tax.nodes:
        # path_to → root..uid as Unique IDs; map each to its sanitized Name.
        parts = [_sanitize(tax.nodes[n].name) for n in tax.path_to(uid)]
        folder = args.out.joinpath(*parts)
        if args.dry_run:
            print(folder)
        else:
            folder.mkdir(parents=True, exist_ok=True)
        created += 1

    verb = "Would create" if args.dry_run else "Created"
    print(f"{verb} {created} folders under {args.out}/", file=sys.stderr)


if __name__ == "__main__":
    main()
