from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from cascading_classifier import CascadingClassifier


# Accept a variety of header spellings — IAB files in the wild are inconsistent.
_ID_ALIASES = ("unique id", "uniqueid", "id")
_PARENT_ALIASES = ("parent", "parent id", "parentid")
_NAME_ALIASES = ("name", "label", "category")   # do NOT include "tier" — that
                                                # conflicts with "Tier 1"/"Tier 2"
                                                # columns in IAB v3.x files.


def _find_col(header: list[str], aliases: tuple[str, ...]) -> int | None:
    low = [c.strip().lower() for c in header]
    for i, h in enumerate(low):
        if h in aliases:
            return i
    return None


@dataclass(frozen=True)
class TaxonomyNode:
    unique_id: str
    parent_id: str | None
    name: str


@dataclass
class Taxonomy:
    """Parsed hierarchy + all the index↔ID mappings the model needs.

    Index ordering follows CSV row order deterministically: rows earlier in the
    file get lower indices at their tier. Do not reorder the CSV after training
    a checkpoint — indices will silently drift.
    """
    nodes: dict[str, TaxonomyNode]
    coarse_ids: list[str]
    coarse_names: list[str]
    coarse_id_by_index: dict[int, str]
    coarse_index_by_id: dict[str, int]

    # parent_id → ordered list of child nodes (Tier 2 "experts")
    children: dict[str, list[TaxonomyNode]] = field(default_factory=dict)

    # (parent_id, sub_index) → child Unique ID
    fine_id_by_index: dict[tuple[str, int], str] = field(default_factory=dict)

    def depth_of(self, uid: str) -> int:
        """1 = root, 2 = Tier 2, etc."""
        d = 1
        cur = self.nodes[uid].parent_id
        while cur is not None:
            d += 1
            cur = self.nodes[cur].parent_id
        return d

    def tier_sizes(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for uid in self.nodes:
            d = self.depth_of(uid)
            out[d] = out.get(d, 0) + 1
        return dict(sorted(out.items()))

    def descendants(self, uid: str) -> list[TaxonomyNode]:
        """All non-self descendants, DFS in CSV order."""
        out: list[TaxonomyNode] = []
        def _walk(nid: str) -> None:
            for c in self.children.get(nid, []):
                out.append(c)
                _walk(c.unique_id)
        _walk(uid)
        return out

    def path_to(self, uid: str) -> list[str]:
        """Root → uid (inclusive) as a list of Unique IDs."""
        out: list[str] = []
        cur: str | None = uid
        while cur is not None:
            out.append(cur)
            cur = self.nodes[cur].parent_id
        return list(reversed(out))

    def mapping(self) -> dict:
        """Plain dict view — useful for serializing alongside model checkpoints.

        ``experts`` includes EVERY parent with children at any depth, not just
        roots. The current CascadingClassifier only wires roots + direct
        children, but the full grouping is kept here so n-tier models can use it.
        """
        return {
            "coarse_index_to_id": {i: uid for i, uid in self.coarse_id_by_index.items()},
            "coarse_id_to_name": {uid: self.nodes[uid].name for uid in self.coarse_ids},
            "tier_sizes": self.tier_sizes(),
            "experts": {
                parent_id: {
                    "parent_name": self.nodes[parent_id].name,
                    "parent_depth": self.depth_of(parent_id),
                    "fine_index_to_id": [c.unique_id for c in children],
                    "fine_index_to_name": [c.name for c in children],
                }
                for parent_id, children in self.children.items()
            },
        }


def load_taxonomy(
    csv_path: Path | str,
    delimiter: str | None = None,
) -> Taxonomy:
    """Parse a relational-ID taxonomy file.

    Auto-detects:
      - delimiter: ``.tsv`` → tab, else comma (override with ``delimiter=``).
      - header row: the first row containing columns matching Unique ID,
        Parent, and Name. Any rows above (e.g. banner / section header) are
        skipped.

    Columns beyond those three are ignored — useful for IAB v3.x files that
    also carry Tier 1..Tier N and Extension columns.
    """
    csv_path = Path(csv_path)
    if delimiter is None:
        delimiter = "\t" if csv_path.suffix.lower() == ".tsv" else ","

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        all_rows = list(csv.reader(f, delimiter=delimiter))

    if not all_rows:
        raise ValueError(f"{csv_path}: empty file")

    header_idx: int | None = None
    id_col = parent_col = name_col = -1
    for i, row in enumerate(all_rows):
        ic = _find_col(row, _ID_ALIASES)
        pc = _find_col(row, _PARENT_ALIASES)
        nc = _find_col(row, _NAME_ALIASES)
        if ic is not None and pc is not None and nc is not None:
            header_idx, id_col, parent_col, name_col = i, ic, pc, nc
            break
    if header_idx is None:
        raise ValueError(
            f"{csv_path}: could not find header row containing columns matching "
            f"{_ID_ALIASES} / {_PARENT_ALIASES} / {_NAME_ALIASES}"
        )

    rows: list[tuple[str, str | None, str]] = []
    max_col = max(id_col, parent_col, name_col)
    for r in all_rows[header_idx + 1:]:
        if len(r) <= max_col:
            continue
        uid = r[id_col].strip()
        parent_raw = r[parent_col].strip()
        name = r[name_col].strip()
        if not uid:
            continue
        rows.append((uid, parent_raw or None, name))

    if not rows:
        raise ValueError(f"{csv_path}: no data rows")

    # Detect root marker: treat "", "0", or a value that isn't itself a known
    # Unique ID as root. We resolve after one pass so order-of-appearance
    # doesn't matter.
    ids_seen = {uid for uid, _, _ in rows}

    def normalize_parent(p: str | None) -> str | None:
        if p is None or p == "" or p == "0":
            return None
        if p not in ids_seen:
            return None  # orphan parent → treat as root
        return p

    nodes: dict[str, TaxonomyNode] = {}
    coarse_ids: list[str] = []
    children: dict[str, list[TaxonomyNode]] = {}

    for uid, parent_raw, name in rows:
        parent = normalize_parent(parent_raw)
        if uid in nodes:
            raise ValueError(f"{csv_path}: duplicate Unique ID {uid!r}")
        node = TaxonomyNode(unique_id=uid, parent_id=parent, name=name)
        nodes[uid] = node
        if parent is None:
            coarse_ids.append(uid)
        else:
            children.setdefault(parent, []).append(node)

    # Cycle check
    for uid, node in nodes.items():
        seen = {uid}
        cur = node.parent_id
        while cur is not None:
            if cur in seen:
                raise ValueError(f"{csv_path}: cycle detected at {uid!r}")
            seen.add(cur)
            cur = nodes[cur].parent_id if cur in nodes else None

    # Within a single parent, direct children must have distinct names — the
    # router for that parent uses names as display labels at that tier.  Names
    # CAN repeat across subtrees (e.g. "News" under multiple roots); routing
    # keys off Unique IDs, so that's fine.
    for parent_id, kids in children.items():
        seen_names: dict[str, str] = {}
        for k in kids:
            if k.name in seen_names:
                raise ValueError(
                    f"{csv_path}: duplicate child name {k.name!r} under parent "
                    f"{parent_id!r} (ids {seen_names[k.name]!r} and {k.unique_id!r})"
                )
            seen_names[k.name] = k.unique_id

    coarse_index_by_id = {uid: i for i, uid in enumerate(coarse_ids)}
    coarse_id_by_index = {i: uid for uid, i in coarse_index_by_id.items()}
    coarse_names = [nodes[uid].name for uid in coarse_ids]

    fine_id_by_index: dict[tuple[str, int], str] = {}
    for parent_id, kids in children.items():
        for j, k in enumerate(kids):
            fine_id_by_index[(parent_id, j)] = k.unique_id

    return Taxonomy(
        nodes=nodes,
        coarse_ids=coarse_ids,
        coarse_names=coarse_names,
        coarse_id_by_index=coarse_id_by_index,
        coarse_index_by_id=coarse_index_by_id,
        children=children,
        fine_id_by_index=fine_id_by_index,
    )


def build_classifier(
    tax: Taxonomy,
    in_dim: int,
    confidence_threshold: float = 0.8,
    max_active_routers: int | None = None,
    hidden_dim: int = 256,
    router_weights_dir: Path | None = None,
    dropout: float = 0.1,
) -> CascadingClassifier:
    """Wire the full taxonomy into an n-tier CascadingClassifier.

    - Root router output dim = ``len(tax.coarse_ids)``, indexed in CSV row order.
    - For EVERY parent with children in the taxonomy (any depth), one lazy
      router is registered whose output dim = number of its direct children.
      All routers are keyed by parent Unique ID.
    - If ``router_weights_dir`` is given, each router lazy-loads from
      ``<dir>/<parent_unique_id>.pt`` on first activation.
    """
    clf = CascadingClassifier(
        in_dim=in_dim,
        root_ids=tax.coarse_ids,
        root_names=tax.coarse_names,
        confidence_threshold=confidence_threshold,
        max_active_routers=max_active_routers,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    for parent_id, kids in tax.children.items():
        weights_path = (
            (router_weights_dir / f"{parent_id}.pt") if router_weights_dir else None
        )
        clf.register_router(
            parent_id=parent_id,
            child_ids=[k.unique_id for k in kids],
            child_names=[k.name for k in kids],
            weights_path=weights_path,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    # Root router is always-resident, so it bypasses the lazy-load path. Load
    # its trained weights here if a checkpoint exists at <dir>/__root__.pt.
    if router_weights_dir is not None:
        root_ckpt = router_weights_dir / "__root__.pt"
        if root_ckpt.exists():
            import torch
            state = torch.load(root_ckpt, map_location=clf.device, weights_only=True)
            clf.root_router.load_state_dict(state)

    return clf


def main() -> None:
    p = argparse.ArgumentParser(description="Parse a relational-ID taxonomy CSV/TSV")
    p.add_argument("path", type=Path)
    p.add_argument("--delimiter", default=None,
                   help="override auto-detect ('\\t' for TSV, ',' for CSV)")
    p.add_argument("--json", action="store_true", help="dump full mapping as JSON")
    p.add_argument("--tree", action="store_true", help="print full DFS tree")
    args = p.parse_args()

    delim = args.delimiter.encode().decode("unicode_escape") if args.delimiter else None
    tax = load_taxonomy(args.path, delimiter=delim)

    if args.json:
        print(json.dumps(tax.mapping(), indent=2, ensure_ascii=False))
        return

    sizes = tax.tier_sizes()
    total = sum(sizes.values())
    parents_with_kids = len(tax.children)
    print(f"{args.path}: {total} nodes, max depth {max(sizes)}")
    for d, n in sizes.items():
        print(f"  Tier {d}: {n} nodes")
    print(f"  Parents with children: {parents_with_kids}")
    print(f"  Roots (Tier 1): {len(tax.coarse_ids)}")

    if args.tree:
        def walk(uid: str, depth: int) -> None:
            node = tax.nodes[uid]
            indent = "    " * (depth - 1)
            print(f"{indent}[{node.unique_id:>6s}] {node.name}")
            for c in tax.children.get(uid, []):
                walk(c.unique_id, depth + 1)
        for uid in tax.coarse_ids:
            walk(uid, 1)
    else:
        print()
        print("Tier 1 + direct children (what CascadingClassifier currently wires):")
        for i, uid in enumerate(tax.coarse_ids):
            kids = tax.children.get(uid, [])
            deeper = len(tax.descendants(uid)) - len(kids)
            extra = f"  [+{deeper} deeper]" if deeper else ""
            print(f"  [{i:3d}] {uid:>6s}  {tax.nodes[uid].name}  "
                  f"({len(kids)} direct children{extra})")


if __name__ == "__main__":
    main()
