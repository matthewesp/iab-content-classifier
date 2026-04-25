"""Train all cascade routers from cached backbone features.

Reads:  data/features.pt    (from scripts/build_features.py)
Writes: models/__root__.pt              (root router weights)
        models/<parent_id>.pt           (one per parent with children)
        models/training_summary.json    (per-router stats)

Each router is trained INDEPENDENTLY on the subset of samples whose taxonomy
path passes through its parent. Backbone is frozen — these are linear-probe
classifiers on top of cached features, so training is essentially instant.

After this runs, VideoClassifier (default) auto-loads the trained weights from
models/ — no inference-side code change needed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascading_classifier import Router               # noqa: E402
from taxonomy import load_taxonomy                    # noqa: E402
from video_processor import get_device                # noqa: E402


def _train_one_router(
    router: torch.nn.Module,
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_val: torch.Tensor | None, y_val: torch.Tensor | None,
    epochs: int, lr: float, batch_size: int, weight_decay: float,
) -> dict:
    opt = AdamW(router.parameters(), lr=lr, weight_decay=weight_decay)
    N = X_train.shape[0]
    history = []
    for epoch in range(epochs):
        router.train(True)
        perm = torch.randperm(N, device=X_train.device)
        loss_sum, correct = 0.0, 0
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            logits = router(X_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * idx.shape[0]
            correct += (logits.argmax(-1) == y_train[idx]).sum().item()
        train_loss, train_acc = loss_sum / N, correct / N
        val_loss, val_acc = None, None
        if X_val is not None and X_val.shape[0] > 0:
            router.train(False)
            with torch.inference_mode():
                vlogits = router(X_val)
                val_loss = F.cross_entropy(vlogits, y_val).item()
                val_acc = (vlogits.argmax(-1) == y_val).float().mean().item()
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
    router.train(False)
    return history[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("data/features.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("models"))
    p.add_argument("--taxonomy", type=Path, default=Path("content_taxonomy_3.1.tsv"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="fraction of samples held out for validation per router; "
                        "set to 0 to train on all data (no val metrics)")
    p.add_argument("--min-samples", type=int, default=2,
                   help="skip routers with fewer training samples than this")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.features.exists():
        sys.exit(f"features not found: {args.features} — run scripts/build_features.py first")

    device = get_device()
    torch.manual_seed(args.seed)

    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    features: torch.Tensor = cache["features"].to(device)
    leaf_ids: list[str] = cache["leaf_ids"]
    feature_dim: int = cache["feature_dim"]
    print(f"loaded {features.shape[0]} samples × {feature_dim}-dim from {args.features}")

    tax = load_taxonomy(args.taxonomy)
    paths = [tax.path_to(lid) for lid in leaf_ids]   # root → leaf, inclusive

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {
        "feature_dim": feature_dim,
        "n_samples": features.shape[0],
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "routers": {},
    }
    started = time.time()

    def train_and_save(parent_label: str, parent_id_or_root: str | None,
                       sub_features: torch.Tensor, targets: torch.Tensor,
                       num_classes: int, ckpt_name: str) -> None:
        N = sub_features.shape[0]
        if N < args.min_samples:
            print(f"  [{parent_label}] only {N} samples (< {args.min_samples}), skipped")
            summary["routers"][parent_label] = {"status": "skipped_few_samples", "n": N}
            return

        # train/val split (shuffled)
        perm = torch.randperm(N, device=device)
        n_val = int(N * args.val_frac) if args.val_frac > 0 else 0
        n_train = N - n_val
        if n_train < 1:
            n_train, n_val = N, 0
        train_idx, val_idx = perm[:n_train], perm[n_train:]
        Xtr, ytr = sub_features[train_idx], targets[train_idx]
        Xv = sub_features[val_idx] if n_val else None
        yv = targets[val_idx] if n_val else None

        router = Router(feature_dim, num_classes, args.hidden_dim, args.dropout).to(device)
        last = _train_one_router(
            router, Xtr, ytr, Xv, yv,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            weight_decay=args.weight_decay,
        )
        torch.save(router.state_dict(), args.out_dir / ckpt_name)

        bits = [f"n={N}", f"classes={num_classes}",
                f"train_loss={last['train_loss']:.3f}", f"train_acc={last['train_acc']:.3f}"]
        if last["val_loss"] is not None:
            bits.append(f"val_loss={last['val_loss']:.3f}")
            bits.append(f"val_acc={last['val_acc']:.3f}")
        print(f"  [{parent_label}] {' '.join(bits)}")

        summary["routers"][parent_label] = {
            "status": "trained",
            "ckpt": ckpt_name,
            "n_train": n_train,
            "n_val": n_val,
            "num_classes": num_classes,
            **last,
        }

    # ---------- Root router ----------
    root_targets_list = []
    valid_indices = []
    for i, path in enumerate(paths):
        if not path:
            continue
        try:
            root_targets_list.append(tax.coarse_ids.index(path[0]))
            valid_indices.append(i)
        except ValueError:
            pass  # leaf has no valid root ancestor (shouldn't happen)

    print(f"[root router] {len(tax.coarse_ids)} classes, {len(valid_indices)} samples")
    train_and_save(
        parent_label="__root__",
        parent_id_or_root=None,
        sub_features=features[torch.tensor(valid_indices, device=device)],
        targets=torch.tensor(root_targets_list, device=device),
        num_classes=len(tax.coarse_ids),
        ckpt_name="__root__.pt",
    )

    # ---------- Per-parent routers ----------
    for parent_id, kids in tax.children.items():
        kid_ids = [k.unique_id for k in kids]
        idxs, targets_local = [], []
        for i, path in enumerate(paths):
            if parent_id not in path:
                continue
            pos = path.index(parent_id)
            if pos + 1 >= len(path):
                continue   # the labeled leaf IS this parent — no child to predict
            next_id = path[pos + 1]
            if next_id not in kid_ids:
                continue   # taxonomy inconsistency, skip defensively
            idxs.append(i)
            targets_local.append(kid_ids.index(next_id))

        parent_name = tax.nodes[parent_id].name
        label = f"{parent_id} ({parent_name})"
        print(f"[{label}] {len(kids)} classes, {len(idxs)} samples")
        if not idxs:
            summary["routers"][parent_id] = {"status": "no_samples", "n": 0,
                                              "name": parent_name, "num_classes": len(kids)}
            continue

        train_and_save(
            parent_label=parent_id,
            parent_id_or_root=parent_id,
            sub_features=features[torch.tensor(idxs, device=device)],
            targets=torch.tensor(targets_local, device=device),
            num_classes=len(kids),
            ckpt_name=f"{parent_id}.pt",
        )

    summary["elapsed_s"] = time.time() - started
    summary["taxonomy_path"] = str(args.taxonomy)
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    n_trained = sum(1 for r in summary["routers"].values() if r.get("status") == "trained")
    n_skipped = len(summary["routers"]) - n_trained
    print(f"\ntrained {n_trained} routers ({n_skipped} skipped) in {summary['elapsed_s']:.1f}s")
    print(f"summary → {args.out_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
