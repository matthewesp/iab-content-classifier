from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from video_processor import get_device


# ---------------------------------------------------------------------------
# Uniform router — same architecture at every tier. What differs per tier is
# only the output dim (number of direct children of the parent this router
# serves) and whether it lives in MPS eagerly (root) or lazily (all others).
# ---------------------------------------------------------------------------
class Router(nn.Module):
    def __init__(self, in_dim: int, num_out: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def default_router_builder(
    in_dim: int, num_out: int, hidden_dim: int = 256, dropout: float = 0.1,
    weights_path: Path | None = None,
) -> Callable[[], nn.Module]:
    """Factory for a Router. The returned closure constructs the module and
    loads weights (if any) on CPU — nothing materializes until it's called."""
    def _build() -> nn.Module:
        m = Router(in_dim, num_out, hidden_dim, dropout)
        if weights_path is not None and weights_path.exists():
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            m.load_state_dict(state)
        m.train(False)
        return m
    return _build


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PathStep:
    parent_id: str | None     # None at the root step; a Unique ID otherwise
    predicted_id: str
    predicted_name: str
    confidence: float
    escalated: bool           # was there a deeper step after this one?


@dataclass(frozen=True)
class CascadeResult:
    path: tuple[PathStep, ...]

    @property
    def final_id(self) -> str:
        return self.path[-1].predicted_id

    @property
    def final_name(self) -> str:
        return self.path[-1].predicted_name

    @property
    def final_confidence(self) -> float:
        return self.path[-1].confidence

    @property
    def depth(self) -> int:
        return len(self.path)

    def id_path(self) -> list[str]:
        return [s.predicted_id for s in self.path]

    def name_path(self) -> list[str]:
        return [s.predicted_name for s in self.path]


# ---------------------------------------------------------------------------
class CascadingClassifier(nn.Module):
    """N-tier classifier with lazy-loaded routers at every depth.

    Architecture:
      - Root router (always on MPS) picks among top-level categories.
      - Every non-leaf parent in the taxonomy (at any depth) gets a lazy
        router registered against its Unique ID. That router predicts among
        the parent's direct children only.

    Memory discipline (unchanged from the 2-tier version):
      - Lazy routers live in a plain dict of builder callables. No
        nn.ModuleDict — we need parent.to(device) to NOT pull every router
        into MPS at init time.
      - Active cache is one global LRU keyed by parent_id; eviction is
        tier-agnostic so a freshly-touched Tier-4 router correctly evicts a
        stale Tier-2 router when the cap is hit.

    Forward pass (BFS by tier):
      - Level 0: root router runs once on the full batch.
      - Level k+1: samples that escalated at level k are grouped by the
        router they need next (their level-k prediction ID); each router
        runs exactly once on its sub-batch, then we move to level k+2.
      - Terminates because the taxonomy is a tree (validated at load time)
        and each step moves strictly deeper.
    """

    def __init__(
        self,
        in_dim: int,
        root_ids: list[str],
        root_names: list[str],
        confidence_threshold: float = 0.8,
        max_active_routers: int | None = None,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        force_leaf: bool = False,
    ) -> None:
        super().__init__()
        if len(root_ids) != len(root_names):
            raise ValueError("root_ids and root_names must have the same length")
        if not root_ids:
            raise ValueError("root_ids must be non-empty")
        if not (0.0 < confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in (0, 1]")

        self.device = get_device()
        self.in_dim = in_dim
        self.root_ids = list(root_ids)
        self.root_names = list(root_names)
        self.confidence_threshold = confidence_threshold
        self.max_active_routers = max_active_routers
        # When True, escalate while a router exists for the predicted node
        # regardless of confidence. The cascade still terminates at "deepest
        # available router" — true taxonomy leaves only if every parent has a
        # trained router. Useful when the root is overconfident due to a
        # single-class training distribution.
        self.force_leaf = force_leaf

        self.root_router = Router(in_dim, len(root_ids), hidden_dim, dropout).to(self.device)

        self._builders: dict[str, Callable[[], nn.Module]] = {}
        self._child_ids: dict[str, list[str]] = {}
        self._child_names: dict[str, list[str]] = {}
        self._active: "OrderedDict[str, nn.Module]" = OrderedDict()

    # -- Registry -----------------------------------------------------------
    def register_router(
        self,
        parent_id: str,
        child_ids: list[str],
        child_names: list[str],
        builder: Callable[[], nn.Module] | None = None,
        *,
        weights_path: Path | None = None,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        if parent_id in self._builders:
            raise ValueError(f"router already registered for parent {parent_id!r}")
        if len(child_ids) != len(child_names):
            raise ValueError("child_ids and child_names must be same length")
        if not child_ids:
            raise ValueError("child_ids must be non-empty")
        if builder is None:
            builder = default_router_builder(
                self.in_dim, len(child_ids), hidden_dim, dropout, weights_path,
            )
        self._builders[parent_id] = builder
        self._child_ids[parent_id] = list(child_ids)
        self._child_names[parent_id] = list(child_names)

    def has_router(self, parent_id: str) -> bool:
        return parent_id in self._builders

    def registered_routers(self) -> list[str]:
        return list(self._builders.keys())

    def active_routers(self) -> list[str]:
        return list(self._active.keys())

    def unload_router(self, parent_id: str) -> bool:
        mod = self._active.pop(parent_id, None)
        if mod is None:
            return False
        mod.to("cpu")
        del mod
        if self.device == "mps":
            torch.mps.empty_cache()
        return True

    def unload_all_routers(self) -> None:
        for pid in list(self._active.keys()):
            self.unload_router(pid)

    # -- Lazy materialization ----------------------------------------------
    def _materialize(self, parent_id: str) -> nn.Module:
        cached = self._active.get(parent_id)
        if cached is not None:
            self._active.move_to_end(parent_id)
            return cached

        builder = self._builders.get(parent_id)
        if builder is None:
            raise KeyError(f"no router registered for parent {parent_id!r}")

        router = builder().to(self.device)
        router.train(False)
        for p in router.parameters():
            p.requires_grad_(False)

        self._active[parent_id] = router
        self._evict_if_needed()
        return router

    def _evict_if_needed(self) -> None:
        if self.max_active_routers is None:
            return
        while len(self._active) > self.max_active_routers:
            _, victim = self._active.popitem(last=False)
            victim.to("cpu")
            del victim
        if self.device == "mps":
            torch.mps.empty_cache()

    # -- Forward ------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> list[CascadeResult]:
        if x.dim() != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"expected (B, {self.in_dim}), got {tuple(x.shape)}")

        x = x.to(self.device, non_blocking=True)
        B = x.shape[0]
        paths: list[list[PathStep]] = [[] for _ in range(B)]

        # Level 0 — root router on the full batch
        root_logits = self.root_router(x)
        root_probs = F.softmax(root_logits, dim=-1)
        root_conf, root_idx = root_probs.max(dim=-1)
        conf_cpu = root_conf.detach().cpu().tolist()
        idx_cpu = root_idx.detach().cpu().tolist()

        active: dict[str, list[int]] = {}
        for i in range(B):
            pred_id = self.root_ids[idx_cpu[i]]
            pred_name = self.root_names[idx_cpu[i]]
            c = conf_cpu[i]
            escalate = self.has_router(pred_id) and (
                self.force_leaf or c < self.confidence_threshold
            )
            paths[i].append(PathStep(
                parent_id=None,
                predicted_id=pred_id,
                predicted_name=pred_name,
                confidence=c,
                escalated=escalate,
            ))
            if escalate:
                active.setdefault(pred_id, []).append(i)

        # Levels 1..depth — group by router needed, run each once per level
        while active:
            next_active: dict[str, list[int]] = {}
            for parent_id, sample_ixs in active.items():
                router = self._materialize(parent_id)
                gather = torch.tensor(sample_ixs, device=self.device)
                sub_batch = x.index_select(0, gather)
                sub_logits = router(sub_batch)
                sub_probs = F.softmax(sub_logits, dim=-1)
                sub_conf, sub_idx = sub_probs.max(dim=-1)
                s_conf = sub_conf.cpu().tolist()
                s_idx = sub_idx.cpu().tolist()

                kids_id = self._child_ids[parent_id]
                kids_name = self._child_names[parent_id]
                for j, i in enumerate(sample_ixs):
                    cid = kids_id[s_idx[j]]
                    cname = kids_name[s_idx[j]]
                    c = s_conf[j]
                    escalate = self.has_router(cid) and (
                        self.force_leaf or c < self.confidence_threshold
                    )
                    paths[i].append(PathStep(
                        parent_id=parent_id,
                        predicted_id=cid,
                        predicted_name=cname,
                        confidence=c,
                        escalated=escalate,
                    ))
                    if escalate:
                        next_active.setdefault(cid, []).append(i)
            active = next_active

        return [CascadeResult(path=tuple(p)) for p in paths]
