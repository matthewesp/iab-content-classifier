"""Per-stage feature cache for VideoClassifier.extract_features.

Layout:
    <root>/<video_key>/<stage>_<params_key>.<ext>

Where:
    video_key  = sha256(abspath + mtime_ns + size)[:16]
                 — invalidates automatically on any edit to the source video.
    params_key = sha256(json(stage_params))[:8]
                 — different param sets coexist side-by-side, so A/B tests
                 (e.g. swapping OCR engine, changing sample_fps) don't trash
                 the prior result.

Stages persist independently, so a single change downstream (e.g. swapping the
text model) only invalidates that stage and the fused tensor — OCR + ASR stay
cached. Iteration cost drops from ~60s/video to <1s once warm.

Stages and on-disk format:
    ocr        json   { "hits": [...] }
    asr        text   transcript
    text_emb   pt     fp32 tensor (cpu)
    audio_emb  pt     fp32 tensor (cpu)
    video_emb  pt     fp32 tensor (cpu)
    fused      pt     fp32 tensor (cpu)

The cache is purely additive — if any params don't match, the file just won't
be found and the stage runs as normal. There is no eviction; delete the cache
directory by hand to reclaim disk.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from video_processor import OCRHit


def video_key(video_path: Path) -> str:
    p = Path(video_path).resolve()
    st = p.stat()
    raw = f"{p}:{st.st_mtime_ns}:{st.st_size}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def params_key(params: dict) -> str:
    raw = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:8]


class StageCache:
    def __init__(self, root: Path, video_path: Path) -> None:
        self.dir = Path(root) / video_key(video_path)
        self.video = Path(video_path)

    def _path(self, stage: str, params: dict, ext: str) -> Path:
        return self.dir / f"{stage}_{params_key(params)}.{ext}"

    # ---- OCR + frame count ----
    # frames_sampled is co-located with OCR hits because both come from the
    # same scan pass. When OCR is cache-hit but video_emb is cache-miss we
    # still re-decode frames; the count here lets the cache-hit / fully-warm
    # path report frames_sampled without touching the video.
    def get_ocr(self, params: dict) -> tuple[int, list[OCRHit]] | None:
        p = self._path("ocr", params, "json")
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        hits = [
            OCRHit(
                frame_index=int(h["frame_index"]),
                timestamp_s=float(h["timestamp_s"]),
                text=h["text"],
                confidence=float(h["confidence"]),
                bbox=[(int(x), int(y)) for x, y in h["bbox"]],
            )
            for h in data["hits"]
        ]
        return int(data["frames_sampled"]), hits

    def set_ocr(self, params: dict, frames_sampled: int, hits: list[OCRHit]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path("ocr", params, "json").write_text(json.dumps({
            "frames_sampled": frames_sampled,
            "hits": [
                {
                    "frame_index": h.frame_index,
                    "timestamp_s": h.timestamp_s,
                    "text": h.text,
                    "confidence": h.confidence,
                    "bbox": [list(pt) for pt in h.bbox],
                }
                for h in hits
            ]
        }))

    # ---- ASR ----
    def get_asr(self, params: dict) -> str | None:
        p = self._path("asr", params, "txt")
        return p.read_text() if p.exists() else None

    def set_asr(self, params: dict, text: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path("asr", params, "txt").write_text(text)

    # ---- Tensors ----
    def get_tensor(self, stage: str, params: dict) -> torch.Tensor | None:
        p = self._path(stage, params, "pt")
        if not p.exists():
            return None
        return torch.load(p, map_location="cpu", weights_only=True)

    def set_tensor(self, stage: str, params: dict, t: torch.Tensor) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        torch.save(t.detach().cpu(), self._path(stage, params, "pt"))
