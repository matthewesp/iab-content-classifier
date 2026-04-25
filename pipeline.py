from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import transformers

from cascading_classifier import CascadeResult
from multimodal_backbone import MultimodalBackbone
from taxonomy import build_classifier, load_taxonomy
from video_processor import OCRHit, VideoProcessor, get_device


@dataclass
class FeatureExtraction:
    """Output of VideoClassifier.extract_features — fused 1344-dim feature plus
    the metadata that produced it. Used both by classify() and by training-time
    feature caching."""
    feature: torch.Tensor          # (1, in_dim)
    frames_sampled: int
    ocr_hits: list[OCRHit]
    ocr_text: str
    asr_text: str
    text_input: str


class VideoClassifier:
    """End-to-end mp4 → IAB category prediction.

    Owns four stages, all on MPS:
      1. VideoProcessor      — ffmpeg audio + cv2 frame sampling + EasyOCR
      2. Whisper-tiny ASR    — transformers.pipeline for spoken transcription
      3. MultimodalBackbone  — DistilBERT / Whisper-encoder / VideoTransformer
      4. CascadingClassifier — n-tier lazy routers over IAB v3.1 taxonomy
    """

    def __init__(
        self,
        taxonomy_path: Path = Path("content_taxonomy_3.1.tsv"),
        confidence_threshold: float = 0.8,
        max_active_routers: int = 16,
        sample_fps: float = 1.0,
        ocr_min_confidence: float = 0.4,
        asr_model: str = "openai/whisper-tiny",
        warn_untrained: bool = True,
        caption_only: bool = True,
        caption_min_y: float = 0.20,            # default band targets TikTok-style
        caption_max_y: float = 0.70,            # captions: upper-middle of frame.
        caption_min_width_frac: float = 0.25,   # large-only: ≥25% frame width
        caption_min_height_frac: float = 0.06,  # large-only: ≥6% frame height (font size)
        caption_min_chars: int = 3,
        router_weights_dir: Path | None = Path("models"),
    ) -> None:
        self.device = get_device()

        weights_present = router_weights_dir is not None and router_weights_dir.exists() and any(router_weights_dir.glob("*.pt"))
        if warn_untrained and not weights_present:
            print(
                "[pipeline] Warning: no trained router checkpoints found in "
                f"{router_weights_dir} — predicted categories are softmax noise "
                "until you run scripts/train_classifier.py.",
                file=sys.stderr,
            )

        self.vp = VideoProcessor(
            sample_fps=sample_fps,
            ocr_min_confidence=ocr_min_confidence,
        )
        self.backbone = MultimodalBackbone()
        self.asr = transformers.pipeline(
            "automatic-speech-recognition",
            model=asr_model,
            device=self.device,
        )
        self.taxonomy = load_taxonomy(taxonomy_path)
        self.in_dim = sum(self.backbone.embed_dims.values())   # 768 + 384 + 192 = 1344
        # router_weights_dir is silently ignored if it doesn't exist yet (no
        # training has been run) — routers stay random-init until then.
        weights_dir = router_weights_dir if (router_weights_dir and router_weights_dir.exists()) else None
        self.classifier = build_classifier(
            self.taxonomy,
            in_dim=self.in_dim,
            confidence_threshold=confidence_threshold,
            max_active_routers=max_active_routers,
            router_weights_dir=weights_dir,
        )

        self.caption_only = caption_only
        self.caption_min_y = caption_min_y
        self.caption_max_y = caption_max_y
        self.caption_min_width_frac = caption_min_width_frac
        self.caption_min_height_frac = caption_min_height_frac
        self.caption_min_chars = caption_min_chars

    def extract_features(self, video_path: Path) -> FeatureExtraction:
        """Run the backbone on a video; return the fused 1344-dim feature plus
        the OCR/ASR metadata. Same featurization classify() uses internally —
        factored out so training-time caching can reuse it without invoking
        the (untrained, noise-producing) classifier."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        with tempfile.TemporaryDirectory(prefix="iab_pipe_") as tmp:
            wav_path = Path(tmp) / "audio.wav"
            self.vp.extract_audio(video_path, wav_path)

            frames, ocr_hits = self._scan_video(video_path)

            asr_text = self._transcribe(wav_path)
            ocr_text = " ".join(h.text for h in ocr_hits).strip()
            text = f"{asr_text} {ocr_text}".strip()

            if text:
                text_emb = self.backbone.encode_text([text], max_length=256)
            else:
                text_emb = torch.zeros(
                    1, self.backbone.embed_dims["text"], device=self.device,
                )
            audio_emb = self.backbone.encode_audio([wav_path])
            video_emb = self.backbone.encode_video(frames[None])    # (1, T, H, W, 3)
            fused = torch.cat([text_emb, audio_emb, video_emb], dim=-1)

        return FeatureExtraction(
            feature=fused,
            frames_sampled=frames.shape[0],
            ocr_hits=ocr_hits,
            ocr_text=ocr_text,
            asr_text=asr_text,
            text_input=text,
        )

    def classify(self, video_path: Path) -> dict:
        ext = self.extract_features(video_path)
        [result] = self.classifier(ext.feature)
        return self._format(
            Path(video_path), ext.frames_sampled, ext.asr_text,
            ext.ocr_hits, ext.ocr_text, ext.text_input, result,
        )

    def _is_caption_like(self, bbox, frame_h: int, frame_w: int, text: str) -> bool:
        # bbox is EasyOCR's 4-point quad. We derive width, height, and vertical
        # center from the extremes — robust to slight rotation. Four cheap gates:
        #   1. length — OCR garbage is often 1-2 chars.
        #   2. width  — wide captions vs narrow logos/labels.
        #   3. height — actual font size proxy; "large only" means thick text,
        #               not just text spread across many narrow characters.
        #   4. y-band — bbox center y must fall in [min_y, max_y]. Defaults
        #               target TikTok-style upper-middle captions; set
        #               caption_min_y=0 / caption_max_y=1 to disable.
        if len(text.strip()) < self.caption_min_chars:
            return False
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        if bw / frame_w < self.caption_min_width_frac:
            return False
        if bh / frame_h < self.caption_min_height_frac:
            return False
        cy = (min(ys) + max(ys)) / 2
        cy_frac = cy / frame_h
        if cy_frac < self.caption_min_y or cy_frac > self.caption_max_y:
            return False
        return True

    def _scan_video(self, video_path: Path) -> tuple[np.ndarray, list[OCRHit]]:
        frames: list[np.ndarray] = []
        ocr_hits: list[OCRHit] = []
        for idx, ts, frame in self.vp.iter_frames(video_path):
            frames.append(frame)
            h, w = frame.shape[:2]
            for bbox, text, conf in self.vp.ocr_frame(frame):
                if conf < self.vp.ocr_min_confidence:
                    continue
                if self.caption_only and not self._is_caption_like(bbox, h, w, text):
                    continue
                ocr_hits.append(OCRHit(
                    frame_index=idx,
                    timestamp_s=ts,
                    text=text,
                    confidence=float(conf),
                    bbox=[(int(x), int(y)) for x, y in bbox],
                ))
        if not frames:
            raise RuntimeError(f"no frames decoded from {video_path}")
        return np.stack(frames), ocr_hits

    def _transcribe(self, wav_path: Path) -> str:
        # return_timestamps=True is required once audio exceeds 30s — Whisper
        # switches to long-form chunked generation which predicts timestamp
        # tokens. The top-level "text" field still holds the concatenated
        # transcription; the per-chunk timestamps live under "chunks".
        out = self.asr(str(wav_path), return_timestamps=True)
        if isinstance(out, dict) and "text" in out:
            return out["text"].strip()
        return ""

    def _format(
        self,
        video_path: Path,
        n_frames: int,
        asr_text: str,
        ocr_hits: list[OCRHit],
        ocr_text: str,
        text_input: str,
        result: CascadeResult,
    ) -> dict:
        return {
            "video_path": str(video_path),
            "video": {
                "frames_sampled": n_frames,
                "sample_fps": self.vp.sample_fps,
            },
            "audio": {"sample_rate": self.vp.audio_sample_rate},
            "ocr": {"hits": len(ocr_hits), "text": ocr_text},
            "asr": {"text": asr_text},
            "text_input": text_input,
            "classification": {
                "final_id": result.final_id,
                "final_name": result.final_name,
                "final_confidence": result.final_confidence,
                "depth": result.depth,
                "id_path": result.id_path(),
                "name_path": result.name_path(),
                "steps": [
                    {
                        "parent_id": s.parent_id,
                        "id": s.predicted_id,
                        "name": s.predicted_name,
                        "confidence": s.confidence,
                        "escalated": s.escalated,
                    }
                    for s in result.path
                ],
            },
        }


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=VideoClassifier.__doc__)
    p.add_argument("video", type=Path)
    p.add_argument("--taxonomy", type=Path, default=Path("content_taxonomy_3.1.tsv"))
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--sample-fps", type=float, default=1.0)
    p.add_argument("--max-active-routers", type=int, default=16)
    args = p.parse_args()

    clf = VideoClassifier(
        taxonomy_path=args.taxonomy,
        confidence_threshold=args.threshold,
        sample_fps=args.sample_fps,
        max_active_routers=args.max_active_routers,
    )
    print(json.dumps(clf.classify(args.video), indent=2))


if __name__ == "__main__":
    _main()
