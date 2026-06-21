from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import transformers

from cache import StageCache
from cascading_classifier import CascadeResult
from multimodal_backbone import MultimodalBackbone
from taxonomy import build_classifier, load_taxonomy
from video_processor import OCRHit, VideoProcessor, get_device

# mlx-whisper is Apple-Silicon-only; absent on Linux / x86 macs / CI. The HF
# transformers pipeline below is the fallback for those environments.
try:
    import mlx_whisper as _mlx_whisper  # type: ignore[import-not-found]
except ImportError:
    _mlx_whisper = None


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
    # Wall-clock seconds per stage, populated when extract_features runs with
    # profiling enabled. MPS-touching stages call torch.mps.synchronize() before
    # reading the timer so the value reflects actual GPU completion, not just
    # kernel launch. Empty in non-profile paths.
    stage_timings: dict[str, float] = field(default_factory=dict)


@dataclass
class PreBackbone:
    """Producer-side output for the batched extraction path: everything needed
    to run the GPU backbones, minus the backbones themselves. Picklable so it
    can cross a process boundary from a producer worker to the GPU consumer.

    If ``cached_feature`` is set, all three embeddings were already on disk and
    no GPU work is needed. Otherwise ``frames``/``wave`` carry the raw inputs
    and the ``*_params`` carry the cache keys the consumer writes back under.
    """
    text: str
    frames_sampled: int
    ocr_text: str
    asr_text: str
    frames: "object | None" = None          # (T,H,W,3) uint8 BGR ndarray
    wave: "object | None" = None            # mono float32 ndarray @ 16k
    cached_feature: torch.Tensor | None = None
    text_emb: torch.Tensor | None = None
    audio_emb: torch.Tensor | None = None
    video_emb: torch.Tensor | None = None
    text_emb_params: dict | None = None
    audio_emb_params: dict | None = None
    video_emb_params: dict | None = None
    video_path: str | None = None


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
        caption_min_width_frac: float = 0.05,   # ≥5% frame width — small text won't
                                                # span 25%; raise for wide-only filters.
        caption_min_height_frac: float | None = None,  # derived from caption_min_pt
                                                       # below if not explicitly set.
        caption_min_pt: float = 18.0,           # text height floor in points; 18pt
                                                # at 1080p ≈ 24px ≈ 2.2% frame height.
        caption_reference_height_px: int = 1080,  # the frame height the pt mapping
                                                  # assumes; override per source res.
        caption_min_chars: int = 3,
        ocr_dedup_threshold: float = 8.0,
        router_weights_dir: Path | None = Path("models"),
        cache_root: Path | None = Path("data/cache"),
        force_leaf: bool = False,
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
        self.asr_model_name = asr_model
        # Prefer mlx-whisper on Apple Silicon — runs Whisper natively on the
        # ANE/GPU and is ~3-5x faster than HF transformers Whisper on M1. The
        # HF pipeline is only loaded as a fallback when mlx-whisper is absent.
        # Note: backbone.audio_encoder still uses HF Whisper-encoder weights
        # for the audio embedding; mlx-whisper doesn't expose hidden states.
        if _mlx_whisper is not None:
            self.asr_engine = "mlx"
            self._mlx_whisper_repo = "mlx-community/whisper-tiny"
            self.asr = None
        else:
            self.asr_engine = "hf"
            self._mlx_whisper_repo = None
            self.asr = transformers.pipeline(
                "automatic-speech-recognition",
                model=asr_model,
                device=self.device,
            )
        print(f"[pipeline] ASR engine: {self.asr_engine}", file=sys.stderr)
        # Per-stage on-disk cache. Set to None to disable (e.g. for benchmarks
        # that need cold-cache numbers). Each stage hashes its own params, so
        # changing sample_fps invalidates only OCR + video_emb, not ASR.
        self.cache_root = Path(cache_root) if cache_root is not None else None
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
            force_leaf=force_leaf,
        )

        self.caption_only = caption_only
        self.caption_min_y = caption_min_y
        self.caption_max_y = caption_max_y
        self.caption_min_width_frac = caption_min_width_frac
        # Derive height fraction from points when not explicitly given.
        # Mapping: 1pt = 4/3 px at 96-DPI standard rendering. Frame height
        # serves as the canvas reference; OCR bbox heights are normalized to
        # the actual frame's height at filter time, so this is invariant to
        # the actual capture resolution as long as captions were authored at
        # the reference scale.
        if caption_min_height_frac is None:
            caption_min_height_frac = (caption_min_pt * 4.0 / 3.0) / caption_reference_height_px
        self.caption_min_height_frac = caption_min_height_frac
        self.caption_min_pt = caption_min_pt
        self.caption_reference_height_px = caption_reference_height_px
        self.caption_min_chars = caption_min_chars
        # Mean per-pixel absdiff threshold (uint8 grayscale, 64x64 downsample)
        # below which a frame is considered a near-duplicate of the last frame
        # we OCR'd, and OCR is skipped. 0 disables dedup. ~8.0 ≈ 3% mean change
        # — fine for static-caption TikTok-style videos; raise for noisier
        # camera content.
        self.ocr_dedup_threshold = ocr_dedup_threshold

    def extract_features(
        self,
        video_path: Path,
        *,
        profile: bool = False,
        use_cache: bool = True,
    ) -> FeatureExtraction:
        """Run the backbone on a video; return the fused 1344-dim feature plus
        the OCR/ASR metadata. Same featurization classify() uses internally —
        factored out so training-time caching can reuse it without invoking
        the (untrained, noise-producing) classifier.

        profile=True populates FeatureExtraction.stage_timings with per-stage
        wall-clock seconds. MPS stages are synchronized before the timer reads,
        so values reflect GPU completion — at the cost of breaking Python/MPS
        pipelining. Leave False in production paths.

        use_cache=True reads/writes per-stage artifacts under self.cache_root.
        A warm cache reduces a ~60s call to <1s. Pass False to force cold
        recomputation (e.g. when benchmarking a backbone swap)."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        timings: dict[str, float] | None = {} if profile else None
        cache = StageCache(self.cache_root, video_path) if (use_cache and self.cache_root is not None) else None

        # Stage param sets — anything that affects a stage's output goes in
        # its dict. Different params produce different cache filenames, so
        # tweaks coexist on disk and don't trash prior runs.
        ocr_params = {
            "sample_fps": self.vp.sample_fps,
            "ocr_dedup_threshold": self.ocr_dedup_threshold,
            "ocr_min_confidence": self.vp.ocr_min_confidence,
            "caption_only": self.caption_only,
            "caption_min_y": self.caption_min_y,
            "caption_max_y": self.caption_max_y,
            "caption_min_width_frac": self.caption_min_width_frac,
            "caption_min_height_frac": self.caption_min_height_frac,
            "caption_min_chars": self.caption_min_chars,
        }
        asr_params = {
            "asr_model": self.asr_model_name,
            "asr_engine": self.asr_engine,
        }
        audio_emb_params = {"audio_model": self.backbone.audio_model_name}
        video_emb_params = {
            "video_model": self.backbone.video_model_name,
            "sample_fps": self.vp.sample_fps,
        }
        # text_emb depends on the actual text content (asr + ocr concatenation),
        # which we don't know until ASR + OCR are resolved. Built below.

        # ---- Cache reads up front so we know what to compute ----
        ocr_cached = cache.get_ocr(ocr_params) if cache else None
        asr_text = cache.get_asr(asr_params) if cache else None
        audio_emb = cache.get_tensor("audio_emb", audio_emb_params) if cache else None
        video_emb = cache.get_tensor("video_emb", video_emb_params) if cache else None

        if ocr_cached is not None:
            frames_sampled, ocr_hits = ocr_cached
        else:
            frames_sampled, ocr_hits = 0, []

        needs_frames = (ocr_cached is None) or (video_emb is None)
        needs_wav = (asr_text is None) or (audio_emb is None)

        with tempfile.TemporaryDirectory(prefix="iab_pipe_") as tmp:
            wav_path = Path(tmp) / "audio.wav"
            if needs_wav:
                t0 = time.perf_counter() if profile else 0.0
                self.vp.extract_audio(video_path, wav_path)
                if profile:
                    timings["extract_audio_s"] = time.perf_counter() - t0

            frames: np.ndarray | None = None
            if needs_frames:
                if ocr_cached is None:
                    # Full scan: produces both frames and OCR.
                    frames, ocr_hits = self._scan_video(video_path, timings=timings)
                    frames_sampled = frames.shape[0]
                    if cache:
                        cache.set_ocr(ocr_params, frames_sampled, ocr_hits)
                else:
                    # OCR cached, but video_emb missing — re-decode frames only,
                    # skipping the OCR call per frame entirely.
                    frames = self._iter_frames_only(video_path, timings=timings)
                    # frames_sampled already populated from cache; sanity-check.

            # ASR
            if asr_text is None:
                t0 = time.perf_counter() if profile else 0.0
                asr_text = self._transcribe(wav_path)
                if profile:
                    self._mps_sync()
                    timings["transcribe_s"] = time.perf_counter() - t0
                if cache:
                    cache.set_asr(asr_params, asr_text)

            ocr_text = " ".join(h.text for h in ocr_hits).strip()
            text = f"{asr_text} {ocr_text}".strip()

            # text_emb (cached on the actual text content + model)
            text_emb_params = {
                "text_model": self.backbone.text_model_name,
                "max_length": 256,
                "text_input": text,
            }
            text_emb = cache.get_tensor("text_emb", text_emb_params) if cache else None
            if text_emb is None:
                t0 = time.perf_counter() if profile else 0.0
                if text:
                    text_emb = self.backbone.encode_text([text], max_length=256)
                else:
                    text_emb = torch.zeros(
                        1, self.backbone.embed_dims["text"], device=self.device,
                    )
                if profile:
                    self._mps_sync()
                    timings["encode_text_s"] = time.perf_counter() - t0
                if cache:
                    cache.set_tensor("text_emb", text_emb_params, text_emb)
            else:
                text_emb = text_emb.to(self.device)

            # audio_emb
            if audio_emb is None:
                t0 = time.perf_counter() if profile else 0.0
                audio_emb = self.backbone.encode_audio([wav_path])
                if profile:
                    self._mps_sync()
                    timings["encode_audio_s"] = time.perf_counter() - t0
                if cache:
                    cache.set_tensor("audio_emb", audio_emb_params, audio_emb)
            else:
                audio_emb = audio_emb.to(self.device)

            # video_emb
            if video_emb is None:
                t0 = time.perf_counter() if profile else 0.0
                video_emb = self.backbone.encode_video(frames[None])    # (1, T, H, W, 3)
                if profile:
                    self._mps_sync()
                    timings["encode_video_s"] = time.perf_counter() - t0
                if cache:
                    cache.set_tensor("video_emb", video_emb_params, video_emb)
            else:
                video_emb = video_emb.to(self.device)

            fused = torch.cat([text_emb, audio_emb, video_emb], dim=-1)

        return FeatureExtraction(
            feature=fused,
            frames_sampled=frames_sampled,
            ocr_hits=ocr_hits,
            ocr_text=ocr_text,
            asr_text=asr_text,
            text_input=text,
            stage_timings=timings or {},
        )

    # ------------------------------------------------------------------
    # Batched extraction path (producer/consumer)
    #
    # extract_features() does decode+OCR+ASR (CPU/OCR-bound, ~85% of time)
    # and the GPU backbone encodes for ONE video. To keep a big GPU saturated
    # we split it: many producer processes run extract_pre_backbone() (the
    # heavy part), and a single consumer batches encode_backbone() across
    # videos. See scripts/build_features.py --batch-size.
    # ------------------------------------------------------------------
    def extract_pre_backbone(self, video_path: Path, *, use_cache: bool = True) -> PreBackbone:
        """Everything up to the GPU backbones: decode frames, OCR, ASR, audio
        waveform, and the resolved text. Cache-aware for OCR/ASR. If all three
        per-video embeddings are already cached, returns them directly (frames /
        wave left None) so the consumer can skip re-encoding on resume."""
        import numpy as np
        import soundfile as sf

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        cache = StageCache(self.cache_root, video_path) if (use_cache and self.cache_root is not None) else None

        ocr_params = {
            "sample_fps": self.vp.sample_fps,
            "ocr_dedup_threshold": self.ocr_dedup_threshold,
            "ocr_min_confidence": self.vp.ocr_min_confidence,
            "caption_only": self.caption_only,
            "caption_min_y": self.caption_min_y,
            "caption_max_y": self.caption_max_y,
            "caption_min_width_frac": self.caption_min_width_frac,
            "caption_min_height_frac": self.caption_min_height_frac,
            "caption_min_chars": self.caption_min_chars,
        }
        asr_params = {"asr_model": self.asr_model_name, "asr_engine": self.asr_engine}
        audio_emb_params = {"audio_model": self.backbone.audio_model_name}
        video_emb_params = {
            "video_model": self.backbone.video_model_name,
            "sample_fps": self.vp.sample_fps,
        }

        ocr_cached = cache.get_ocr(ocr_params) if cache else None
        asr_text = cache.get_asr(asr_params) if cache else None
        frames_sampled, ocr_hits = ocr_cached if ocr_cached is not None else (0, [])

        # Resolve OCR.
        frames: "np.ndarray | None" = None
        if ocr_cached is None:
            frames, ocr_hits = self._scan_video(video_path)
            frames_sampled = frames.shape[0]
            if cache:
                cache.set_ocr(ocr_params, frames_sampled, ocr_hits)

        ocr_text = " ".join(h.text for h in ocr_hits).strip()

        # Resolve ASR (needs the wav). Read the waveform once; reuse for audio emb.
        wave: "np.ndarray | None" = None
        with tempfile.TemporaryDirectory(prefix="iab_pre_") as tmp:
            wav_path = Path(tmp) / "audio.wav"
            audio_extracted = False
            if asr_text is None:
                self.vp.extract_audio(video_path, wav_path)
                audio_extracted = True
                asr_text = self._transcribe(wav_path)
                if cache:
                    cache.set_asr(asr_params, asr_text)

            text = f"{asr_text} {ocr_text}".strip()
            text_emb_params = {
                "text_model": self.backbone.text_model_name,
                "max_length": 256,
                "text_input": text,
            }
            text_emb = cache.get_tensor("text_emb", text_emb_params) if cache else None
            audio_emb = cache.get_tensor("audio_emb", audio_emb_params) if cache else None
            video_emb = cache.get_tensor("video_emb", video_emb_params) if cache else None

            if text_emb is not None and audio_emb is not None and video_emb is not None:
                # Fully cached — no GPU work needed for this video.
                fused = torch.cat(
                    [text_emb.cpu(), audio_emb.cpu(), video_emb.cpu()], dim=-1
                ).squeeze(0)
                return PreBackbone(
                    text=text, frames=None, wave=None, frames_sampled=frames_sampled,
                    ocr_text=ocr_text, asr_text=asr_text, cached_feature=fused,
                )

            # Need to encode → make sure we have frames + waveform on hand.
            if video_emb is None and frames is None:
                frames = self._iter_frames_only(video_path)
            if audio_emb is None:
                if not audio_extracted:
                    self.vp.extract_audio(video_path, wav_path)
                w, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
                if w.ndim == 2:
                    w = w.mean(axis=1)
                wave = w

        return PreBackbone(
            text=text, frames=frames, wave=wave, frames_sampled=frames_sampled,
            ocr_text=ocr_text, asr_text=asr_text, cached_feature=None,
            text_emb=text_emb, audio_emb=audio_emb, video_emb=video_emb,
            text_emb_params=text_emb_params, audio_emb_params=audio_emb_params,
            video_emb_params=video_emb_params, video_path=str(video_path),
        )

    def encode_backbone_batch(self, items: list[PreBackbone]) -> list[torch.Tensor]:
        """Consumer half: batch the three encoders across `items`, fuse, and
        return one (in_dim,) CPU feature per item (same order). Writes per-video
        emb cache so a later resume hits warm cache. Items whose feature was
        fully cached pass straight through."""
        order_feats: list[torch.Tensor | None] = [None] * len(items)

        # Indices that still need each modality computed.
        need = [i for i, it in enumerate(items) if it.cached_feature is None]
        for i, it in enumerate(items):
            if it.cached_feature is not None:
                order_feats[i] = it.cached_feature

        if need:
            # ---- text (batched) ----
            texts = [items[i].text for i in need]
            text_embs = self.backbone.encode_text(
                [t if t else " " for t in texts], max_length=256
            )  # (k, Dt)
            # ---- audio (batched from arrays) ----
            import numpy as np
            waves = [items[i].wave for i in need]
            audio_embs = self.backbone.encode_audio_arrays(waves)  # (k, Da)
            # ---- video (batched, variable T) ----
            frames_list = [items[i].frames for i in need]
            video_embs = self.backbone.encode_video_batch(frames_list)  # (k, Dv)

            for j, i in enumerate(need):
                it = items[i]
                t_e = text_embs[j:j + 1]
                a_e = audio_embs[j:j + 1]
                v_e = video_embs[j:j + 1]
                fused = torch.cat([t_e, a_e, v_e], dim=-1).squeeze(0).detach().cpu()
                order_feats[i] = fused
                # Warm the per-video cache for cheap resumes.
                if self.cache_root is not None and it.video_path:
                    cache = StageCache(self.cache_root, Path(it.video_path))
                    cache.set_tensor("text_emb", it.text_emb_params, t_e)
                    cache.set_tensor("audio_emb", it.audio_emb_params, a_e)
                    cache.set_tensor("video_emb", it.video_emb_params, v_e)

        return [f for f in order_feats]  # type: ignore[return-value]

    def _iter_frames_only(
        self,
        video_path: Path,
        *,
        timings: dict[str, float] | None = None,
    ) -> np.ndarray:
        # Decode + collect frames without the OCR call. Used when OCR results
        # are cached but video_emb still needs to be computed.
        frames: list[np.ndarray] = []
        iter_total = 0.0
        gen = self.vp.iter_frames(video_path)
        while True:
            t0 = time.perf_counter() if timings is not None else 0.0
            try:
                _, _, frame = next(gen)
            except StopIteration:
                break
            if timings is not None:
                iter_total += time.perf_counter() - t0
            frames.append(frame)
        if not frames:
            raise RuntimeError(f"no frames decoded from {video_path}")
        if timings is not None:
            timings["iter_frames_s"] = iter_total
        return np.stack(frames)

    def _mps_sync(self) -> None:
        # Force any pending MPS kernels to complete so the next perf_counter
        # read reflects actual GPU work, not just kernel launch dispatch. No-op
        # on CPU / CUDA.
        if self.device == "mps":
            torch.mps.synchronize()

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

    def _scan_video(
        self,
        video_path: Path,
        *,
        timings: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, list[OCRHit]]:
        # When timings is provided, accumulate per-frame iter / OCR / dedup
        # seconds + counts. iter_frames is a generator so we time around the
        # `next()` resumption rather than the whole loop body.
        #
        # Dedup: each new frame is compared against the last frame we actually
        # ran OCR on (NOT the immediate previous frame, to avoid slow drift
        # silently accumulating past the threshold). Comparison is mean absdiff
        # on a 64x64 grayscale downsample — ~6000x cheaper than full-res diff
        # and good enough to distinguish "same caption" from "scene cut".
        frames: list[np.ndarray] = []
        ocr_hits: list[OCRHit] = []
        iter_total = 0.0
        ocr_total = 0.0
        dedup_total = 0.0
        ocr_calls = 0
        dedup_skipped = 0
        last_ocr_thumb: np.ndarray | None = None
        gen = self.vp.iter_frames(video_path)
        while True:
            t_iter = time.perf_counter() if timings is not None else 0.0
            try:
                idx, ts, frame = next(gen)
            except StopIteration:
                break
            if timings is not None:
                iter_total += time.perf_counter() - t_iter
            frames.append(frame)
            h, w = frame.shape[:2]

            # Dedup gate.
            skip_ocr = False
            if self.ocr_dedup_threshold > 0:
                t_dd = time.perf_counter() if timings is not None else 0.0
                thumb = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    (64, 64), interpolation=cv2.INTER_AREA,
                )
                if last_ocr_thumb is not None:
                    diff = float(cv2.absdiff(thumb, last_ocr_thumb).mean())
                    if diff < self.ocr_dedup_threshold:
                        skip_ocr = True
                if timings is not None:
                    dedup_total += time.perf_counter() - t_dd
            else:
                thumb = None

            if skip_ocr:
                if timings is not None:
                    dedup_skipped += 1
                continue

            t_ocr = time.perf_counter() if timings is not None else 0.0
            results = self.vp.ocr_frame(frame)
            if timings is not None:
                self._mps_sync()
                ocr_total += time.perf_counter() - t_ocr
                ocr_calls += 1
            last_ocr_thumb = thumb
            for bbox, text, conf in results:
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
        if timings is not None:
            timings["iter_frames_s"] = iter_total
            timings["ocr_s"] = ocr_total
            timings["dedup_s"] = dedup_total
            timings["ocr_n_calls"] = float(ocr_calls)
            timings["dedup_skipped"] = float(dedup_skipped)
        return np.stack(frames), ocr_hits

    def _transcribe(self, wav_path: Path) -> str:
        if self.asr_engine == "mlx":
            # mlx-whisper handles long-form chunking internally. Returns the
            # same shape ({"text": ..., "segments": [...]}) as openai-whisper.
            out = _mlx_whisper.transcribe(
                str(wav_path), path_or_hf_repo=self._mlx_whisper_repo,
            )
            return (out.get("text") or "").strip()
        # HF fallback path. return_timestamps=True is required once audio
        # exceeds 30s — Whisper switches to long-form chunked generation which
        # predicts timestamp tokens. The top-level "text" field still holds
        # the concatenated transcription; per-chunk stamps live under "chunks".
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
