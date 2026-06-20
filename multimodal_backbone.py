from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    WhisperModel,
)

from video_processor import get_device


# ---------------------------------------------------------------------------
# Video transformer — factorized space-time attention, sized for 8GB MacBooks.
# ---------------------------------------------------------------------------
class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VideoTransformer(nn.Module):
    """Factorized space-then-time ViT.

    Two stages keep the attention matrix shape O(N²) per frame and O(T²) across
    time, rather than O((N·T)²) for joint space-time. For T=8 frames at 224x224
    with patch=16 (N=196), joint attention scores would be 1568² ≈ 10MB per head
    per layer in fp32 — factorized stays at 196² ≈ 150KB per head, keeping the
    whole activation budget well under 1GB on an 8GB machine.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        spatial_depth: int = 2,
        temporal_depth: int = 2,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        max_frames: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must divide patch_size"

        grid = img_size // patch_size
        num_patches = grid * grid

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.max_frames = max_frames

        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
        )
        self.spatial_pos = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_frames, embed_dim))

        self.spatial_blocks = nn.ModuleList(
            [_TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
             for _ in range(spatial_depth)]
        )
        self.temporal_blocks = nn.ModuleList(
            [_TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
             for _ in range(temporal_depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, 3, H, W)
        B, T, C, H, W = frames.shape
        if T > self.max_frames:
            idx = torch.linspace(0, T - 1, self.max_frames, device=frames.device).long()
            frames = frames.index_select(1, idx)
            T = self.max_frames

        x = self.patch_embed(frames.reshape(B * T, C, H, W))
        x = x.flatten(2).transpose(1, 2)
        x = x + self.spatial_pos
        for blk in self.spatial_blocks:
            x = blk(x)
        x = x.mean(dim=1)

        x = x.view(B, T, -1) + self.temporal_pos[:, :T]
        for blk in self.temporal_blocks:
            x = blk(x)

        x = self.norm(x).mean(dim=1)
        return x


# ---------------------------------------------------------------------------
# Multimodal backbone — three encoders that share a single device.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModalityEmbeddings:
    text: torch.Tensor | None
    audio: torch.Tensor | None
    video: torch.Tensor | None


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


class MultimodalBackbone(nn.Module):
    def __init__(
        self,
        text_model_name: str = "distilbert-base-uncased",
        audio_model_name: str = "openai/whisper-tiny",
        video_model_name: str = "facebook/dino-vits16",
        use_pretrained_video: bool = True,
        video_kwargs: dict | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.device = get_device()
        self.use_pretrained_video = use_pretrained_video
        # Stored so callers (e.g. cache.py) can include them in cache keys —
        # different model means different embedding, must invalidate.
        self.text_model_name = text_model_name
        self.audio_model_name = audio_model_name
        self.video_model_name = video_model_name if use_pretrained_video else "custom_video_transformer"

        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name).to(self.device)

        # Encoder-only — drop the decoder after extraction to save ~60% of Whisper weights
        self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_name)
        whisper = WhisperModel.from_pretrained(audio_model_name)
        self.audio_encoder = whisper.encoder.to(self.device)
        del whisper

        if use_pretrained_video:
            # DINO ViT-S/16 by default — self-supervised, 384-dim CLS features.
            # The image processor handles the model's exact resize/normalize stats;
            # we don't reuse the legacy IMAGENET buffers in this path.
            self.video_image_processor = AutoImageProcessor.from_pretrained(video_model_name)
            self.video_encoder = AutoModel.from_pretrained(video_model_name).to(self.device)
            video_dim = self.video_encoder.config.hidden_size
        else:
            self.video_image_processor = None
            self.video_encoder = VideoTransformer(**(video_kwargs or {})).to(self.device)
            video_dim = self.video_encoder.embed_dim

        # Buffers stay registered for the legacy path — harmless when unused.
        self.register_buffer("_img_mean", _IMAGENET_MEAN.to(self.device), persistent=False)
        self.register_buffer("_img_std", _IMAGENET_STD.to(self.device), persistent=False)

        self.embed_dims = {
            "text": self.text_encoder.config.hidden_size,
            "audio": self.audio_encoder.config.d_model,
            "video": video_dim,
        }

        if freeze:
            self.freeze_pretrained()

        # fp16 autocast on CUDA: backbones are frozen + inference-only, so no
        # NaN risk from training dynamics. Halves activation memory and gives
        # ~1.5-2x on backbone forwards. MPS autocast is edge-case-y so we
        # leave it disabled there; downstream fused/router math stays fp32.
        self.amp_enabled = self.device == "cuda"

    def _amp_ctx(self):
        if self.amp_enabled:
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def freeze_pretrained(self) -> None:
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        for p in self.audio_encoder.parameters():
            p.requires_grad_(False)
        self.text_encoder.train(False)
        self.audio_encoder.train(False)
        if self.use_pretrained_video:
            for p in self.video_encoder.parameters():
                p.requires_grad_(False)
            self.video_encoder.train(False)

    @torch.inference_mode()
    def encode_text(self, texts: list[str], max_length: int = 128) -> torch.Tensor:
        tokens = self.text_tokenizer(
            texts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        with self._amp_ctx():
            out = self.text_encoder(**tokens)
            cls = out.last_hidden_state[:, 0]
        return cls.float()

    @torch.inference_mode()
    def encode_audio(self, wav_paths: list[Path]) -> torch.Tensor:
        waves: list[np.ndarray] = []
        for p in wav_paths:
            audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr != self.audio_feature_extractor.sampling_rate:
                raise ValueError(
                    f"{p} has sr={sr}, expected {self.audio_feature_extractor.sampling_rate}. "
                    f"Re-extract with VideoProcessor(audio_sample_rate={self.audio_feature_extractor.sampling_rate})."
                )
            waves.append(audio)

        features = self.audio_feature_extractor(
            waves,
            sampling_rate=self.audio_feature_extractor.sampling_rate,
            return_tensors="pt",
        ).input_features.to(self.device)

        with self._amp_ctx():
            out = self.audio_encoder(features)
            pooled = out.last_hidden_state.mean(dim=1)
        return pooled.float()

    @torch.inference_mode()
    def encode_video(self, frames_bgr: np.ndarray | torch.Tensor) -> torch.Tensor:
        # frames_bgr: (B, T, H, W, 3) uint8 BGR, or a pre-built float tensor.
        if self.use_pretrained_video:
            return self._encode_video_pretrained(frames_bgr)
        return self._encode_video_legacy(frames_bgr)

    def _encode_video_pretrained(self, frames_bgr: np.ndarray | torch.Tensor) -> torch.Tensor:
        # The HF image processor wants list-of-arrays/PIL on CPU. Convert BGR→RGB
        # on-device first, then move to CPU for preprocessing — exactly one
        # device→host transfer per video. The processor handles model-specific
        # resize/center-crop/normalize so we don't risk drift from DINO's stats.
        if isinstance(frames_bgr, torch.Tensor):
            x = frames_bgr.detach().cpu().numpy()
        else:
            x = frames_bgr
        if x.dtype != np.uint8:
            x = (np.clip(x, 0, 1) * 255).astype(np.uint8) if x.max() <= 1.0 else x.astype(np.uint8)

        B, T = x.shape[:2]
        rgb = x[..., [2, 1, 0]]                          # BGR → RGB, copy
        flat = list(rgb.reshape(B * T, *rgb.shape[2:]))  # list of (H, W, 3) uint8

        inputs = self.video_image_processor(images=flat, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)

        with self._amp_ctx():
            out = self.video_encoder(pixel_values=pixel_values)
            cls = out.last_hidden_state[:, 0]            # (B*T, D)
            pooled = cls.view(B, T, -1).mean(dim=1)      # (B, D)
        return pooled.float()

    def _encode_video_legacy(self, frames_bgr: np.ndarray | torch.Tensor) -> torch.Tensor:
        # One numpy→device hop here; keep every subsequent op on-device so we
        # don't ping-pong the unified-memory buffer.
        if isinstance(frames_bgr, np.ndarray):
            x = torch.from_numpy(frames_bgr).to(self.device, non_blocking=True)
        else:
            x = frames_bgr.to(self.device, non_blocking=True)

        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)

        x = x[..., [2, 1, 0]].permute(0, 1, 4, 2, 3).contiguous()

        target = self.video_encoder.img_size
        if x.shape[-1] != target or x.shape[-2] != target:
            B, T = x.shape[:2]
            x = F.interpolate(
                x.reshape(B * T, 3, x.shape[-2], x.shape[-1]),
                size=(target, target), mode="bilinear", align_corners=False,
            ).reshape(B, T, 3, target, target)

        x = (x - self._img_mean) / self._img_std
        with self._amp_ctx():
            out = self.video_encoder(x)
        return out.float()

    def forward(
        self,
        texts: list[str] | None = None,
        wav_paths: list[Path] | None = None,
        frames: np.ndarray | torch.Tensor | None = None,
    ) -> ModalityEmbeddings:
        return ModalityEmbeddings(
            text=self.encode_text(texts) if texts else None,
            audio=self.encode_audio(wav_paths) if wav_paths else None,
            video=self.encode_video(frames) if frames is not None else None,
        )
