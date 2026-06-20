from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch

import easyocr


# AVFOUNDATION is the macOS-native backend (VideoToolbox HW decode) but it
# does not exist on Linux. FFMPEG is universally available — we use it on
# everything but darwin so the same code runs on Colab/Linux unchanged.
_VIDEO_CAPTURE_BACKEND = (
    cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_FFMPEG
)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass(frozen=True)
class OCRHit:
    frame_index: int
    timestamp_s: float
    text: str
    confidence: float
    bbox: list[tuple[int, int]]


@dataclass(frozen=True)
class VideoResult:
    video_path: Path
    audio_path: Path
    device: str
    frames_sampled: int
    ocr: list[OCRHit]


class VideoProcessor:
    def __init__(
        self,
        languages: list[str] | None = None,
        sample_fps: float = 1.0,
        ocr_min_confidence: float = 0.4,
        audio_sample_rate: int = 16_000,
    ) -> None:
        self.languages = languages or ["en"]
        self.sample_fps = sample_fps
        self.ocr_min_confidence = ocr_min_confidence
        self.audio_sample_rate = audio_sample_rate
        self.device = get_device()

        # EasyOCR 1.7+ auto-selects MPS when gpu=True and CUDA is absent.
        # quantize must be False off-CPU: int8 ops only exist for fbgemm/qnnpack.
        use_gpu = self.device != "cpu"
        self.reader = easyocr.Reader(
            self.languages,
            gpu=use_gpu,
            quantize=not use_gpu,
            verbose=False,
        )

    def extract_audio(
        self,
        video_path: Path,
        audio_path: Path | None = None,
    ) -> Path:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found on PATH")

        video_path = Path(video_path)
        audio_path = Path(audio_path) if audio_path else video_path.with_suffix(".wav")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", str(self.audio_sample_rate),
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            str(audio_path),
        ]
        subprocess.run(cmd, check=True)
        return audio_path

    def iter_frames(
        self,
        video_path: Path,
    ) -> Iterator[tuple[int, float, np.ndarray]]:
        # AVFOUNDATION on darwin (VideoToolbox HW decode), FFMPEG everywhere
        # else (Linux/Colab). See _VIDEO_CAPTURE_BACKEND at module top.
        cap = cv2.VideoCapture(str(video_path), _VIDEO_CAPTURE_BACKEND)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")

        try:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(round(src_fps / self.sample_fps)))

            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    yield idx, idx / src_fps, frame
                idx += 1
        finally:
            cap.release()

    def ocr_frame(self, frame_bgr: np.ndarray) -> list[tuple[list, str, float]]:
        # Hand the BGR uint8 ndarray straight to EasyOCR. On Apple Silicon's
        # unified memory, the only unavoidable copy is numpy→MPS heap, done once
        # inside the reader. Pre-wrapping in a torch tensor or converting to RGB
        # here would add a redundant pass over the pixels.
        with torch.inference_mode():
            return self.reader.readtext(frame_bgr)

    def process(
        self,
        video_path: Path,
        audio_path: Path | None = None,
    ) -> VideoResult:
        video_path = Path(video_path)
        audio_out = self.extract_audio(video_path, audio_path)

        hits: list[OCRHit] = []
        sampled = 0
        for idx, ts, frame in self.iter_frames(video_path):
            sampled += 1
            for bbox, text, conf in self.ocr_frame(frame):
                if conf < self.ocr_min_confidence:
                    continue
                hits.append(OCRHit(
                    frame_index=idx,
                    timestamp_s=ts,
                    text=text,
                    confidence=float(conf),
                    bbox=[(int(x), int(y)) for x, y in bbox],
                ))

        return VideoResult(
            video_path=video_path,
            audio_path=audio_out,
            device=self.device,
            frames_sampled=sampled,
            ocr=hits,
        )
