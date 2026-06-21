# Detect-Gated, Region-Aware OCR

**Date:** 2026-06-20
**Status:** Approved (design)

## Problem

Feature extraction is OCR-bound. Per video (`data/profile_postmlx.json`, MPS,
cache off, ~45s wall): OCR ≈ 30s (~70%), all three GPU backbones ≈ 6s (~13%).
The existing whole-frame visual dedup already culls 66 sampled frames → ~18
`readtext()` calls, but each surviving call runs the full EasyOCR pipeline
(CRAFT detection + per-crop CRNN recognition) even when:

1. the frame contains **no text** at all, or
2. the frame's text is the **same caption** as a frame already recognized
   (e.g. static TikTok caption over changing background — whole-frame absdiff
   sees the background change and re-OCRs unnecessarily).

## Goal

Skip the expensive recognition step in those two cases, reducing `ocr_s`
without losing captured text.

## Key Insight

EasyOCR 1.7.2 exposes `detect()` and `recognize()` separately; `readtext()` is
just both. Because `readtext()` runs the *same* detector internally, **gating
recognition on `detect()` finding boxes is lossless** — if `detect()` returns no
boxes, `readtext()` would have produced no text either. Only the region-dedup
(case 2) carries a small, threshold-tunable accuracy risk.

## Design

### Architecture

Preserve the current separation of concerns:
- `VideoProcessor` remains the EasyOCR wrapper.
- `pipeline._scan_video` remains the per-frame orchestrator and owns the cascade.

### Components

- `VideoProcessor.detect_text(frame) -> (horizontal_list, free_list)` — wraps
  `reader.detect()`.
- `VideoProcessor.recognize_text(frame, horizontal_list, free_list) -> list[(bbox, text, conf)]`
  — wraps `reader.recognize()`; returns the same tuple shape `readtext()` yields
  today, so the downstream confidence + caption-geometry filters are unchanged.
- `VideoProcessor.ocr_frame()` stays as a thin `detect`+`recognize` composition
  for any non-cascade caller (back-compat).

### Cascade (per sampled frame, in `_scan_video`)

1. **Cheap gate (existing):** 64×64 whole-frame grayscale absdiff vs the last
   *recognized* frame's thumb. Below `ocr_dedup_threshold` → skip.
2. **Detect gate (new, lossless):** `detect_text()`. No boxes → record zero
   hits for the frame, skip recognition.
3. **Region gate (new):** crop the union bounding box of all detected boxes,
   downsample to a small grayscale tile, absdiff vs the last recognized frame's
   region tile. Below `ocr_region_dedup_threshold` → **emit nothing**, skip
   recognition.
4. **Recognize:** `recognize_text()` on the detected boxes → existing
   `ocr_min_confidence` + `_is_caption_like` filters → `OCRHit`s. Update the
   last-recognized thumb and last region tile.

Region comparison is against the **last recognized frame only** (consistent with
the existing whole-frame gate). A caption that disappears and reappears
identically after different content will be re-recognized; acceptable for v1.

### Duplicate output behavior

When the region gate fires, the frame emits **no** hits. The caption's text was
already captured when it first appeared; skipping avoids repeating the same
caption many times in the concatenated OCR text blob (which also saves
DistilBERT token budget). Net accuracy-neutral or better.

### Configuration

New fields, all tunable and added to `ocr_params` so the per-video OCR cache
auto-invalidates when behavior changes:

- detect-gate: always on (lossless), built into `ocr_frame()` and the scan
  cascade.
- `ocr_region_dedup_threshold: float = 0.0` — **default OFF** (see Calibration
  Finding). When > 0, mean grayscale absdiff on the region tile (same 0–255
  scale as `ocr_dedup_threshold`); enable (e.g. `8.0`) only for static-text /
  static-background content.

## Calibration Finding (as built)

Measured the region-tile absdiff against whether the recognized *text* actually
changed, on `data/test_vids/`. The pixel signature does **not** separate "same
caption" from "different caption" on caption-over-video content: same-text
frames diffed 21–28 (raw) / 40–63 (Otsu-binarized) while some different-text
frames diffed as low as 14 / 48. Cause: TikTok captions overlay moving video, so
the box crop is dominated by background motion, not glyphs. Any threshold
aggressive enough to skip true duplicates also merges distinct captions — and the
distinct captions (e.g. different national-park names) are exactly the
category-relevant text.

**Decision:** Feature 1 (detect-gate) ships on by default — it is lossless and
saves recognition on textless frames (e.g. a no-caption clip skipped 100% of
recognition in testing). Feature 2 (region-dedup) ships as an opt-in knob
defaulted OFF, suitable only for static-text content; its gate logic is covered
by a deterministic test rather than threshold-tuned on noisy real video.

Detector defaults (`text_threshold`, `low_text`, `min_size`, etc.) use EasyOCR's
defaults unless a need to expose them arises.

### Profiling counters

Add to the `timings` dict emitted by `_scan_video`: `detect_calls`,
`recognize_calls`, `region_skipped` (alongside existing `ocr_s`, `ocr_n_calls`,
`dedup_skipped`).

## Cost Analysis

On frames that pass the cheap gate, cost is strictly ≤ today:
- textless frames: detect only (was detect+recognize) → win.
- unchanged-caption frames: detect only → win.
- new-caption frames: detect+recognize → same as today.

`detect()` runs only on cheap-gate survivors (~18/66), not all frames, so total
detection count does not increase versus today.

## Testing (TDD)

Implemented in `tests/test_ocr_cascade.py` (standalone-runnable; no pytest dep):

1. **Equivalence (proves detect-gate lossless):** `detect_text()` + `recognize_text()`
   produce identical text+confidence to `reader.readtext()` on a real frame.
2. **Detect-gate:** a textless (blank) frame yields no hits and does **not** call
   the recognizer.
3. **Region-dedup gate logic:** deterministic test with controlled frames — a
   frame whose text region is unchanged (but background moved) is skipped, while
   a frame with a genuinely different region is still recognized; no text is
   invented.

## Out of Scope

- Per-box (vs union) region signatures — possible future refinement.
- LRU of recent region tiles to dedup reappearing captions across gaps.
- Batched recognition across frames/videos.
