# IAB Content Classifier

Multimodal video classifier targeting the [IAB Content Taxonomy v3.1](https://iabtechlab.com/standards/content-taxonomy/) — 37 Tier-1 categories, ~700 nodes total across 4 tiers. Pulls signal from spoken audio (Whisper-tiny ASR), on-screen text (EasyOCR), and visual content (DINO ViT-S/16) per video; fuses to a 1536-dim feature; runs a lazy hierarchical cascade of linear-probe routers to land on a leaf category.

Runs on Apple Silicon (mlx-whisper, MPS, frame-dedup OCR). The same code runs on CUDA (Colab) — fp16 autocast on backbones kicks in automatically; mlx-whisper is replaced by HF Whisper.

## How it works

Two phases. Feature extraction is heavy; training is a linear probe on cached features and is essentially free.

```
videos/*.mp4
    │
    │  scripts/build_features.py
    │  per video: ffmpeg → cv2 frames → EasyOCR (with frame dedup)
    │             → mlx-whisper ASR → 3 frozen backbones → torch.cat
    │  per-stage cache: each stage skips if its inputs/params haven't changed
    ▼
data/cache/<video_hash>/{ocr,asr,text_emb,audio_emb,video_emb}.*
    │
    ▼
data/features.pt        (N, 1536) fp32 + leaf_ids + metadata
    │
    │  scripts/train_classifier.py
    │  one router per parent in the taxonomy, each trained independently
    ▼
models/__root__.pt      (root: 37 Tier-1 classes)
models/<parent_id>.pt   (per-parent: predicts that parent's direct children)
models/training_summary.json
    │
    │  pipeline.VideoClassifier.classify(video)
    │  cascade: root → child → ... while top-class confidence < threshold
    ▼
{ "final_id": "...", "name_path": [...], "steps": [...] }
```

The split exists because backbones are frozen — features only need to be extracted once per labeling iteration, then router training is sub-second.

## Files

### Source

| file | what it does |
|---|---|
| `pipeline.py` | `VideoClassifier` — end-to-end orchestration. Owns the backbone, ASR engine, taxonomy, cascade, and cache. Two entry points: `extract_features(video)` (no classifier — used for training) and `classify(video)` (extract + cascade). |
| `multimodal_backbone.py` | `MultimodalBackbone` — three frozen encoders (DistilBERT, Whisper-tiny encoder, DINO ViT-S/16). All forwards run under `torch.inference_mode`; on CUDA they additionally use `torch.autocast(fp16)`. Outputs cast back to fp32 so downstream `cat` + linear probes stay in fp32. Also defines `VideoTransformer` — a smaller factorized space-time ViT used as an alternative to DINO when memory is tight. |
| `video_processor.py` | `VideoProcessor` — `ffmpeg` audio extraction, `cv2` frame iteration (`AVFOUNDATION` on macOS for VideoToolbox HW decode, `FFMPEG` everywhere else), per-frame EasyOCR. `get_device()` helper: MPS → CUDA → CPU priority. |
| `cascading_classifier.py` | `Router` (2-layer MLP probe over the 1536-dim feature) and `CascadingClassifier`. Routers are lazy-loaded — only the root is materialized at init; child routers come off disk on first escalation. `confidence_threshold` and `max_active_routers` cap inference cost. |
| `taxonomy.py` | Parses `content_taxonomy_3.1.tsv` into a `Taxonomy` dataclass: `nodes` dict, `coarse_ids` (Tier 1), `children` (parent→ordered children), `path_to(leaf_id)` (root→leaf id path). Index order is fixed by CSV row order — don't reorder after training a checkpoint. |
| `cache.py` | `StageCache` — per-video, per-stage on-disk artifact cache. Layout: `<root>/<video_key>/<stage>_<params_key>.<ext>`. `video_key` = sha256(abspath + mtime + size); `params_key` = sha256(json of stage params). Different param sets coexist (good for A/B tests); video edits invalidate automatically. |

### Scripts

| script | what it does |
|---|---|
| `scripts/build_features.py` | Reads `--labels CSV`, runs `extract_features` per video, stacks results into `--out features.pt`. Cache-aware via `--cache-root` (default `data/cache`). Re-runs after partial completion are cheap because finished videos hit warm cache. |
| `scripts/train_classifier.py` | Reads `features.pt`, trains the root router + one router per non-leaf parent. AdamW + CE, `--val-frac 0.2` per router, `--min-samples 2` to skip empty parents. Each router is independent and writes to `models/<id>.pt`. |
| `scripts/profile_pipeline.py` | Per-stage wall-clock profiler. Calls `extract_features(profile=True)` so each stage's timer is bracketed by `torch.mps.synchronize()` — values reflect actual GPU completion. Use to measure before/after any optimization. `--use-cache` off by default (profiling needs cold runs). |

### Data + models

| path | contents |
|---|---|
| `data/videos/` | Source mp4s. |
| `data/labels.csv` | `video_path,leaf_id` per row. `video_path` resolved relative to CWD; `leaf_id` must exist in the taxonomy. |
| `data/labels.example.csv`, `data/labels.smoke.csv` | Reference label files (older datasets). |
| `data/cache/<video_hash>/` | Per-stage cache. `ocr_<h>.json` (hits + frames_sampled), `asr_<h>.txt`, `text_emb_<h>.pt`, `audio_emb_<h>.pt`, `video_emb_<h>.pt`. Delete the directory to force a cold rebuild. |
| `data/features.pt` | `{features: (N, 1536) fp32, leaf_ids, video_paths, feature_dim, taxonomy_path, skipped}`. Output of `build_features.py`. |
| `data/profile_*.json` | Profiler baselines. |
| `models/__root__.pt` | Root router weights — 37-way classifier over IAB Tier-1 categories. |
| `models/<parent_id>.pt` | Per-parent router weights. Loaded lazily at inference when the cascade escalates into that parent. |
| `models/training_summary.json` | Per-router stats (n samples, train/val loss/acc, elapsed, status). |

### Other

| path | what it is |
|---|---|
| `content_taxonomy_3.1.tsv` | IAB v3.1 (705 rows). Cols: `Unique ID`, `Parent`, `Name`, `Tier 1..4`. Single source of truth — the parser handles the multi-spelling header weirdness in IAB exports. |
| `pyproject.toml` | Deps: `torch`, `transformers`, `easyocr`, `opencv-python`, `soundfile`, `mlx-whisper` (the last is gated `sys_platform == 'darwin' and platform_machine == 'arm64'` — Linux/CUDA installs skip it silently). |
| `colab/setup.ipynb` | End-to-end Colab recipe: Drive mount, `HF_HOME` export, `apt install ffmpeg`, repo clone, `pip install -e .`, sanity check, `build_features` + `train_classifier` against Drive paths. |

## Setup

```bash
brew install ffmpeg     # if not already on PATH
uv sync                  # installs everything else
```

`mlx-whisper` only installs on Apple Silicon; the pipeline auto-falls-back to HF Whisper elsewhere.

## Usage

### 1. Label your videos

`data/labels.csv`:

```csv
video_path,leaf_id
data/videos/foo.mp4,151
data/videos/bar.mp4,179
```

`leaf_id` is the IAB taxonomy node id. Look it up in `content_taxonomy_3.1.tsv` (col 0 = `Unique ID`, col 2 = `Name`).

### 2. Extract features (the slow part)

```bash
uv run python scripts/build_features.py \
    --labels data/labels.csv \
    --out    data/features.pt
```

Per video on M1: **~40s cold, ~0.02s warm**. The cache makes interrupted runs cheap — rerun and finished videos skip everything.

### 3. Train

```bash
uv run python scripts/train_classifier.py \
    --features data/features.pt \
    --out-dir  models/
```

Sub-second per router on M1 — features are already in memory; routers are tiny MLPs.

### 4. Classify

```bash
uv run python pipeline.py data/videos/somevideo.mp4
```

Or from Python:

```python
from pipeline import VideoClassifier
clf = VideoClassifier()
result = clf.classify("data/videos/somevideo.mp4")
print(result["classification"]["name_path"])
```

## Configuration knobs

`VideoClassifier(...)` constructor params worth knowing:

| param | default | effect |
|---|---|---|
| `sample_fps` | `1.0` | Frames sampled per second of video for OCR + DINO. Lowering halves OCR + video-encode cost. |
| `ocr_dedup_threshold` | `8.0` | Mean per-pixel absdiff (uint8 grayscale, 64×64 thumb) below which a frame is treated as a duplicate of the last OCR'd frame and EasyOCR is skipped. `0` disables dedup. |
| `caption_min_pt` | `18.0` | Text size floor in points; converted to a fraction of frame height via `(pt × 4/3) / caption_reference_height_px`. Smaller → catches more text + more noise. |
| `caption_reference_height_px` | `1080` | Frame height the pt mapping assumes. Override per source resolution. |
| `caption_min_width_frac` | `0.05` | Min bbox width as fraction of frame width. Raise to keep only wide captions. |
| `caption_min_y` / `caption_max_y` | `0.20` / `0.70` | Vertical band where captions are expected (TikTok-style upper-middle). Open to `0.0`/`1.0` for subtitles or full-frame OCR. |
| `caption_min_chars` | `3` | Drop OCR hits shorter than this — most are detector noise. |
| `confidence_threshold` | `0.8` | Cascade stops when a router's top-class confidence ≥ this; otherwise escalates to that class's child router. |
| `max_active_routers` | `16` | LRU cap on materialized child routers. Memory ceiling for inference. |
| `cache_root` | `Path("data/cache")` | Per-stage cache directory. `None` disables. |

## Performance (M1 MacBook, current state)

Per-video feature extraction, cold cache, ~60s landscape video at `sample_fps=1.0`:

| stage | seconds |
|---|---|
| OCR (EasyOCR on MPS, with dedup) | ~30 |
| Frame iter (cv2 + AVFOUNDATION) | ~4 |
| DINO ViT-S/16 (per-frame) | ~3 |
| Whisper-tiny ASR (mlx-whisper) | ~2 |
| DistilBERT + Whisper-encoder + ffmpeg | <1 |
| **wall-clock total** | **~40s** |

Warm cache: **~0.02s/video** (4000×, bit-identical features).

Per-router training: sub-second on M1 (AdamW + CE on cached fp32 features, no DataLoader overhead).

How it got here, vs the original baseline of 168.7s/video:

1. Frame deduplication before OCR (4.7× on OCR).
2. Per-stage on-disk cache (warm = free).
3. mlx-whisper replacing HF Whisper for ASR (8.3× on transcription).

See `data/profile_baseline.json`, `data/profile_postdedup.json`, `data/profile_postmlx.json` for the numbers.

## Colab

Open `colab/setup.ipynb` in Colab via File → Open → GitHub. Edit `REPO_URL` and `IAB_DRIVE` in cell 1. Run cells in order.

T4 GPU is roughly 3-5× faster than M1 for this workload — CUDA EasyOCR is well-optimized and HF Whisper on CUDA matches mlx-whisper on M1. The cache layer makes Colab session timeouts (12h max, 90min idle) cheap to recover from: re-run the build cell and finished videos skip everything.

## Current dataset

40 labeled videos across 4 leaf categories, all under parent `150 Attractions`:

- `151` Amusement and Theme Parks (10)
- `179` Bars & Restaurants (10)
- `181` Casinos & Gambling (10)
- `153` Historic Site and Landmark Tours (10)

Trains the root router trivially (only one Tier-1 class represented) and the Tier-2 Attractions router meaningfully (4-way classification on 32 train / 8 val with `val_frac=0.2`).
