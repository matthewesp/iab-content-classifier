"""Tests for the detect-gated, region-aware OCR cascade.

See docs/superpowers/specs/2026-06-20-detect-gated-ocr-design.md.

Standalone-runnable (no pytest dependency required):
    .venv/bin/python tests/test_ocr_cascade.py
Also discoverable by pytest if it's installed.

These are integration tests: they load EasyOCR and run on real frames from
data/test_vids/, so they're slow (~model load) but prove the real behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VID = ROOT / "data/test_vids/tiktok_search_Amusement_and_Theme_Parks_7585362462262758686.mp4"


def _first_frame_with_text(vp):
    """Return the first sampled frame whose readtext() finds any text."""
    for _, _, frame in vp.iter_frames(VID):
        if vp.reader.readtext(frame):
            return frame
    raise RuntimeError(f"no text found in any sampled frame of {VID}")


def _norm(results):
    """(bbox, text, conf) tuples → comparable (text, rounded-conf) list."""
    return [(t, round(float(c), 4)) for _, t, c in results]


def test_detect_then_recognize_matches_readtext():
    """Splitting readtext() into detect_text()+recognize_text() must yield the
    identical text+confidence results — this is what makes the detect-gate
    lossless (no boxes from detect() ⇒ readtext() finds nothing either)."""
    from video_processor import VideoProcessor

    vp = VideoProcessor(sample_fps=1.0)
    frame = _first_frame_with_text(vp)

    ref = vp.reader.readtext(frame)                      # ground truth
    horizontal, free = vp.detect_text(frame)
    got = vp.recognize_text(frame, horizontal, free)

    assert _norm(got) == _norm(ref), f"\n ref={_norm(ref)}\n got={_norm(got)}"


def test_detect_gate_skips_recognition_on_textless_frame():
    """Feature 1: a frame with no detectable text must yield no hits AND must
    not call the recognizer (the expensive half)."""
    import numpy as np
    from video_processor import VideoProcessor

    vp = VideoProcessor(sample_fps=1.0)
    blank = np.zeros((640, 360, 3), dtype=np.uint8)

    horizontal, free = vp.detect_text(blank)
    assert not horizontal and not free, "expected no boxes on a blank frame"

    calls = {"n": 0}
    orig = vp.recognize_text
    vp.recognize_text = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1]
    try:
        assert vp.ocr_frame(blank) == []
    finally:
        vp.recognize_text = orig
    assert calls["n"] == 0, "recognizer was called on a textless frame"


def test_region_dedup_skips_unchanged_region_but_not_changed(monkeypatch=None):
    """Feature 2 gate logic, isolated from EasyOCR with controlled frames:

      f0: text region R, background A
      f1: SAME region R, DIFFERENT background  → passes cheap gate, region matches → SKIP
      f2: DIFFERENT region, different background → recognized (must NOT over-skip)

    Proves region-dedup skips an unchanged caption (even when the background
    moved) and still recognizes a genuinely different caption.
    """
    import numpy as np
    from pipeline import VideoClassifier

    BOX = [10, 50, 20, 40]                      # x0, x1, y0, y1
    region_R = np.tile(np.linspace(0, 255, 40, dtype=np.uint8), (20, 1))  # fixed pattern
    region_X = np.full((20, 40), 90, dtype=np.uint8)                      # different content

    def _frame(region, bg_bottom):
        f = np.zeros((80, 80, 3), dtype=np.uint8)
        f[20:40, 10:50] = region[:, :, None]    # text region
        f[50:80, :] = bg_bottom                 # background differs → beats cheap gate
        return f

    f0 = _frame(region_R, 0)
    f1 = _frame(region_R, 200)                  # same region, different bg
    f2 = _frame(region_X, 120)                  # different region

    clf = VideoClassifier(warn_untrained=False, cache_root=None)
    clf.caption_only = False                    # skip geometry filter for synthetic boxes
    clf.ocr_region_dedup_threshold = 8.0        # enable Feature 2

    seq = [(0, 0.0, f0), (1, 1.0, f1), (2, 2.0, f2)]
    texts = {0: "HELLO", 1: "HELLO", 2: "WORLD"}
    quad = [[10, 20], [50, 20], [50, 40], [10, 40]]

    clf.vp.iter_frames = lambda _p: iter(seq)
    clf.vp.detect_text = lambda _f: ([BOX], [])
    # recognize_text is keyed by which frame object it's handed.
    def _rec(frame, *_a):
        for i, _, fr in seq:
            if fr is frame:
                return [(quad, texts[i], 0.99)]
        return []
    clf.vp.recognize_text = _rec

    t: dict = {}
    _, hits = clf._scan_video(VID, timings=t)

    assert t["detect_calls"] == 3
    assert t["region_skipped"] == 1, t            # f1 skipped
    assert t["recognize_calls"] == 2, t           # f0 + f2 only
    assert {h.text for h in hits} == {"HELLO", "WORLD"}


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
