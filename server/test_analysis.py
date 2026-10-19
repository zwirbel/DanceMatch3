"""Small deterministic smoke checks for the DanceMatch signal pipeline.

Usage: python test_analysis.py --synthetic
       python test_analysis.py /path/to/a-short-recording.mp4
"""
import argparse
import json
import sys

import librosa
import numpy as np

from main import SAMPLE_RATE, analyze_signal


def click_track(bpm: float, beats_per_bar: int, seconds: int = 18) -> np.ndarray:
    sr = SAMPLE_RATE
    y = np.zeros(sr * seconds, dtype=np.float32)
    period = 60.0 / bpm
    tone_t = np.arange(int(sr * 0.045)) / sr
    tone = np.sin(2 * np.pi * 1000 * tone_t) * np.exp(-tone_t * 70)
    accents = [1.0] + [0.45] * (beats_per_bar - 1)
    for beat, t in enumerate(np.arange(0.1, seconds - 0.1, period)):
        start = int(t * sr)
        n = min(len(tone), len(y) - start)
        y[start:start+n] += tone[:n] * accents[beat % beats_per_bar]
    return y


def check_synthetic():
    outcomes = []
    for target, meter in ((104, 4), (90, 3), (126, 4), (174, 3)):
        result = analyze_signal(click_track(target, meter))
        bpm = result["bpm"]
        equivalent = min(abs(np.log2(bpm / target)), abs(np.log2((2 * bpm) / target)), abs(np.log2((bpm / 2) / target)))
        assert equivalent < 0.10, f"{target} BPM input returned {bpm} BPM"
        outcomes.append({"expected_bpm": target, "expected_meter": f"{meter}/4", **result})
    print(json.dumps(outcomes, ensure_ascii=False, indent=2))


def check_file(path: str):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=25)
    result = analyze_signal(y, sr)
    print(json.dumps({"file": path, "duration_seconds": round(len(y) / sr, 1), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        check_synthetic()
    elif args.path:
        check_file(args.path)
    else:
        parser.error("choose --synthetic or an audio file path")
