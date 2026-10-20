#!/usr/bin/env python3
"""Streaming track-level DanceMatch baseline; avoids caching the full dataset locally."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
from itertools import islice
from pathlib import Path

# Isolate Hugging Face files to a DanceMatch-specific cache so it can be cleaned
# safely without deleting unrelated caches. Set this BEFORE importing datasets/hub.
DEFAULT_CACHE_BASE = Path("/content") if Path("/content").exists() else Path(tempfile.gettempdir())
CACHE_ROOT = Path(os.environ.get("DANCEMATCH_CACHE_DIR", str(DEFAULT_CACHE_BASE / "dancematch_hf_cache")))
os.environ["HF_HOME"] = str(CACHE_ROOT)
os.environ["HF_HUB_CACHE"] = str(CACHE_ROOT / "hub")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_ROOT / "datasets")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import librosa
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

LABEL_KEYS = ("answer", "label", "dance", "genre", "class", "correct_choice")
AUDIO_KEYS = ("audio", "music", "sound", "clip")


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def disk_report(prefix: str) -> None:
    disk_root = Path("/content") if Path("/content").exists() else Path.cwd()
    usage = shutil.disk_usage(disk_root)
    print(f"{prefix}: {usage.free / (1024**3):.2f} GiB frei von {usage.total / (1024**3):.2f} GiB")


def cleanup_cache(keep_cache: bool) -> None:
    if keep_cache:
        print(f"Hugging Face-Cache bleibt für Wiederverwendung erhalten: {CACHE_ROOT}")
        return
    if CACHE_ROOT.exists():
        bytes_before = directory_size(CACHE_ROOT)
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)
        print(f"Projekt-Cache bereinigt (vorher ca. {bytes_before / (1024**2):.1f} MiB): {CACHE_ROOT}")
    disk_report("Speicher nach Cache-Bereinigung")


def pick_key(columns, candidates):
    lower = {str(c).lower(): c for c in columns}
    return next((lower[k] for k in candidates if k in lower), None)


def audio_features(audio_obj):
    # datasets v3 returns a dict; newer versions may return an AudioDecoder.
    if isinstance(audio_obj, dict):
        y = audio_obj.get("array")
        sr = audio_obj.get("sampling_rate", 22050)
    elif hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()
        y = getattr(samples, "data", None)
        sr = getattr(samples, "sample_rate", 22050)
        if hasattr(y, "detach"):
            y = y.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported audio value type: {type(audio_obj).__name__}")
    if y is None:
        raise ValueError("Audio sample is missing its array.")
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=0) if y.shape[0] < y.shape[-1] else np.mean(y, axis=1)
    if len(y) < sr * 3:
        raise ValueError("Audio clip shorter than 3 seconds.")
    # Keep only the first 18 seconds for analysis.
    y = y[: int(sr * 18)]
    y = librosa.resample(y, orig_sr=sr, target_sr=22050) if sr != 22050 else y
    sr = 22050
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=2048, hop_length=512)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    tempo = librosa.feature.tempo(onset_envelope=onset, sr=sr, hop_length=512)
    feats = []
    for matrix in (mfcc, chroma, contrast, centroid, rolloff, zcr):
        feats.extend(np.mean(matrix, axis=1).tolist())
        feats.extend(np.std(matrix, axis=1).tolist())
    feats.extend([float(np.median(tempo)), float(np.std(onset)), float(np.mean(onset))])
    return np.asarray(feats, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Yi3852/MuCUE")
    ap.add_argument("--split", default="ballroom_genres", help="Named split within the dataset")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-items", type=int, default=0, help="Optional small smoke test; 0 means all rows")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--keep-cache", action="store_true", help="Keep the isolated Hugging Face cache for faster reruns")
    args = ap.parse_args()

    # Remove leftovers from an interrupted previous run before checking free space.
    if not args.keep_cache and CACHE_ROOT.exists():
        old_bytes = directory_size(CACHE_ROOT)
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)
        print(f"Bereinige Cache-Reste des vorherigen Laufs: {old_bytes / (1024**2):.1f} MiB")
    disk_report("Vor dem Streaming-Download")
    disk_root = Path("/content") if Path("/content").exists() else Path.cwd()
    free_gib = shutil.disk_usage(disk_root).free / (1024**3)
    if free_gib < 0.75:
        raise SystemExit(
            f"Nur {free_gib:.2f} GiB freier Speicher. Aus Sicherheitsgründen startet das Training nicht. "
            "Bitte Runtime neu starten oder Speicher freigeben und erneut ausführen."
        )

    X, y, skipped = [], [], 0
    seen = 0
    try:
        try:
            # Streaming is the key safeguard: do not download/cache the full Arrow dataset.
            ds = load_dataset(args.dataset, name="default", split=args.split, streaming=True)
        except Exception as e:
            raise SystemExit(f"Could not stream dataset/split {args.dataset}/{args.split}: {e}") from e

        columns = getattr(ds, "column_names", None)
        if not columns:
            raise SystemExit("Dataset schema is unavailable in streaming mode.")
        audio_key = pick_key(columns, AUDIO_KEYS)
        label_key = pick_key(columns, LABEL_KEYS)
        if not audio_key or not label_key:
            raise SystemExit("Dataset schema not recognized. Columns: " + ", ".join(map(str, columns)))

        rows = ds if args.max_items == 0 else islice(ds, args.max_items)
        print(f"Streaming aktiv; verarbeite {'alle' if args.max_items == 0 else args.max_items} Zeilen maximal.")
        for i, row in enumerate(rows):
            seen += 1
            try:
                label = row.get(label_key)
                if not isinstance(label, str) or not label.strip():
                    skipped += 1
                    continue
                X.append(audio_features(row[audio_key]))
                y.append(label.strip())
            except Exception as e:
                skipped += 1
                print(f"Skipping row {i}: {e}")
            finally:
                # Drop the row/audio decoder before requesting the next sample.
                del row
                if i % 25 == 0:
                    gc.collect()
                    disk_report(f"Streaming-Fortschritt: {seen} Zeilen")
        del ds
        gc.collect()
    finally:
        # Runs on success, exceptions, SystemExit and keyboard interruption.
        cleanup_cache(args.keep_cache)

    if len(X) < 30 or len(set(y)) < 2:
        raise SystemExit(f"Not enough usable labeled tracks ({len(X)} clips, {len(set(y))} classes).")
    X = np.vstack(X)
    y = np.asarray(y)
    counts = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    # Hold out whole source rows (one feature vector per source clip).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=args.seed, n_jobs=-1)
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    report_out = {
        "dataset": args.dataset, "dataset_split": args.split,
        "usable_tracks": len(y), "rows_seen": seen, "skipped_rows": skipped, "class_counts": counts,
        "split": {"train": len(y_train), "test": len(y_test), "track_level": True},
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=sorted(set(y))).tolist(),
        "class_order": sorted(set(y)),
        "caveat": "Small-data feature baseline; not validated for 18-second phone recordings or the nine target classes."
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "dance_baseline.joblib")
    (out / "evaluation.json").write_text(json.dumps(report_out, indent=2), encoding="utf-8")
    print(json.dumps({
        "usable_tracks": len(y), "rows_seen": seen, "skipped_rows": skipped,
        "class_counts": counts, "macro_f1": report_out["macro_f1"], "artifacts": str(out)
    }, indent=2))


if __name__ == "__main__":
    main()
