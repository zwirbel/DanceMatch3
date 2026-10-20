#!/usr/bin/env python3
"""Track-level DanceMatch baseline. Experimental; not production inference."""
from __future__ import annotations
import argparse, json
from pathlib import Path
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

def pick_key(columns, candidates):
    lower = {str(c).lower(): c for c in columns}
    return next((lower[k] for k in candidates if k in lower), None)

def audio_features(audio_obj):
    if isinstance(audio_obj, dict):
        y = audio_obj.get("array")
        sr = audio_obj.get("sampling_rate", 22050)
    else:
        raise ValueError("Audio column is not decoded into an {array, sampling_rate} object.")
    if y is None:
        raise ValueError("Audio sample is missing its array.")
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=0) if y.shape[0] < y.shape[-1] else np.mean(y, axis=1)
    if len(y) < sr * 3:
        raise ValueError("Audio clip shorter than 3 seconds.")
    # Use a reproducible 18-second crop; resample for manageable CPU cost.
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
    ap.add_argument("--config", default="ballroom_genres")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-items", type=int, default=0, help="Optional small smoke test; 0 means all rows")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    try:
        ds = load_dataset(args.dataset, args.config)
    except Exception as e:
        raise SystemExit(f"Could not load dataset/config {args.dataset}/{args.config}: {e}")
    rows = []
    for split_name, split in ds.items():
        for row in split:
            rows.append(row)
    if args.max_items:
        rows = rows[:args.max_items]
    if not rows:
        raise SystemExit("Dataset returned no rows.")
    audio_key = pick_key(rows[0].keys(), AUDIO_KEYS)
    label_key = pick_key(rows[0].keys(), LABEL_KEYS)
    if not audio_key or not label_key:
        raise SystemExit("Dataset schema not recognized. Columns: " + ", ".join(rows[0].keys()))
    X, y, skipped = [], [], 0
    for i, row in enumerate(rows):
        label = row.get(label_key)
        if not isinstance(label, str) or not label.strip():
            skipped += 1
            continue
        try:
            X.append(audio_features(row[audio_key]))
            y.append(label.strip())
        except Exception as e:
            skipped += 1
            print(f"Skipping row {i}: {e}")
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
        "dataset": args.dataset, "config": args.config,
        "usable_tracks": len(y), "skipped_rows": skipped, "class_counts": counts,
        "split": {"train": len(y_train), "test": len(y_test), "track_level": True},
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=sorted(set(y))).tolist(),
        "class_order": sorted(set(y)),
        "caveat": "Small-data feature baseline; not validated for 18-second phone recordings or the nine target classes."
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "dance_baseline.joblib")
    (out / "evaluation.json").write_text(json.dumps(report_out, indent=2), encoding="utf-8")
    print(json.dumps({"usable_tracks": len(y), "class_counts": counts, "macro_f1": report_out["macro_f1"], "artifacts": str(out)}, indent=2))

if __name__ == "__main__":
    main()
