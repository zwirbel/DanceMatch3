"""DanceMatch audio analysis API.

This is a signal-processing based recommender, not a trained music genre model.
It reports uncertainty and tempo alternatives because short excerpts often
produce half/double-time ambiguity.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

VERSION = "1.0.0"
MAX_BYTES = 25 * 1024 * 1024
MIN_SECONDS = 8
MAX_SECONDS = 25
SAMPLE_RATE = 22_050
HOP = 256

app = FastAPI(title="DanceMatch Audio API", version=VERSION)
origins = [x.strip() for x in os.getenv(
    "ALLOWED_ORIGINS", "https://snazzy-unicorn-0b9f44.netlify.app"
).split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/")
def root():
    return {"service": "DanceMatch Audio API", "version": VERSION, "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


def _autocorrelation_candidates(onset: np.ndarray, sr: int, hop: int) -> list[tuple[float, float]]:
    """Return normalized onset periodicity scores for plausible BPMs."""
    x = onset.astype(np.float64)
    x -= np.mean(x)
    norm = np.linalg.norm(x)
    if norm < 1e-8:
        return []
    x /= norm
    rate = sr / hop
    candidates: list[tuple[float, float]] = []
    for bpm in np.arange(55.0, 211.0, 0.5):
        lag = int(round(60.0 * rate / bpm))
        if lag < 1 or lag >= len(x) // 2:
            continue
        # Several beat multiples help when every second/third beat is accented.
        def corr(k: int) -> float:
            if k >= len(x) - 2:
                return 0.0
            left, right = x[k:], x[:-k]
            return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))
        score = 0.70 * corr(lag) + 0.20 * corr(2 * lag) + 0.10 * corr(max(1, round(lag / 2)))
        # Weak preference for the conventional social-dance tempo range; the
        # evidence remains dominant and both octave interpretations are returned.
        if bpm < 65 or bpm > 205:
            score *= 0.95
        candidates.append((float(bpm), score))
    return sorted(candidates, key=lambda x: x[1], reverse=True)


def _meter_estimate(strength: np.ndarray, beat_frames: np.ndarray) -> tuple[str, float, dict[str, float]]:
    """Compare beat-strength accents folded into 3- and 4-beat bars."""
    if len(beat_frames) < 12:
        return "Unklar", 0.0, {"3/4": 0.0, "4/4": 0.0}
    values = strength[np.clip(beat_frames, 0, len(strength) - 1)]
    values = np.asarray(values, dtype=float)
    if np.std(values) < 1e-5:
        return "Unklar", 0.0, {"3/4": 0.0, "4/4": 0.0}

    scores: dict[int, float] = {}
    for n in (3, 4):
        # Align all possible beat phases to reduce the effect of starting mid-bar.
        best = -1.0
        for phase in range(n):
            groups = [values[phase::n]]
            if not groups[0].size:
                continue
            means = np.array([np.mean(values[(phase + i) % n::n]) for i in range(n)])
            # A strong first beat with lower following beats signals triple meter
            # better than simple beat-to-beat loudness variation.
            first_bonus = max(0.0, means[0] - float(np.mean(means[1:])))
            contrast = (float(np.std(means)) + 0.45 * first_bonus) / (float(np.mean(means)) + 1e-8)
            best = max(best, contrast)
        scores[n] = max(0.0, best)

    total = scores[3] + scores[4] + 1e-8
    confidence = abs(scores[3] - scores[4]) / total
    # Only assert a likely meter with enough accent evidence and sufficient data.
    winner = 3 if scores[3] > scores[4] else 4
    if confidence >= 0.18 and max(scores.values()) >= 0.12:
        return f"{winner}/4 wahrscheinlich", float(min(0.85, confidence)), {"3/4": scores[3], "4/4": scores[4]}
    return "Unklar", float(min(0.4, confidence)), {"3/4": scores[3], "4/4": scores[4]}


def _recommend(bpm: float, meter: str, confidence: str) -> tuple[str, list[str], str]:
    triple = meter.startswith("3/4")
    if triple:
        if bpm >= 145:
            return "Wiener Walzer", ["Discofox (Halftime prüfen)", "Langsamer Walzer"], "Dreiertakt wahrscheinlich; höre auf den Walzerschritt 1–2–3."
        return "Langsamer Walzer", ["Rumba", "Discofox"], "Dreiertakt wahrscheinlich; der Walzerrhythmus sollte sich klar anfühlen."
    if bpm < 94:
        return "Rumba", ["Langsamer Discofox", "Langsamer Walzer (nur bei 3/4)"], "Ruhiges Tempo; falls jeder zweite Puls passt, kann es Discofox in Halftime sein."
    if bpm < 112:
        return "Discofox", ["Rumba", "Bachata (bei passendem Rhythmus)"], "Mittleres Tempo; die Betonung entscheidet zwischen Discofox und Rumba."
    if bpm < 128:
        return "Cha-Cha-Cha", ["Discofox", "Salsa (bei passendem Rhythmus)"], "Tempo passt zu Cha-Cha-Cha; den typischen Cha-Cha-Grundschritt gegenhören."
    if bpm < 150:
        return "Discofox", ["Cha-Cha-Cha", "Jive (Halftime prüfen)"], "Flottes Tempo; Discofox ist die robusteste Empfehlung für Popmusik."
    if bpm < 178:
        return "Jive", ["Discofox (Halftime prüfen)", "Cha-Cha-Cha"], "Schnelles Tempo; achte auf den federnden Jive-Rhythmus."
    return "Quickstep", ["Jive (Halftime prüfen)", "Wiener Walzer (nur bei 3/4)"], "Sehr schnelles Tempo; Quickstep nur wählen, wenn der 4/4-Galopp passt."


def analyze_signal(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Analyze decoded mono PCM, also exposed for deterministic fixture tests."""
    y = np.asarray(y, dtype=np.float32)
    rms = float(np.sqrt(np.mean(y * y))) if y.size else 0.0
    if rms < 0.002:
        raise ValueError("Die Aufnahme ist sehr leise. Bitte näher an der Musik aufnehmen.")
    y = librosa.util.normalize(y)
    if len(y) < int(sr * MIN_SECONDS):
        raise ValueError(f"Bitte mindestens {MIN_SECONDS} Sekunden Musik verwenden.")

    # Onset envelope emphasizes new attacks while downweighting sustained notes.
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, aggregate=np.median)
    onset = librosa.util.normalize(onset)
    candidates = _autocorrelation_candidates(onset, sr, HOP)
    if not candidates or candidates[0][1] < 0.025:
        raise ValueError("Kein stabiler Puls erkannt. Bitte einen klaren Musikausschnitt aufnehmen.")

    # Beat tracking contributes an independent estimate; do not let its common
    # octave errors overwrite strong periodicity evidence.
    tracked_tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset, sr=sr, hop_length=HOP, trim=False, units="frames"
    )
    tracked = float(np.asarray(tracked_tempo).reshape(-1)[0])
    # A slow detected pulse may actually be a measure/downbeat (e.g. one bar
    # every ~1 s in fast triple meter). Prefer a plausible dance pulse while
    # keeping the raw periodicity score and the beat tracker as separate clues.
    def interpretation_score(item: tuple[float, float]) -> float:
        value, periodicity_score = item
        dance_range_prior = -0.12 if value < 70 else (0.025 if 78 <= value <= 200 else 0.0)
        tracker_bonus = 0.0
        if 55 <= tracked <= 210:
            tracker_bonus = 0.07 * np.exp(-abs(np.log2(value / tracked)) * 2)
        return periodicity_score + dance_range_prior + tracker_bonus
    ranked = sorted(candidates, key=interpretation_score, reverse=True)
    bpm, periodicity = ranked[0]
    alternatives = []
    for alt in (bpm / 2, bpm * 2, bpm * 3 if bpm < 70 else bpm / 3 if bpm > 150 else -1):
        if 55 <= alt <= 210:
            alternatives.append(round(float(alt)))

    beat_frames = np.asarray(beat_frames, dtype=int)
    if len(beat_frames) < 8:
        beat_period = sr * 60.0 / (bpm * HOP)
        beat_frames = np.arange(0, len(onset), beat_period).astype(int)
    meter, meter_confidence, meter_scores = _meter_estimate(onset, beat_frames)

    # Stability across short windows helps avoid overconfident answers on a
    # tempo change, speech, or noisy recording.
    window_bpms = []
    for chunk in np.array_split(onset, 3):
        if len(chunk) < 12:
            continue
        local = _autocorrelation_candidates(chunk, sr, HOP)
        if local:
            window_bpms.append(local[0][0])
    stability = 0.0
    if len(window_bpms) >= 2:
        normalized = [min(abs(np.log2(x / bpm)), abs(np.log2((2*x) / bpm)), abs(np.log2((x/2) / bpm))) for x in window_bpms]
        stability = float(np.clip(1 - np.std(normalized) * 2.5, 0, 1))
    confidence_score = float(np.clip(0.55 * np.clip(periodicity * 4, 0, 1) + 0.45 * stability, 0, 1))
    confidence = "hoch" if confidence_score >= 0.70 else "mittel" if confidence_score >= 0.42 else "niedrig"
    dance, alternatives_dance, tip = _recommend(float(bpm), meter, confidence)

    return {
        "bpm": round(float(bpm), 1),
        "tempo_alternatives": alternatives,
        "meter": meter,
        "meter_confidence": round(meter_confidence, 2),
        "meter_scores": {k: round(float(v), 3) for k, v in meter_scores.items()},
        "dance": dance,
        "dance_alternatives": alternatives_dance,
        "confidence": confidence,
        "confidence_score": round(confidence_score, 2),
        "beat_consistency": round(stability, 2),
        "tip": tip,
        "note": "Tempo- und Akzentanalyse, kein trainierter Tanzstil-Klassifikator. Bei Unsicherheit Halb-/Doppelzeit und Taktgefühl gegenhören.",
    }


@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    raw = await audio.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Audiodatei zu gross (maximal 25 MB).")
    if not raw:
        raise HTTPException(400, "Die Audiodatei ist leer.")
    suffix = Path(audio.filename or "audio.webm").suffix.lower()
    if suffix not in {".webm", ".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".oga", ".aac", ".flac"}:
        suffix = ".webm"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(raw)
            temp_path = f.name
        y, sr = librosa.load(temp_path, sr=SAMPLE_RATE, mono=True, duration=MAX_SECONDS + 1)
        result = analyze_signal(y, sr)
        result["duration_seconds"] = round(len(y) / sr, 1)
        return result
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        # Do not return decoder internals or paths to callers.
        raise HTTPException(422, "Audio konnte nicht gelesen werden. Bitte WAV, MP3, M4A oder WebM verwenden.") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
