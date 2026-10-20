#!/usr/bin/env python3
"""Fetch a small, explicitly licensed starter set for DanceMatch3.

Run in an internet-enabled Python/Colab runtime:
    python training/prepare_supplemental_audio.py

This is not a complete training dataset. It only downloads the two sources in the manifest.
"""
import csv
import html
import re
import subprocess
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
    import requests

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "supplemental_audio_sources.csv"
OUT = ROOT / "supplemental_audio"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"

def direct_audio_candidates(page_text):
    page_text = html.unescape(page_text).replace("\\/", "/").replace("\\u0026", "&")
    patterns = [
        r'https?://files\.freemusicarchive\.org/[^"\'<>\\\s]+?\.(?:mp3|wav|ogg|flac)(?:\?[^"\'<>\\\s]*)?',
        r'https?://[^"\'<>\\\s]+?\.(?:mp3|wav|ogg|flac)(?:\?[^"\'<>\\\s]*)?',
    ]
    found = []
    for pattern in patterns:
        for url in re.findall(pattern, page_text, flags=re.IGNORECASE):
            url = url.rstrip("),;")
            if url not in found:
                found.append(url)
    return found

def download_one(row):
    dance = row["dance"].strip()
    target_dir = OUT / dance
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = (row["artist"] + " - " + row["title"]).replace("/", "-")
    page_url = row["source_url"].strip()
    page = requests.get(page_url, headers={"User-Agent": USER_AGENT}, timeout=30)
    page.raise_for_status()
    candidates = direct_audio_candidates(page.text)
    last_error = None
    for url in candidates:
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=60)
            response.raise_for_status()
            ctype = response.headers.get("Content-Type", "").lower()
            first = next(response.iter_content(64 * 1024), b"")
            if not first or "text/html" in ctype or b"<html" in first[:1000].lower():
                continue
            path = target_dir / (stem + ".mp3")
            with path.open("wb") as f:
                f.write(first)
                for chunk in response.iter_content(64 * 1024):
                    if chunk:
                        f.write(chunk)
            if path.stat().st_size > 20_000:
                print("Downloaded:", path)
                return path
            path.unlink(missing_ok=True)
        except Exception as exc:
            last_error = exc
    # Fallback in case the source page does not expose its audio URL in static HTML.
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yt-dlp"])
        template = str(target_dir / (stem + ".%(ext)s"))
        subprocess.run([
            sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-progress",
            "-x", "--audio-format", "mp3", "-o", template, page_url
        ], check=True)
        files = [p for p in target_dir.glob(stem + ".*")
                 if p.suffix.lower() in {".mp3", ".m4a", ".wav", ".ogg", ".opus"}]
        if files:
            print("Downloaded via yt-dlp:", files[0])
            return files[0]
    except Exception as exc:
        last_error = exc
    raise RuntimeError(
        "Could not download source page. Open it directly and verify license. Last error: "
        + str(last_error)
    )

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    metadata = []
    failures = []
    for row in rows:
        if row.get("download", "").strip().lower() not in {"yes", "true", "1"}:
            continue
        try:
            path = download_one(row)
            metadata.append({
                "file_name": str(path.relative_to(ROOT)),
                "dance": row["dance"],
                "title": row["title"],
                "artist": row["artist"],
                "source_url": row["source_url"],
                "license": row["license"],
            })
        except Exception as exc:
            failures.append(f'{row["dance"]} / {row["title"]}: {exc}')
            print("WARNING:", failures[-1])
    with (OUT / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["file_name", "dance", "title", "artist", "source_url", "license"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata)
    counts = {
        name: sum(1 for item in metadata if item["dance"].lower() == name.lower())
        for name in ("Bachata", "Discofox", "Salsa")
    }
    print("\nDownloaded counts:", counts)
    print("WARNING: one title per class is not enough for robust training; Discofox is still unsourced.")
    if failures:
        print("\nFailures:")
        for item in failures:
            print(" -", item)

if __name__ == "__main__":
    main()
