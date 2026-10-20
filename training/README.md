# DanceMatch 3.0 — first supervised audio-classifier experiment

This experiment is deliberately a **baseline**, not a claim of reliable nine-class recognition.

## Public research data discovered

- **MuCUE ballroom_genres** (Hugging Face; dataset card reports Apache-2.0): https://huggingface.co/datasets/Yi3852/MuCUE
  - Useful first labeled corpus for ballroom classes such as Cha-Cha-Cha, Jive, Quickstep, Rumba, Waltz and Viennese Waltz.
- **Salsa Dataset** (Zenodo): https://zenodo.org/records/13120822
  - 124 Salsa tracks with expert beat annotations. Audio fragments are available; full audio is available by request for fair non-profit research. This is useful for beat analysis, but it is not a balanced nine-style genre dataset.
- Ballroom dataset information: https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/ballroom.py

**Important data gap:** these sources do not by themselves provide a verified, balanced and clearly reusable set for all nine target styles. In particular, Discofox and Bachata need additional properly labeled audio with reuse rights. Do not scrape commercial songs or assume that a streamable track may be used for training.

## Goal

Train and evaluate a track-level baseline on labeled audio, then compare it with the existing tempo/accent heuristics. Split train/test by *track*, never by overlapping 18-second crops from the same track, to avoid leakage.

## Run the first experiment

Use Python 3.11. From this directory:

```bash
pip install -r requirements.txt
python train_baseline.py --dataset Yi3852/MuCUE --split ballroom_genres --max-items 0
```

The script uses Hugging Face streaming mode so it does not create a full local copy of the dataset. It extracts features one track at a time, keeps only compact feature vectors, isolates the temporary cache under `DANCEMATCH_CACHE_DIR`, and removes that cache in a `finally` block (including on errors). It makes a stratified track-level train/test split, reports macro-F1 and per-class recall, and writes a model plus a JSON report under `artifacts/`. If free disk space is below 0.75 GiB, it stops before downloading.

If the dataset's published schema changes, the script stops with a clear message rather than silently guessing labels. Add `--keep-cache` only if you deliberately want to retain the isolated cache for faster repeated runs; the Colab notebook does not enable this option.

## Nine-class target

`Discofox, Cha-Cha-Cha, Rumba, Jive, Slow Waltz, Viennese Waltz, Quickstep, Bachata, Salsa`

Do not label a model as nine-class until the training data includes all nine classes with sufficient independent tracks. For a meaningful first evaluation, aim for at least 100 distinct tracks per class, ideally more, and use independent music for final testing.
