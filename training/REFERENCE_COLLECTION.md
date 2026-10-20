# DanceMatch reference track catalogue

This is a **curated candidate list**, not yet a ground-truth training dataset. It contains 45 song/version candidates across the nine target dance styles. Each row links to a public guide or playlist source that supports the candidate association.

## Important rules before training
- A title alone is not a reliable label. Specific arrangements, remixes, live versions, tempo edits and covers can change dance suitability.
- Verify each exact recording with at least two experienced dancers or a trusted dance-music source. Record reviewer, version/ISRC, duration and confirmation status.
- Do not download or train on Spotify streams. Spotify account access does not grant audio-training rights. Use appropriately licensed files, datasets with compatible licenses, or recordings made with permission.
- For each confirmed track, extract several non-overlapping 18-second windows. Keep all windows from one source track in the same train/validation/test split.
- Include hard negatives and ambiguous tracks (e.g. pop songs that could plausibly fit Discofox or Cha-Cha) and allow a multi-label or "uncertain" outcome where appropriate.
- Evaluate macro-F1, per-class recall, confusion matrix, calibration, and phone-recording robustness. Do not report confidence as a probability until calibrated on held-out tracks.

## Public research sources
- MuCUE dataset card: https://huggingface.co/datasets/Yi3852/MuCUE
- Ballroom dataset loader/metadata: https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/ballroom.py
- Salsa beat dataset: https://zenodo.org/records/13120822
- Ballroom song and tempo guides: https://www.ballroomdancesongs.com/guides/ballroom-practice-playlist-guide
- Jive examples: https://www.ballroomdancesongs.com/dances/jive
- Rumba examples: https://www.ballroomdancesongs.com/dances/rumba-international
- Quickstep examples: https://www.ballroomdancesongs.com/guides/best-quickstep-songs
- Discofox song discovery: https://www.dance-charts.de/2026070217463/die-100-besten-discofox-hits-aller-zeiten
- Bachata examples: https://dynamicbachata.com/blog/best-bachata-songs-beginners/
- Salsa examples: https://danceseekers.com/blog/music-to-know-salsa-bachata-kizomba

## Status
- [x] Candidate seed list created
- [ ] Confirm exact recordings and labels
- [ ] Obtain reusable audio assets for all nine classes
- [ ] Train baseline and report held-out results
- [ ] Integrate only after evaluation
