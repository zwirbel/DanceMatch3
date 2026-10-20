# Supplemental dance audio sources

This is a small, license-conscious starter manifest, not a completed training dataset.

Sources identified as CC BY 4.0 on their Free Music Archive pages:
- Bachata — Elijah_K: https://freemusicarchive.org/music/elijah-k/single/bachata/
- Salsa — Alex Morgan: https://freemusicarchive.org/index.php/music/alex-morgan/bossa-nova-latin-jazz-vol-2/latin-salsa-caliente-dance-floor-cut/

Only these two starter sources are listed for automatic download. This is not enough data to train robust Bachata or Salsa classes, and Discofox remains unsourced. Do not treat multiple snippets of one track as independent training titles.

Run `python training/prepare_supplemental_audio.py` in an internet-enabled Python/Colab runtime. The script attempts to extract a direct audio URL from each source page and falls back to yt-dlp if needed. Verify the source-page license and attribution before using any downloaded file in a product.
