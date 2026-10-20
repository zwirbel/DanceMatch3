# DanceMatch 3.0

Mobile-first web app for recording 18 seconds of music, estimating tempo and likely meter locally in the browser, and suggesting suitable social dances. The optional Python API is included for hosted deployments. Dance suggestions are transparent tempo/accent heuristics; this is not a trained genre classifier.

## Project layout

- `web/` — static mobile web app, local audio analyzer, and PWA assets for Netlify
- `server/` — FastAPI + librosa analysis service for Render
- `render.yaml` — Render Blueprint for the API

## Local development

Install FFmpeg and Python 3.11 or 3.12. In `server/`:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Set `ALLOWED_ORIGINS=http://localhost:8888,https://snazzy-unicorn-0b9f44.netlify.app` for local and deployed frontend testing. Serve `web/` over localhost (for example `python -m http.server 8888 -d ../web`). Set `window.DANCEMATCH_API_URL` in `web/config.js` to `http://localhost:8000` while testing.

## Zero-cost mode: analyze locally in the browser

Leave `window.DANCEMATCH_API_URL` empty in `web/config.js`. The browser decodes the recording or selected audio file and performs onset, tempo, and likely-meter analysis on the device. Audio is not uploaded and no API host, card, or extra account is needed. The user can then publish the contents of `web/` to the existing Netlify site.

## Optional Python API on Render

1. Push the contents of this folder to a Git repository you control.
2. In Render, create a **Blueprint** from that repository and select `render.yaml`.
3. Keep the service on the Free plan to avoid a paid service. Initial build downloads the Docker image and Python dependencies and may take several minutes. Render describes Free instances as suitable for testing and hobby projects, not production use.
4. Wait until `/health` returns `{"status":"ok","version":"1.0.0"}`. Copy the HTTPS service URL.
5. In `web/config.js`, set `window.DANCEMATCH_API_URL` to that URL, without a trailing slash. The browser analyzer remains the default when this value is empty.

Free services sleep after 15 minutes without traffic; waking can take about a minute. The ephemeral filesystem is fine for this app because audio is temporary and no user data is stored. Free services have usage limits, so check the Render dashboard; do not select a paid plan unless intended. See [Render's Free service limitations](https://render.com/docs/free).

## Deploy the frontend to the existing Netlify site

Publish the contents of `web/` as the site's static publish directory. If using a Git-connected site, set the publish directory to `web` and leave build command empty. Ensure `config.js` contains the Render HTTPS URL, then publish. The app will register its service worker and can be installed from a mobile browser's Add to Home Screen menu.

## Analysis API

- `GET /health` — health check
- `POST /analyze` — multipart form field `audio`; accepts up to 25 MB, decodes the first 25 seconds, requires at least 8 seconds
- `ALLOWED_ORIGINS` — comma-separated frontend origins allowed by CORS

Audio is written to a temporary file only while FFmpeg/librosa decode and analyze it; the file is removed in a `finally` block. No audio is persisted by the application. Hosting providers may apply their own platform-level operational logging and retention.

## Validation

From the repository root, run `node web/analyzer.test.js` to check eight tempo/meter fixtures covering Rumba, Discofox, Cha-Cha-Cha, Jive, Quickstep, and both waltzes, plus the minimum-duration guard.

From `server/`, run `python test_analysis.py /path/to/test-audio.mp4`. This checks decoding and JSON response via the local API. `python test_analysis.py --synthetic` generates known 4/4 and 3/4 click tracks and checks BPM estimates without requiring any private test recording.

## Known limits

- A few seconds of music may not contain enough information to decide meter reliably; the API reports `Unklar` instead of claiming certainty.
- Half/double-time ambiguity is common in pop and dance music. The API returns alternatives and confidence.
- The app recommends Rumba, Discofox, Cha-Cha-Cha, Jive, Quickstep, and the two waltzes from tempo and likely meter. Bachata and Salsa are included as conditional alternatives in overlapping tempo ranges, with listening cues for their characteristic rhythms. These are still tempo-based recommendations, not reliable genre recognition; Bachata, Salsa, Cha-Cha-Cha, and Discofox overlap substantially.
- There is no user account, rate limiting, abuse monitoring, or production SLA in this low-cost prototype.
