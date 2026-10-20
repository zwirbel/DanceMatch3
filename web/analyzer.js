(function (root) {
  'use strict';

  const FFT_SIZE = 1024;
  const HOP = 256;
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const angle = -2 * Math.PI / size;
      const wr0 = Math.cos(angle), wi0 = Math.sin(angle);
      for (let start = 0; start < n; start += size) {
        let wr = 1, wi = 0;
        const half = size >> 1;
        for (let j = 0; j < half; j++) {
          const even = start + j, odd = even + half;
          const tr = wr * re[odd] - wi * im[odd];
          const ti = wr * im[odd] + wi * re[odd];
          re[odd] = re[even] - tr; im[odd] = im[even] - ti;
          re[even] += tr; im[even] += ti;
          const nextWr = wr * wr0 - wi * wi0;
          wi = wr * wi0 + wi * wr0; wr = nextWr;
        }
      }
    }
  }

  function onsetEnvelope(input, sampleRate) {
    const stride = Math.max(1, Math.floor(sampleRate / 11025));
    const sr = sampleRate / stride;
    const count = Math.floor(input.length / stride);
    if (count < sr * 8) throw new Error('Bitte mindestens 8 Sekunden Musik verwenden.');
    const mono = new Float32Array(count);
    let rms = 0;
    for (let i = 0, j = 0; j < count; j++, i += stride) {
      let sample = 0;
      for (let c = 0; c < stride && i + c < input.length; c++) sample += input[i + c];
      sample /= stride;
      mono[j] = sample;
      rms += sample * sample;
    }
    rms = Math.sqrt(rms / count);
    if (rms < 0.002) throw new Error('Die Aufnahme ist sehr leise. Bitte näher an die Musik aufnehmen.');
    const highpass = new Float32Array(count);
    let previous = 0;
    for (let i = 0; i < count; i++) { highpass[i] = mono[i] - 0.96 * previous; previous = mono[i]; }
    const frames = Math.floor((count - FFT_SIZE) / HOP) + 1;
    const flux = new Float64Array(frames);
    const oldMagnitude = new Float64Array(FFT_SIZE / 2);
    const re = new Float64Array(FFT_SIZE), im = new Float64Array(FFT_SIZE);
    for (let frame = 0; frame < frames; frame++) {
      const offset = frame * HOP;
      for (let i = 0; i < FFT_SIZE; i++) {
        const window = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (FFT_SIZE - 1));
        re[i] = highpass[offset + i] * window; im[i] = 0;
      }
      fft(re, im);
      let sum = 0;
      for (let bin = 1; bin < FFT_SIZE / 2; bin++) {
        const magnitude = Math.hypot(re[bin], im[bin]);
        const rise = Math.max(0, magnitude - oldMagnitude[bin]);
        sum += rise;
        oldMagnitude[bin] = magnitude;
      }
      flux[frame] = sum;
    }
    let mean = 0;
    for (const x of flux) mean += x;
    mean /= flux.length;
    let norm = 0;
    for (let i = 0; i < flux.length; i++) { flux[i] = Math.max(0, flux[i] - mean * 0.15); norm += flux[i] * flux[i]; }
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < flux.length; i++) flux[i] /= norm;
    return { onset: flux, sr, frameRate: sr / HOP, duration: count / sr };
  }

  function correlation(x, lag) {
    if (lag < 1 || lag >= x.length - 2) return 0;
    const whole = Math.floor(lag), fraction = lag - whole;
    let dot = 0, aa = 0, bb = 0;
    for (let i = whole + 1; i < x.length; i++) {
      const a = x[i], b = x[i - whole] * (1 - fraction) + x[i - whole - 1] * fraction;
      dot += a * b; aa += a * a; bb += b * b;
    }
    return dot / (Math.sqrt(aa * bb) + 1e-12);
  }

  function candidates(onset, frameRate) {
    const x = Float64Array.from(onset);
    const mean = x.reduce((a, b) => a + b, 0) / x.length;
    let energy = 0;
    for (let i = 0; i < x.length; i++) { x[i] -= mean; energy += x[i] * x[i]; }
    if (energy < 1e-10) return [];
    const out = [];
    for (let bpm = 55; bpm <= 210; bpm += 0.5) {
      const lag = 60 * frameRate / bpm;
      if (lag < 1 || lag >= x.length / 2) continue;
      const score = 0.7 * correlation(x, lag) + 0.2 * correlation(x, 2 * lag) + 0.1 * correlation(x, Math.max(1, Math.round(lag / 2)));
      const prior = bpm < 70 ? -0.06 : bpm >= 195 && bpm <= 210 ? 0.015 : bpm >= 150 && bpm < 195 ? 0.01 : bpm >= 100 && bpm < 150 ? 0.015 : bpm >= 78 ? 0 : 0;
      out.push({ bpm, score: score + prior, periodicity: score });
    }
    return out.sort((a, b) => b.score - a.score);
  }

  function meterEstimate(onset, frameRate, bpm) {
    const period = frameRate * 60 / bpm;
    const beatCount = Math.floor(onset.length / period);
    if (beatCount < 12) return { meter: 'Unklar', confidence: 0, scores: { '3/4': 0, '4/4': 0 } };
    const radius = Math.max(1, Math.min(3, Math.round(period * 0.12)));
    const beats = Array.from({ length: beatCount }, (_, i) => {
      const center = Math.round(i * period);
      let peak = 0;
      for (let j = Math.max(0, center - radius); j <= Math.min(onset.length - 1, center + radius); j++) peak = Math.max(peak, onset[j]);
      return peak;
    });
    const mean = beats.reduce((a, b) => a + b, 0) / beats.length;
    const variance = beats.reduce((a, b) => a + (b - mean) ** 2, 0) / beats.length;
    if (variance < 1e-8) return { meter: 'Unklar', confidence: 0, scores: { '3/4': 0, '4/4': 0 } };
    const scores = {};
    for (const n of [3, 4]) {
      const centered = beats.map(x => x - mean);
      const corr = correlation(centered, n);
      const phaseMeans = Array.from({ length: n }, (_, p) => {
        const vals = beats.filter((_, i) => i % n === p);
        return vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
      });
      const accent = (Math.max(...phaseMeans) - Math.min(...phaseMeans)) / (mean + 1e-8);
      scores[n] = Math.max(0, corr) * 0.75 + Math.min(1, accent) * 0.25;
    }
    const total = scores[3] + scores[4] + 1e-8;
    const confidence = Math.abs(scores[3] - scores[4]) / total;
    const winner = scores[3] > scores[4] ? 3 : 4;
    if (confidence >= 0.18 && Math.max(scores[3], scores[4]) >= 0.12) {
      return { meter: winner + '/4 wahrscheinlich', confidence: Math.min(0.85, confidence), scores: { '3/4': scores[3], '4/4': scores[4] } };
    }
    return { meter: 'Unklar', confidence: Math.min(0.4, confidence), scores: { '3/4': scores[3], '4/4': scores[4] } };
  }

  function recommend(bpm, meter) {
    const triple = meter.startsWith('3/4');
    const bachata = bpm >= 95 && bpm <= 155
      ? ['Bachata (bei typischem 1–2–3–Tap-Rhythmus)'] : [];
    const salsa = bpm >= 105 && bpm <= 190
      ? ['Salsa (bei typischem Clave-/Latin-Rhythmus)'] : [];

    if (triple) return bpm >= 145
      ? ['Wiener Walzer', ['Langsamer Walzer', 'Discofox (Halftime prüfen)'], 'Dreiertakt wahrscheinlich; höre auf den Walzerschritt 1–2–3.']
      : ['Langsamer Walzer', ['Wiener Walzer (nur bei höherem Tempo)', 'Rumba'], 'Dreiertakt wahrscheinlich; höre auf den Walzerschritt 1–2–3.'];

    if (bpm < 88) return ['Rumba', ['Langsamer Discofox', 'Langsamer Walzer (nur bei 3/4)'], 'Ruhiges Tempo. Prüfe, ob der Grundpuls zur Rumba passt oder Discofox im halben Tempo natürlicher wirkt.'];
    if (bpm < 105) return ['Rumba', ['Discofox', ...bachata], 'Ruhiges bis mittleres 4/4-Tempo. Der Musikcharakter ist entscheidend; Tempo allein erkennt den Tanz nicht sicher.'];
    if (bpm < 118) return ['Discofox', ['Rumba', ...bachata, ...salsa], 'Mittleres Tempo: Discofox ist ein guter Startpunkt. Bei klarer Bachata- oder Salsa-Rhythmik kann ein Latin-Tanz besser passen.'];
    if (bpm < 132) return ['Cha-Cha-Cha', ['Discofox', ...bachata, ...salsa], 'Das Tempo passt zu Cha-Cha-Cha. Höre auf den typischen „cha-cha-cha“-Schritt; Salsa und Bachata brauchen zusätzlich ihren charakteristischen Rhythmus.'];
    if (bpm < 150) return ['Discofox', ['Cha-Cha-Cha', ...bachata, ...salsa, 'Jive (Halftime prüfen)'], 'Flottes Tempo. Vergleiche den geraden Discofox-Puls mit der Latin-Rhythmik von Salsa oder Bachata.'];
    if (bpm < 178) return ['Jive', ['Discofox (Halftime prüfen)', ...salsa, 'Quickstep (bei passendem 4/4-Galopp)'], 'Schnelles Tempo. Prüfe Halbzeit und Musikcharakter: Salsa kann schnell wirken, ohne dass es Jive ist.'];
    return ['Quickstep', ['Jive (Halftime prüfen)', ...salsa, 'Discofox (Halftime prüfen)'], 'Sehr schnelles Tempo. Quickstep passt nur, wenn der federnde 4/4-Galopp hörbar ist; bei Latin-Rhythmik auch Salsa gegenprüfen.'];
  }

  function analyzePcm(samples, sampleRate) {
    const { onset, frameRate, duration } = onsetEnvelope(samples, sampleRate);
    const ranked = candidates(onset, frameRate);
    if (!ranked.length || ranked[0].periodicity < 0.025) throw new Error('Kein stabiler Puls erkannt. Bitte einen klaren Musikausschnitt aufnehmen.');
    const bpm = ranked[0].bpm;
    const meter = meterEstimate(onset, frameRate, bpm);
    const alternatives = [bpm / 2, bpm * 2, bpm < 70 ? bpm * 3 : bpm > 150 ? bpm / 3 : -1]
      .filter(x => x >= 55 && x <= 210).map(Math.round);
    const dance = recommend(bpm, meter.meter);
    const tempoConfidenceScore = clamp(ranked[0].periodicity * 2.8, 0, 0.85);
    const overlappingDanceRange = !meter.meter.startsWith('3/4') && bpm >= 105 && bpm <= 155;
    const confidenceScore = tempoConfidenceScore * (meter.meter === 'Unklar' ? 0.55 : 1) * (overlappingDanceRange ? 0.7 : 1);
    return {
      bpm: Math.round(bpm * 10) / 10,
      tempo_alternatives: alternatives,
      meter: meter.meter,
      meter_confidence: Math.round(meter.confidence * 100) / 100,
      meter_scores: { '3/4': Math.round(meter.scores['3/4'] * 1000) / 1000, '4/4': Math.round(meter.scores['4/4'] * 1000) / 1000 },
      dance: dance[0],
      dance_alternatives: dance[1],
      confidence: confidenceScore >= 0.7 ? 'hoch' : confidenceScore >= 0.42 ? 'mittel' : 'niedrig',
      confidence_score: Math.round(confidenceScore * 100) / 100,
      beat_consistency: 0,
      tip: dance[2],
      note: 'Lokale Tempo- und Akzentanalyse, kein trainierter Tanzstil-Klassifikator. Bachata, Salsa, Cha-Cha-Cha und Discofox überschneiden sich im Tempo; höre zusätzlich auf den charakteristischen Rhythmus.',
      duration_seconds: Math.round(duration * 10) / 10
    };
  }

  async function analyzeBlob(blob) {
    const AudioContextClass = root.AudioContext || root.webkitAudioContext;
    if (!AudioContextClass) throw new Error('Dieser Browser unterstützt die lokale Audioanalyse nicht.');
    const context = new AudioContextClass();
    try {
      const decoded = await context.decodeAudioData(await blob.arrayBuffer());
      const length = Math.min(decoded.length, Math.floor(decoded.sampleRate * 25));
      const mono = new Float32Array(length);
      const channels = decoded.numberOfChannels;
      for (let c = 0; c < channels; c++) {
        const data = decoded.getChannelData(c);
        for (let i = 0; i < length; i++) mono[i] += data[i] / channels;
      }
      return analyzePcm(mono, decoded.sampleRate);
    } finally {
      await context.close().catch(() => {});
    }
  }

  const api = { analyzePcm, analyzeBlob };
  root.DanceMatchAnalyzer = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
