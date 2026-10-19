const assert = require('node:assert/strict');
const { analyzePcm } = require('./analyzer.js');

function clickTrack(bpm, beatsPerBar, seconds = 20, sampleRate = 11025) {
  const samples = new Float32Array(Math.floor(seconds * sampleRate));
  const interval = sampleRate * 60 / bpm;
  for (let beat = 0, at = 0; at + sampleRate * 0.018 < samples.length; beat++, at += interval) {
    const accent = beat % beatsPerBar === 0 ? 1 : 0.42;
    const length = Math.floor(sampleRate * 0.018);
    for (let i = 0; i < length; i++) {
      const fade = 1 - i / length;
      samples[Math.floor(at) + i] += accent * fade * Math.sin(2 * Math.PI * 1200 * i / sampleRate);
    }
  }
  return samples;
}

for (const fixture of [
  { bpm: 100, meter: '4/4', bar: 4, dance: 'Rumba' },
  { bpm: 104, meter: '4/4', bar: 4, dance: 'Rumba' },
  { bpm: 90, meter: '3/4', bar: 3, dance: 'Langsamer Walzer' },
  { bpm: 126, meter: '4/4', bar: 4, dance: 'Cha-Cha-Cha' },
  { bpm: 140, meter: '4/4', bar: 4, dance: 'Discofox' },
  { bpm: 174, meter: '4/4', bar: 4, dance: 'Jive' },
  { bpm: 174, meter: '3/4', bar: 3, dance: 'Wiener Walzer' },
  { bpm: 205, meter: '4/4', bar: 4, dance: 'Quickstep' }
]) {
  const result = analyzePcm(clickTrack(fixture.bpm, fixture.bar), 11025);
  assert.ok(Math.abs(result.bpm - fixture.bpm) <= 1.5, `${fixture.bpm} BPM became ${result.bpm}`);
  assert.ok(result.meter.startsWith(fixture.meter), `${fixture.meter} became ${result.meter}`);
  assert.equal(result.dance, fixture.dance);
  if (fixture.bpm === 126) assert.ok(result.dance_alternatives.some(name => name.startsWith('Bachata')), 'Bachata should be shown as a possible dance in its overlapping tempo range');
  console.log(`${fixture.bpm} BPM ${fixture.meter}: ${result.bpm}, ${result.meter}, ${result.dance}`);
}

assert.throws(() => analyzePcm(new Float32Array(11025 * 4), 11025), /mindestens 8 Sekunden/);
console.log('Short-clip validation passed.');
