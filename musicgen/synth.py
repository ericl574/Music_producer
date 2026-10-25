"""08 — Dependency-light synthesizer: MIDI -> actual WAV audio.

Sound is deliberately BASIC. A professional arrangement needs better samples,
virtual instruments, automation, transitions, mixing, and mastering.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import pretty_midi


def sound_wave(pitch: int, duration: float, rate: int, name: str, rng):
    count = max(1, int(duration * rate))
    t = np.arange(count, dtype=np.float32) / rate
    f = float(pretty_midi.note_number_to_hz(pitch))
    lower = name.lower()
    if "drum" in lower:
        if pitch == 36:  # kick
            phase = 2 * np.pi * (f * t + 40 * (1 - np.exp(-t * 30)) / 30)
            wave = np.sin(phase) * np.exp(-t * 25)
        elif pitch == 38:  # snare
            wave = rng.standard_normal(count).astype(np.float32) * np.exp(-t * 25)
        else:  # hi-hat
            noise = rng.standard_normal(count).astype(np.float32)
            wave = np.concatenate([np.zeros(1, dtype=np.float32), np.diff(noise)]) * np.exp(-t * 65)
    elif "bass" in lower:
        wave = (np.sin(2 * np.pi * f * t) + .30 * np.sin(4 * np.pi * f * t))
        wave *= np.minimum(1, t / .015) * np.minimum(1, (duration - t) / .06)
    elif "pad" in lower:
        wave = (np.sin(2 * np.pi * f * t) + .22 * np.sin(2 * np.pi * 2 * f * t))
        wave *= np.minimum(1, t / .30) * np.minimum(1, (duration - t) / .35)
    elif "arp" in lower:
        wave = np.sin(2 * np.pi * f * t) + .36 * np.sin(2 * np.pi * 2.01 * f * t)
        wave *= np.minimum(1, t / .005) * np.exp(-t * 8)
    elif "chords" in lower:
        wave = np.sin(2 * np.pi * f * t) + .18 * np.sin(4 * np.pi * f * t)
        wave *= np.minimum(1, t / .02) * np.minimum(1, (duration - t) / .12)
    else:  # lead / other
        wave = (np.sin(2 * np.pi * f * t)
                + .32 * np.sin(4 * np.pi * f * t)
                + .15 * np.sin(6 * np.pi * f * t))
        wave *= np.minimum(1, t / .01) * np.minimum(1, (duration - t) / .1)
    return np.asarray(wave, dtype=np.float32)


def render(midi: pretty_midi.PrettyMIDI, output: str | Path,
           rate: int = 22050, max_seconds: int = 360):
    duration = min(midi.get_end_time() + 2, max_seconds)
    count = int(duration * rate)
    left = np.zeros(count, dtype=np.float32)
    right = np.zeros(count, dtype=np.float32)
    rng = np.random.default_rng(27)
    total = 0
    for track_idx, instrument in enumerate(midi.instruments):
        pan = ((track_idx * 3) % 7 - 3) / 8
        lgain = np.sqrt((1 - pan) / 2)
        rgain = np.sqrt((1 + pan) / 2)
        for note in instrument.notes:
            start = int(note.start * rate)
            if start >= count:
                continue
            dur = min(6, max(.02, note.end - note.start))
            wave = sound_wave(note.pitch, dur, rate, instrument.name, rng)
            end = min(start + len(wave), count)
            if end <= start:
                continue
            gain = .10 * (note.velocity / 127)
            left[start:end] += wave[:end-start] * (gain * lgain)
            right[start:end] += wave[:end-start] * (gain * rgain)
            total += 1
    # Very lightweight echo (illustrates effects, not professional reverb).
    delay = int(.22 * rate)
    if delay < count:
        left[delay:] += .14 * left[:-delay].copy()
        right[delay:] += .14 * right[:-delay].copy()
    audio = np.column_stack([left, right])
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = .90 * audio / peak
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output), rate, (audio * 32767).astype(np.int16))
    return {"seconds": duration, "notes_rendered": total, "file": str(output)}
