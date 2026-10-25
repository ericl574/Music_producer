"""02 — Read MIDI, retain instrument roles, quantize notes, encode events.

Quantization uses a 16th-note grid at four beats per bar. Tempo changes are
mapped to beats before quantizing; final generated MIDI has one chosen tempo.
This is educational code, not a lossless representation of every MIDI feature.
"""

from __future__ import annotations
import gzip
import json
import re
from pathlib import Path
import numpy as np
import pretty_midi
from .tokens import BOS, EOS, BAR, ROLES, token_id


def classify_role(name: str, is_drum: bool = False) -> str:
    if is_drum:
        return "drums"
    s = name.lower()
    for words, role in (
        (("bass", "sub"), "bass"),
        (("chord", "piano", "keys", "guitar", "string"), "chords"),
        (("pad", "choir", "atmos", "ambient"), "pad"),
        (("arp", "pluck", "kalimba", "harp"), "arp"),
        (("lead", "vocal", "voice", "melody", "chip"), "lead"),
        (("drum", "kick", "snare", "hat"), "drums"),
    ):
        if any(w in s for w in words):
            return role
    return "other"


def seconds_to_beats(midi: pretty_midi.PrettyMIDI, seconds: np.ndarray) -> np.ndarray:
    """Convert seconds to musical beats even if tempo changes within a song."""
    times, tempos = midi.get_tempo_changes()
    times = np.asarray(times, dtype=float)
    tempos = np.asarray(tempos, dtype=float)
    if not len(times):
        return seconds * (120 / 60)
    offset = np.zeros(len(times))
    if len(times) > 1:
        offset[1:] = np.cumsum(np.diff(times) * tempos[:-1] / 60)
    indices = np.clip(np.searchsorted(times, seconds, side="right") - 1, 0, len(times) - 1)
    return offset[indices] + (seconds - times[indices]) * tempos[indices] / 60


def approximate_bpm(midi: pretty_midi.PrettyMIDI) -> float:
    _, tempi = midi.get_tempo_changes()
    return round(float(np.median(tempi)), 2) if len(tempi) else 120.0


def encode_midi(path: str | Path, max_bars: int | None = None) -> dict:
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = []
    for inst in midi.instruments:
        role = classify_role(inst.name, inst.is_drum)
        if not inst.notes:
            continue
        starts = seconds_to_beats(midi, np.array([n.start for n in inst.notes]))
        ends = seconds_to_beats(midi, np.array([n.end for n in inst.notes]))
        for note, start, end in zip(inst.notes, starts, ends):
            abs_step = max(0, int(round(float(start) * 4)))
            bar, step = divmod(abs_step, 16)
            if max_bars is not None and bar >= max_bars:
                continue
            dur = max(1, min(32, int(round(float(end - start) * 4))))
            vel = max(1, min(8, (note.velocity + 15) // 16))
            notes.append((bar, step, role, int(note.pitch), dur, vel))
    notes.sort(key=lambda n: (n[0], n[1], ROLES.index(n[2]), n[3]))
    tokens = [BOS]
    if notes:
        next_index = 0
        last_bar = notes[-1][0]
        for bar in range(last_bar + 1):
            tokens.append(BAR)
            while next_index < len(notes) and notes[next_index][0] == bar:
                _, step, role, pitch, duration, velocity = notes[next_index]
                tokens.extend([
                    token_id(f"STEP_{step}"), token_id(f"ROLE_{role}"),
                    token_id(f"PITCH_{pitch}"), token_id(f"DUR_{duration}"),
                    token_id(f"VEL_{velocity}"),
                ])
                next_index += 1
    tokens.append(EOS)
    return {
        "name": Path(path).stem,
        "bpm": approximate_bpm(midi),
        "n_notes": len(notes),
        "n_bars": notes[-1][0] + 1 if notes else 0,
        "tokens": tokens,
    }


def save_record(record: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(record, f)


def load_record(path: str | Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def decode_midi(tokens: list[int], bpm: float = 128.0) -> pretty_midi.PrettyMIDI:
    """Parse complete BAR -> (STEP ROLE PITCH DUR VEL)* sequences into MIDI.

    Incomplete trailing events are dropped safely; this never invents new notes.
    """
    from .tokens import ID_TO_TOKEN
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    programs = {"lead": 80, "chords": 0, "pad": 88, "bass": 38,
                "arp": 10, "drums": 0, "other": 0}
    instruments = {
        role: pretty_midi.Instrument(program=programs[role], is_drum=role == "drums", name=role)
        for role in ROLES
    }
    bar = -1
    i = 0
    beat_seconds = 60 / bpm
    while i < len(tokens):
        event = ID_TO_TOKEN.get(int(tokens[i]), "")
        if event == "BAR":
            bar += 1
            i += 1
            continue
        if event == "EOS":
            break
        if not event.startswith("STEP_") or bar < 0 or i + 4 >= len(tokens):
            i += 1
            continue
        fields = [ID_TO_TOKEN.get(int(t), "") for t in tokens[i : i + 5]]
        if not (fields[1].startswith("ROLE_") and fields[2].startswith("PITCH_")
                and fields[3].startswith("DUR_") and fields[4].startswith("VEL_")):
            i += 1
            continue
        try:
            step = int(fields[0][5:])
            role = fields[1][5:]
            pitch = int(fields[2][6:])
            duration = int(fields[3][4:])
            velocity = int(fields[4][4:])
            if not (0 <= step < 16 and role in instruments and 0 <= pitch < 128
                    and 1 <= duration <= 32 and 1 <= velocity <= 8):
                raise ValueError("Invalid event")
            start = (bar * 16 + step) / 4 * beat_seconds
            end = start + duration / 4 * beat_seconds
            instruments[role].notes.append(pretty_midi.Note(
                velocity=min(127, velocity * 16), pitch=pitch, start=start, end=end
            ))
            i += 5
        except ValueError:
            i += 1
    midi.instruments.extend(inst for inst in instruments.values() if inst.notes)
    return midi


def clean_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
