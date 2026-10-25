"""07 — A hierarchical arrangement planner: motif -> full 128-bar song.

A Transformer predicts LOCAL musical notes. This deterministic planner gives
GLOBAL structure: recurring theme, instrumentation, dynamics, drum patterns.
A later research version could learn this planner from labeled section data.
"""
from __future__ import annotations
import random
import pretty_midi

SECTION_PLAN = (
    ("intro", 16), ("buildup", 16), ("drop_1", 32),
    ("bridge", 16), ("drop_2", 32), ("outro", 16),
)


def add_note(inst, pitch, start, duration, velocity=90):
    if duration > 0:
        inst.notes.append(pretty_midi.Note(velocity=max(1, min(127, int(velocity))),
                            pitch=max(0, min(127, int(pitch))),
                            start=max(0, start), end=max(0, start) + duration))


def arrange(motif: pretty_midi.PrettyMIDI, bpm=128, tonic=57, seed=27):
    """Build 128 bars in 4/4. tonic=57 (A3), default harmony is A minor.

    Note: harmony is TEMPLATE-BASED, not learned from seven songs.
    Model-generated melody may still clash. This is our first experimental baseline.
    """
    rng = random.Random(seed)
    whole = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    programs = {"lead": 80, "chords": 0, "pad": 88, "bass": 38, "arp": 10}
    tracks = {k: pretty_midi.Instrument(program=v, name=k) for k, v in programs.items()}
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    beat = 60 / bpm
    bar_seconds = 4 * beat
    phrase_seconds = 4 * bar_seconds
    sources = [inst for inst in motif.instruments if inst.notes and not inst.is_drum]
    if not sources:
        raise ValueError("Motif has no playable notes")
    # Prefer recognized lead; otherwise use the first source track.
    lead_source = next((i for i in sources if "lead" in i.name.lower()), sources[0])
    arp_source = next((i for i in sources if "arp" in i.name.lower()), None)
    # I-vi? No: in natural minor, this template is i-VI-III-VII.
    roots = [tonic, tonic - 4, tonic + 3, tonic - 2]
    chord_types = [(0, 3, 7), (0, 4, 7), (0, 4, 7), (0, 4, 7)]
    cursor = 0
    for section, bars in SECTION_PLAN:
        start_bar = cursor
        is_drop = section.startswith("drop")
        for j in range(bars):
            absolute_bar = start_bar + j
            section_fraction = j / max(1, bars - 1)
            root = roots[absolute_bar % 4]
            chord = chord_types[absolute_bar % 4]
            t = absolute_bar * bar_seconds
            # Pads play quietly in calm sections, more intensely during drops.
            if section in ("intro", "bridge", "outro", "buildup", "drop_1", "drop_2"):
                pad_vel = 30 if section in ("intro", "outro") else (42 if section == "bridge" else 55)
                for interval in chord:
                    add_note(tracks["pad"], root + interval, t, bar_seconds * .97, pad_vel)
            if is_drop or section == "buildup":
                for interval in chord:
                    add_note(tracks["chords"], root + 12 + interval, t, beat * 1.85,
                             63 if section == "buildup" else 83)
                    if is_drop:
                        add_note(tracks["chords"], root + 12 + interval,
                                 t + beat * 2, beat * 1.8, 73)
            if is_drop or section == "buildup":
                for offset in (0, 2):
                    add_note(tracks["bass"], root - 24, t + offset * beat,
                             beat * 1.65, 85 if is_drop else 55)
            # Copy the SAME 4-bar phrase throughout the song: coherence by design.
            phrase_bar = absolute_bar % 4
            for n in lead_source.notes:
                source_bar = int(n.start // bar_seconds)
                if source_bar % 4 != phrase_bar:
                    continue
                local_time = n.start % bar_seconds
                # Intro/outro thinner; 2nd drop layers melody an octave occasionally.
                if section in ("intro", "outro") and (j % 4 < 2 or rng.random() < .30):
                    continue
                if section == "bridge" and j % 2 == 1:
                    continue
                vel_scale = {"intro": .45, "buildup": .60, "drop_1": .92,
                             "bridge": .55, "drop_2": 1.0, "outro": .40}[section]
                pitch = n.pitch + (12 if section == "drop_2" and j % 8 >= 4 else 0)
                add_note(tracks["lead"], pitch, t + local_time,
                         min(n.end - n.start, bar_seconds - local_time),
                         n.velocity * vel_scale)
            if arp_source and (is_drop or section == "buildup"):
                for n in arp_source.notes:
                    if int(n.start // bar_seconds) % 4 == phrase_bar:
                        local = n.start % bar_seconds
                        add_note(tracks["arp"], n.pitch, t + local,
                                 min(n.end - n.start, bar_seconds - local),
                                 n.velocity * (.75 if is_drop else .45))
            # Procedural electronic drums: kick/snare/hats. Not sourced from originals.
            if is_drop or section == "buildup":
                for beat_idx in range(4):
                    add_note(drums, 36, t + beat_idx * beat, .12, 105 if is_drop else 75)
                    if beat_idx in (1, 3):
                        add_note(drums, 38, t + beat_idx * beat, .15, 96 if is_drop else 65)
                    for eighth in range(2):
                        if not is_drop and section_fraction < .40 and eighth == 1:
                            continue
                        add_note(drums, 42, t + (beat_idx + eighth / 2) * beat,
                                 .07, 47 if is_drop else 32)
            elif section == "bridge" and j % 2 == 0:
                add_note(drums, 42, t, .06, 18)
        cursor += bars
    whole.instruments.extend([track for track in tracks.values() if track.notes])
    if drums.notes:
        whole.instruments.append(drums)
    return whole
