"""Run: python scripts/00_inspect.py --input-dir data/midi"""

# Allow `python scripts/XX_name.py` from the repository root without pip -e.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import pretty_midi
from musicgen.midi_data import classify_role, approximate_bpm

p = argparse.ArgumentParser(description="Inspect track metadata and note statistics")
p.add_argument("--input-dir", default="data/midi")
a = p.parse_args()
paths = sorted(Path(a.input_dir).glob("*.mid")) + sorted(Path(a.input_dir).glob("*.midi"))
if not paths:
    raise SystemExit(f"No .mid files found in {a.input_dir}")
print(f"Found {len(paths)} MIDI files\n")
for path in paths:
    midi = pretty_midi.PrettyMIDI(str(path))
    print(f"{path.name} | {midi.get_end_time():.1f}s | approx {approximate_bpm(midi):g} BPM")
    for inst in midi.instruments:
        if inst.notes:
            role = classify_role(inst.name, inst.is_drum)
            print(f"  {inst.name or '(unnamed)':18} => {role:7} "
                  f"| {len(inst.notes):4} notes | program {inst.program}")
