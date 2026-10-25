"""Run: python scripts/06_render.py --midi outputs/full_song.mid"""

# Allow `python scripts/XX_name.py` from the repository root without pip -e.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from musicgen.synth import render
import pretty_midi

p = argparse.ArgumentParser(description="Render arrangement into a simple electronic WAV")
p.add_argument("--midi", default="outputs/full_song.mid")
p.add_argument("--out", default="outputs/full_song.wav")
p.add_argument("--rate", type=int, default=22050)
a = p.parse_args()
info = render(pretty_midi.PrettyMIDI(a.midi), a.out, rate=a.rate)
print(f"Wrote {info['file']} | {info['seconds']:.1f}s | "
      f"{info['notes_rendered']} notes rendered")
