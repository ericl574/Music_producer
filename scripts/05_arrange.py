"""Run: python scripts/05_arrange.py --motif outputs/motif.mid"""

# Allow `python scripts/XX_name.py` from the repository root without pip -e.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import pretty_midi
from musicgen.arrangement import arrange, SECTION_PLAN

p = argparse.ArgumentParser(description="Turn a short motif into a 128-bar arrangement")
p.add_argument("--motif", default="outputs/motif.mid")
p.add_argument("--out", default="outputs/full_song.mid")
p.add_argument("--bpm", type=int, default=128)
p.add_argument("--tonic", type=int, default=57, help="MIDI pitch tonic, default A3")
a = p.parse_args()
motif = pretty_midi.PrettyMIDI(a.motif)
song = arrange(motif, bpm=a.bpm, tonic=a.tonic)
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
song.write(str(out))
print(f"Wrote {out} | {sum(b for _, b in SECTION_PLAN)} bars, "
      f"{song.get_end_time():.1f}s of MIDI events")
