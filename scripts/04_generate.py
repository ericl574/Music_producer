"""Run --demo without any trained model; run --checkpoint for ML output."""

# Allow `python scripts/XX_name.py` from the repository root without pip -e.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from musicgen.generation import demo_motif, generate_motif
from musicgen.model import load_model
from musicgen.midi_data import load_record
from musicgen.tokens import BAR, EOS

p = argparse.ArgumentParser(description="Generate a 4-bar symbolic-music motif")
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--demo", action="store_true", help="A deterministic musical engineering baseline")
g.add_argument("--checkpoint", help="Compatible trained .pt model")
p.add_argument("--bars", type=int, default=4)
p.add_argument("--notes-per-bar", type=int, default=7)
p.add_argument("--bpm", type=int, default=128)
p.add_argument("--temp", type=float, default=.85)
p.add_argument("--primer-record", help="Optional processed MIDI record used as context; may repeat input")
p.add_argument("--primer-bars", type=int, default=1)
p.add_argument("--out", default="outputs/motif.mid")
a = p.parse_args()
if a.demo:
    motif = demo_motif(a.bpm)
    tokens = None
else:
    model, _ = load_model(a.checkpoint)
    primer = None
    if a.primer_record:
        source = load_record(a.primer_record)["tokens"]
        seen = 0
        primer = []
        for token in source:
            if token == EOS:
                break
            if token == BAR:
                seen += 1
                if seen > a.primer_bars:
                    break
            primer.append(token)
        print(f"Conditioning on {seen} bars of original input. Evaluate copying risk.")
    tokens, motif = generate_motif(model, bars=a.bars,
                                   notes_per_bar=a.notes_per_bar, bpm=a.bpm,
                                   temperature=a.temp, primer=primer)
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
motif.write(str(out))
if tokens is not None:
    out.with_suffix(".tokens.json").write_text(json.dumps(tokens, indent=2))
print(f"Wrote {out} | {sum(len(i.notes) for i in motif.instruments)} notes")
