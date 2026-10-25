"""Run: python scripts/01_prepare.py --input-dir data/midi --out data/processed"""

# Allow `python scripts/XX_name.py` from the repository root without pip -e.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from musicgen.midi_data import encode_midi, save_record, clean_name

p = argparse.ArgumentParser(description="Quantize MIDI to fixed-vocabulary event sequences")
p.add_argument("--input-dir", default="data/midi")
p.add_argument("--out", default="data/processed")
p.add_argument("--max-bars", type=int, default=None)
a = p.parse_args()
paths = sorted(Path(a.input_dir).glob("*.mid")) + sorted(Path(a.input_dir).glob("*.midi"))
if not paths:
    raise SystemExit("No MIDI files. See README: copying your local transcriptions.")
for path in paths:
    try:
        record = encode_midi(path, max_bars=a.max_bars)
        out = Path(a.out) / f"{clean_name(path.stem)}.json.gz"
        save_record(record, out)
        print(f"{path.name}: {record['n_notes']} notes, {record['n_bars']} bars, "
              f"{len(record['tokens'])} tokens -> {out}")
    except Exception as error:
        print(f"ERROR: {path.name}: {error}")
