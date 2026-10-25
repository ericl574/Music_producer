"""Small exact-n-gram check for potential memorization (NOT a legal guarantee).

Run: python scripts/07_check_copying.py --generated outputs/motif.mid
"""

# Allow `python scripts/XX_name.py` from the repository root without pip -e.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from musicgen.midi_data import encode_midi
from musicgen.tokens import ID_TO_TOKEN

p = argparse.ArgumentParser(description="Check longest exact pitch n-gram overlap")
p.add_argument("--generated", default="outputs/motif.mid")
p.add_argument("--references", default="data/midi")
a = p.parse_args()

def pitches(path):
    encoded = encode_midi(path)["tokens"]
    return [token for token in encoded if ID_TO_TOKEN[token].startswith("PITCH_")]

generated = pitches(a.generated)
paths = sorted(Path(a.references).glob("*.mid"))
if not paths:
    raise SystemExit("Reference files not found")
for path in paths:
    source = pitches(path)
    longest = 0
    # Exact subsequences of 4-12 pitched events; crude proxy only.
    for n in range(4, min(12, len(generated), len(source)) + 1):
        grams = {tuple(source[i:i+n]) for i in range(len(source)-n+1)}
        if any(tuple(generated[i:i+n]) in grams for i in range(len(generated)-n+1)):
            longest = n
    print(f"{path.name}: longest tested overlap = {longest} consecutive pitch tokens")
print("WARNING: This ignores timing, transposition, arrangement, and audio similarity.")
