"""Run: python scripts/03_adapt.py --checkpoint checkpoints/general.pt

Fine-tune only AFTER training a compatible model on broader licensed music.
Fine-tuning a 7-song model on the exact same 7 songs is NOT transfer learning.
"""

# Allow `python scripts/XX_name.py` from the repository root without pip -e.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from musicgen.training import train

p = argparse.ArgumentParser(description="Fine-tune compatible general pretrained model")
p.add_argument("--checkpoint", required=True)
p.add_argument("--data", default="data/processed")
p.add_argument("--out", default="checkpoints/porter_adapted.pt")
p.add_argument("--steps", type=int, default=25)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-5)
a = p.parse_args()
train(a.data, a.out, steps=a.steps, batch_size=a.batch, lr=a.lr,
      checkpoint=a.checkpoint)
