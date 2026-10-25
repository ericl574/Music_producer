"""Run: python scripts/02_train.py --data data/processed --steps 80"""

# Allow `python scripts/XX_name.py` from the repository root without pip -e.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from musicgen.training import train

p = argparse.ArgumentParser(description="Train Transformer from scratch (or pretrain on large corpus)")
p.add_argument("--data", default="data/processed")
p.add_argument("--out", default="checkpoints/toy_scratch.pt")
p.add_argument("--steps", type=int, default=80)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--context", type=int, default=128)
p.add_argument("--lr", type=float, default=3e-4)
a = p.parse_args()
train(a.data, a.out, steps=a.steps, batch_size=a.batch, context=a.context, lr=a.lr)
