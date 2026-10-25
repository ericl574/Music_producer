"""03 — Build (context tokens, next-token targets) training windows."""

from pathlib import Path
import torch
from torch.utils.data import Dataset
from .midi_data import load_record
from .tokens import PAD


class TokenWindows(Dataset):
    def __init__(self, records: list[dict], context: int = 128, stride: int = 64):
        self.context = context
        self.windows = []
        for record in records:
            ids = record["tokens"]
            for start in range(0, len(ids) - 1, stride):
                subseq = ids[start : start + context + 1]
                if len(subseq) < 2:
                    continue
                x = subseq[:-1] + [PAD] * max(0, context - (len(subseq) - 1))
                y = subseq[1:] + [PAD] * max(0, context - (len(subseq) - 1))
                self.windows.append((x[:context], y[:context]))
        if not self.windows:
            raise ValueError("No training windows; prepare some nonempty MIDI files first.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x, y = self.windows[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def get_records(folder: str | Path) -> list[dict]:
    paths = sorted(Path(folder).glob("*.json.gz"))
    if not paths:
        raise FileNotFoundError(f"No processed .json.gz records in {folder}. Run prepare.py first.")
    return [load_record(p) for p in paths]
