"""04 — A GPT-like causal Transformer for next-MIDI-token prediction."""

from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from .tokens import VOCAB


@dataclass
class ModelConfig:
    vocab_size: int = len(VOCAB)
    context: int = 128
    width: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1


class MusicTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.position = nn.Embedding(cfg.context, cfg.width)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.width, nhead=cfg.heads, dim_feedforward=cfg.width * 4,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=cfg.layers)
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, cfg.vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2:
            raise ValueError("Input should have shape (batch, time).")
        batch, time = ids.shape
        if time > self.cfg.context:
            raise ValueError(f"Input length {time} exceeds context {self.cfg.context}.")
        positions = torch.arange(time, device=ids.device)
        h = self.embedding(ids) + self.position(positions)[None, :, :]
        mask = torch.triu(torch.ones(time, time, dtype=torch.bool, device=ids.device), 1)
        hidden = self.layers(h, mask=mask)
        return self.head(self.norm(hidden))


def device_for_mac() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_model(model: MusicTransformer, path, **meta):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": asdict(model.cfg),
                "vocab": VOCAB, "meta": meta}, path)


def load_model(path, device=None):
    from .tokens import VOCAB
    device = device or device_for_mac()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["vocab"] != VOCAB:
        raise ValueError("Checkpoint uses another vocabulary. Check tokenization!")
    model = MusicTransformer(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ckpt
