"""05 — Pretraining and fine-tuning use the SAME next-token objective.

For pretraining, use a large, licensed, diverse MIDI corpus.
For fine-tuning, load those weights and use a *smaller learning rate* on a
specialized dataset. Fine-tuning on 7 songs may overfit severely.
"""
from __future__ import annotations
import random
from itertools import cycle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import TokenWindows, get_records
from .model import ModelConfig, MusicTransformer, device_for_mac, load_model, save_model
from .tokens import PAD


def train(data_dir, output, steps=100, batch_size=8, context=128,
          lr=3e-4, checkpoint=None, seed=27, log_every=20):
    random.seed(seed)
    torch.manual_seed(seed)
    records = get_records(data_dir)
    random.shuffle(records)
    if len(records) > 1:
        validation_records = records[:1]
        training_records = records[1:]
    else:
        training_records, validation_records = records, []
        print("WARNING: Only one song; no held-out validation possible.")
    if checkpoint:
        model, ckpt = load_model(checkpoint)
        context = model.cfg.context
        print(f"Loaded pretrained weights from {checkpoint}")
    else:
        model = MusicTransformer(ModelConfig(context=context)).to(device_for_mac())
        print("Starting from RANDOM weights (this is NOT pretrained).")
    device = next(model.parameters()).device
    print(f"Device: {device} | training songs: {len(training_records)} | "
          f"validation songs: {len(validation_records)}")
    dataset = TokenWindows(training_records, context=context)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    valid_loader = None
    if validation_records:
        valid_loader = DataLoader(TokenWindows(validation_records, context=context),
                                  batch_size=batch_size, shuffle=False)
    iterator = iter(loader)
    for step in range(1, steps + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1), ignore_index=PAD)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            msg = f"step {step:>5}/{steps}: train_loss={loss.item():.3f}"
            if valid_loader:
                model.eval()
                losses = []
                with torch.no_grad():
                    for idx, (vx, vy) in enumerate(valid_loader):
                        if idx == 4:  # keep evaluation quick
                            break
                        vx, vy = vx.to(device), vy.to(device)
                        vl = model(vx)
                        losses.append(F.cross_entropy(vl.reshape(-1, vl.shape[-1]),
                            vy.reshape(-1), ignore_index=PAD).item())
                msg += f" val_loss={sum(losses)/len(losses):.3f}"
            print(msg, flush=True)
    save_model(model, output, trained_on=str(data_dir), steps=steps,
               finetuned_from=str(checkpoint) if checkpoint else None,
               training_song_count=len(training_records))
    print(f"Saved checkpoint to {output}")
