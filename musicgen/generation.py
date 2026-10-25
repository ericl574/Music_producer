"""06 — Sample new tokens from a trained Transformer.

Constrained sampling ensures valid event grammar. It does NOT guarantee
musically good notes. Use a bar/phrase planner for long-range structure.
"""
from __future__ import annotations
import torch
import pretty_midi
from .tokens import BOS, BAR, TOKEN_TO_ID, ROLES, token_ids
from .midi_data import decode_midi


@torch.no_grad()
def pick(model, prefix: list[int], allowed: list[int], temperature: float, top_k: int):
    device = next(model.parameters()).device
    context = prefix[-model.cfg.context:]
    x = torch.tensor([context], dtype=torch.long, device=device)
    logits = model(x)[0, -1, allowed].float() / max(temperature, 0.05)
    if top_k > 0 and len(allowed) > top_k:
        values, index = torch.topk(logits, min(top_k, len(allowed)))
        ids = [allowed[i] for i in index.tolist()]
        probs = torch.softmax(values, dim=-1)
    else:
        ids = allowed
        probs = torch.softmax(logits, dim=-1)
    chosen = torch.multinomial(probs, 1).item()
    return int(ids[chosen])


def generate_motif(model, bars=4, notes_per_bar=7, bpm=128,
                   temperature=0.85, top_k=12, root_pitch_class=9,
                   primer: list[int] | None = None):
    """Generate a fixed number of bars, optionally after a source primer.

    root_pitch_class=9 means A. Notes are constrained to natural minor for the
    first working prototype, a creative constraint rather than a learned key.
    """
    model.eval()
    tokens = list(primer) if primer else [BOS]
    if not tokens or tokens[0] != BOS:
        raise ValueError("Primer must start with BOS")
    scale = {(root_pitch_class + interval) % 12 for interval in (0, 2, 3, 5, 7, 8, 10)}
    for bar in range(bars):
        tokens.append(BAR)
        last_step = 0
        for n in range(notes_per_bar):
            candidates = token_ids("STEP", range(last_step, 16))
            if n >= 2:
                candidates += [BAR]  # BAR means stop current bar; not appended yet
            step_token = pick(model, tokens, candidates, temperature, top_k)
            if step_token == BAR:
                break
            tokens.append(step_token)
            last_step = int(next(k[5:] for k, v in TOKEN_TO_ID.items() if v == step_token))
            role_token = pick(model, tokens, token_ids("ROLE", ROLES), temperature, top_k)
            tokens.append(role_token)
            role = next(k[5:] for k, v in TOKEN_TO_ID.items() if v == role_token)
            ranges = {"bass": (35, 59), "pad": (45, 80), "chords": (48, 81),
                      "arp": (55, 92), "lead": (55, 88), "drums": (36, 81),
                      "other": (55, 88)}
            low, high = ranges[role]
            pitches = [p for p in range(low, high + 1)
                       if role == "drums" or p % 12 in scale]
            tokens.append(pick(model, tokens, token_ids("PITCH", pitches), temperature, top_k))
            tokens.append(pick(model, tokens, token_ids("DUR", range(1, 17)), temperature, top_k))
            tokens.append(pick(model, tokens, token_ids("VEL", range(3, 9)), temperature, top_k))
    midi = decode_midi(tokens, bpm)
    if not midi.instruments:
        raise RuntimeError("Model produced no notes. Try more training or --demo.")
    return tokens, midi


def demo_motif(bpm=128, root=57) -> pretty_midi.PrettyMIDI:
    """Deterministic original four-bar pattern to test the full pipeline before ML."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    lead = pretty_midi.Instrument(program=80, name="lead")
    arp = pretty_midi.Instrument(program=10, name="arp")
    step_time = 60 / bpm / 4
    # Four bars, each melody has a recognizable rhythmic hook.
    offsets = [[0, 3, 7, 10], [0, 3, 5, 7], [3, 7, 10, 12], [7, 5, 3, 0]]
    for bar, line in enumerate(offsets):
        for pos, degree in zip((0, 4, 8, 12), line):
            start = (bar * 16 + pos) * step_time
            lead.notes.append(pretty_midi.Note(
                velocity=105, pitch=root + 12 + degree,
                start=start, end=start + step_time * 3,
            ))
        for pos in (0, 2, 4, 6, 8, 10, 12, 14):
            degree = (0, 7, 3, 7)[(pos // 2) % 4]
            start = (bar * 16 + pos) * step_time
            arp.notes.append(pretty_midi.Note(
                velocity=45, pitch=root + 12 + degree,
                start=start, end=start + step_time * 1.4,
            ))
    midi.instruments.extend([lead, arp])
    return midi
