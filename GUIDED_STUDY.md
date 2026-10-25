# Learn the repository in 10 small lessons

Your objective is to understand the *why*, not just run commands. For every
lesson: read the code, predict an output, run it, and explain the result.

## Lesson 0 — MIDI fundamentals (15–25 minutes)

**Read:** `scripts/00_inspect.py`

**Run:** `python scripts/00_inspect.py`

- A MIDI note is (pitch, onset, offset, velocity), not waveform audio.
- A MIDI *track name* (`Lead (MIDI)`) isn't the same as its General MIDI program.
- Your files often have descriptive names but piano instrument programs.

**Explain:** Why can't these files reproduce the original synth timbres?

## Lesson 1 — Vocabulary and tokenization (20–30 min)

**Read:** `musicgen/tokens.py`

A musical event looks like:

    BAR STEP_0 ROLE_lead PITCH_69 DUR_4 VEL_6

`BAR`: new bar. `STEP_0`: start of bar. `PITCH_69`: A4.
`DUR_4`: four sixteenth notes (one beat). `VEL_6`: velocity bin.

**Explain:** Why must pretraining and fine-tuning use identical vocabulary IDs?

## Lesson 2 — MIDI -> model input (30–45 min)

**Read:** `musicgen/midi_data.py`, then `scripts/01_prepare.py`

**Run:** `python scripts/01_prepare.py`

Quantization maps floating time to a regular 16th-note grid. Tempo changes are
handled when converting seconds to beats. Overlapping notes remain possible.

**Experiment:** Inspect the first 35 token strings:

```bash
python - <<'PY'
from musicgen.dataset import get_records
from musicgen.tokens import decode_ids
song = get_records('data/processed')[0]
print(song['name'], song['n_notes'], song['n_bars'])
print(decode_ids(song['tokens'][:35]))
PY
```

**Explain:** Why does discretization simplify learning, and what information is lost?

## Lesson 3 — The training objective (20–30 min)

**Read:** `musicgen/dataset.py`

For `[BOS, BAR, STEP_0, ...]`:

- x = `[BOS, BAR, STEP_0]`
- y = `[BAR, STEP_0, ...]`

At every position predict the **next token**. Padding targets are ignored.

**Explain:** Why is there only a one-token shift? What does cross-entropy measure?

## Lesson 4 — The neural network (40–60 min)

**Read:** `musicgen/model.py`

- Embedding maps discrete token IDs to learned vectors.
- Position embedding supplies order information.
- Causal attention cannot read future events.
- Output linear layer gives vocabulary logits for each position.

For input `[batch=8, time=128]`, output shape is `[8, 128, len(VOCAB)]`.

**Explain:** Why would non-causal attention make training too easy and generation
invalid? Why doesn't a Transformer inherently understand bars or musical keys?

## Lesson 5 — Learn the weights (40–60 min plus training)

**Read:** `musicgen/training.py`, then `scripts/02_train.py`

**Run:** `python scripts/02_train.py --steps 80`

`loss.backward()` computes gradients. `optimizer.step()` updates parameters.
A validation song is withheld so training progress isn't judged solely by
memorization. On seven songs this is a **toy from-scratch experiment**.

**Explain:** What is different between architecture, randomly initialized
weights, and learned weights? Why is decreasing training loss not enough?

## Lesson 6 — Next-token sampling (30–45 min)

**Read:** `musicgen/generation.py`, then `scripts/04_generate.py`

**Run:** `python scripts/04_generate.py --checkpoint checkpoints/toy_scratch.pt`

Use constrained categories: STEP, ROLE, PITCH, DUR, VEL. Sampling temperature
changes probability sharpness. Sampling repeatedly creates autoregressive
output. The A-minor pitch restriction is a hand-authored prior, not training.

**Experiment:** Compare `--temp 0.5` with `--temp 1.2`.

**Explain:** Why not always take `argmax`? Why doesn't 4 bars of generated notes
automatically make a complete composition?

## Lesson 7 — Long-range song structure (30–45 min)

**Read:** `musicgen/arrangement.py`, then `scripts/05_arrange.py`

- Intro: atmospheric and sparse.
- Buildup: adds chords, bass and growing drums.
- Drop 1: full instrumentation.
- Bridge: sparse.
- Drop 2: repeated hook, optionally higher layer.
- Outro: reduce density.

This is a **deterministic planner**, not a learned long-horizon model.

**Explain:** Why might a local next-token model fail to bring back a hook
three minutes later? How does repeating a 4-bar phrase solve part of that?

## Lesson 8 — MIDI isn't audio (20–35 min)

**Read:** `musicgen/synth.py`, then `scripts/06_render.py`

Waves synthesize pitches; envelopes control note onset/decay. Kick/snare/hats
are generated with simple oscillators and noise. This is useful for a demo,
not production-quality electronic music.

**Explain:** What makes the same MIDI notes sound completely different in a DAW?

## Lesson 9 — Transfer learning, evaluation, and originality (40–60 min)

**Read:** `scripts/03_adapt.py`, `scripts/07_check_copying.py`

- Pretraining = learning broad patterns from a separate, larger corpus.
- Fine-tuning = gradient updates to existing weights on target music.
- Conditioning = provide music context at inference; **no weight change**.
- Memorization = reproducing substantial stretches rather than composing new ones.

**Critical:** Seven songs are not enough for serious from-scratch pretraining.
A truly general checkpoint must be trained separately and be architecture+
vocabulary compatible. Exact n-gram checks don't prove legal originality.

**Explain:** Why can a pretrained model plus seven examples outperform a
random Transformer trained only on seven? When might fine-tuning make it worse?

## Your first two milestones

**Milestone A (engineering):** Run `--demo` -> 128-bar MIDI -> audible WAV.
This verifies the pipeline only; no neural model was used.

**Milestone B (ML):** Tokenize actual data -> train from scratch -> generate
4-bar motif -> arrange -> WAV. Judge loss, sample coherence, and memorization.

**Milestone C (research, later):** Pretrain on a larger licensed dataset or
integrate a genuinely compatible pretrained music model, then compare
conditioning vs. fine-tuning against baseline with listening tests.
