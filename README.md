# Neural Music Producer — learn a music Transformer by building it

A **real, runnable educational project**. It accepts multi-track MIDI files,
encodes them as musical tokens, trains a small causal Transformer, samples new
motifs, arranges a four-minute electronic song, and synthesizes a WAV file.

This is NOT a professionally trained music model and the basic audio renderer
is NOT a Porter Robinson-quality production engine. The arrangement is mostly
hand-designed. Those limitations are deliberate and visible in the source.

## Start here: the exact order

1. `GUIDED_STUDY.md` — conceptual roadmap, tests, interview questions.
2. `scripts/00_inspect.py` — examine your 7 songs and MIDI tracks.
3. `musicgen/tokens.py` — understand the vocabulary (events -> integers).
4. `musicgen/midi_data.py` — quantize and encode MIDI; `scripts/01_prepare.py`.
5. `musicgen/dataset.py` — shift labels by one for next-token training.
6. `musicgen/model.py` — embedding, causal attention, logits.
7. `musicgen/training.py` — pretraining and adaptation logic; `scripts/02_train.py`.
8. `musicgen/generation.py` — sample new event tokens; `scripts/04_generate.py`.
9. `musicgen/arrangement.py` — motif -> sections; `scripts/05_arrange.py`.
10. `musicgen/synth.py` — notes -> synthesized WAV; `scripts/06_render.py`.
11. `scripts/03_adapt.py` — revisit once you have a **separately pretrained** compatible checkpoint.
12. `scripts/07_check_copying.py` — preliminary novelty check.

**Read one module, run its script, then come back and explain it in your own
words.** Don't try to learn the whole repo in a single sitting.

## 0. Setup (Mac / conda)

From inside the new project folder:

```bash
conda activate ml
python -m pip install -r requirements.txt
python -m pytest -q
```

Your seven Porter Robinson MIDI files are **not shipped** with this repo.
After extracting this project next to your `lab1` folder within
`MIT_DeepLearning`, copy your *existing local* data:

```bash
# Run FROM MIT_DeepLearning (the parent folder):
cp lab1/data/midi/*.mid neural-music-producer/data/midi/
cd neural-music-producer
```

Don't publicly upload those third-party transcriptions without permission.
You can also work with MIDI files you're licensed to use.

## 1. Baseline demo (works without any MIDI files or ML training)

This checks the ENTIRE arrangement and audio path:

```bash
python scripts/04_generate.py --demo
python scripts/05_arrange.py
python scripts/06_render.py
```

Outputs:

- `outputs/motif.mid` — editable 4-bar melody and arpeggio.
- `outputs/full_song.mid` — 128 bars, six sections, multi-track arrangement.
- `outputs/full_song.wav` — basic stereo synthesizer playback (~4 minutes).

**IMPORTANT:** Demo notes are coded, not ML-generated. Drums, harmony, structure,
and basic synthetic sounds are programmed too. This demo must not be described
as a neural network composing a professional song.

## 2. Inspect and tokenize your actual seven MIDI files

```bash
python scripts/00_inspect.py --input-dir data/midi
python scripts/01_prepare.py --input-dir data/midi --out data/processed
```

The output is one `.json.gz` per input song, with metadata and integer tokens.
The tokenizer represents multiple instruments, 16th-note timing, pitch,
duration, and quantized velocity. It is NOT perfectly lossless: it assumes 4/4,
quantizes note timing, discards controller curves/sound design, and clips note
durations to eight beats.

## 3. Train from scratch — toy experiment on seven songs

```bash
python scripts/02_train.py --data data/processed \
  --steps 80 --batch 8 --context 128 \
  --out checkpoints/toy_scratch.pt
```

This is useful for LEARNING, but **NOT pretraining in the meaningful large-data
sense**. Seven MIDI transcriptions are insufficient for reliable musical
composition, and a random-initialized Transformer can produce bad phrases.
The validation loss is on a held-out song (if at least two are supplied).

Sample with your little trained model:

```bash
python scripts/04_generate.py --checkpoint checkpoints/toy_scratch.pt \
  --bars 4 --notes-per-bar 7
python scripts/05_arrange.py
python scripts/06_render.py
```

Evaluate the result by listening, comparing against baseline, and checking
whether the model is memorizing input songs.

## 4. What *real* pretraining would look like

Obtain a larger MIDI corpus **with appropriate training rights**, placed in
`data/general_midi/`. Keep held-out source songs separate from the Porter
adaptation set. Do NOT mix the target MIDI corpus into this pretraining data
if you want a clean comparison.

```bash
python scripts/01_prepare.py --input-dir data/general_midi --out data/general_processed
python scripts/02_train.py --data data/general_processed \
  --steps 10000 --out checkpoints/general.pt
```

10,000 steps is just an example, not evidence of successful convergence.
A truly good general-purpose checkpoint can require far more compute and data.

Then optional fine-tuning:

```bash
python scripts/03_adapt.py --checkpoint checkpoints/general.pt \
  --data data/processed --steps 25 \
  --out checkpoints/porter_adapted.pt
```

**This checkpoint format is custom.** You cannot load arbitrary Hugging Face
music model weights into our model class. A different pretrained architecture
would require its own tokenizer, model loader, and adaptation method.

## 5. Conditioning without changing weights

Give the model the start of one specific MIDI transcription as a prompt:

```bash
python scripts/04_generate.py --checkpoint checkpoints/general.pt \
  --primer-record data/processed/porter-robinson-divinity.json.gz \
  --primer-bars 1 --bars 4
```

Here, the first bar of output is literally the original reference (provided
as context), so do not claim the complete output is fully original. Test its
continuation and potential copying separately. Prompting does not update
weights, whereas fine-tuning does.

## 6. Check potential copying (limited method)

```bash
python scripts/07_check_copying.py --generated outputs/motif.mid
```

This only checks a very simple exact pitch-token subsequence. A low score does
NOT prove originality, copyright clearance, or musical independence.

## Project tree

```text
neural-music-producer/
├── README.md
├── GUIDED_STUDY.md
├── requirements.txt
├── .gitignore
├── musicgen/
│   ├── tokens.py
│   ├── midi_data.py
│   ├── dataset.py
│   ├── model.py
│   ├── training.py
│   ├── generation.py
│   ├── arrangement.py
│   └── synth.py
├── scripts/
│   ├── 00_inspect.py
│   ├── 01_prepare.py
│   ├── 02_train.py
│   ├── 03_adapt.py
│   ├── 04_generate.py
│   ├── 05_arrange.py
│   ├── 06_render.py
│   └── 07_check_copying.py
├── tests/test_pipeline.py
├── data/midi/           # COPY your 7 local files here
├── checkpoints/         # model weights excluded from Git
└── outputs/             # generated music excluded from Git
```

## Known limitations / future research

- **Dataset:** 7 songs are tiny. Need broader licensed corpus or compatible pretrained model.
- **Tokenization:** tempo changes reduced to event positions; 4/4 only; one chord can be
  represented as several simultaneous note events; instruments mapped to roles.
- **Context:** 128 tokens is only a short span; the song planner provides longer form.
- **Conditioning:** uses source MIDI prefix, not text prompts or artist embeddings.
- **Arrangement:** fixed six-part template, not inferred from source composition.
- **Harmonies:** fixed A-minor chord progression may clash with model-generated notes.
- **Rendering:** simple oscillator synthesizer, no signature production or vocals.
- **Originality:** exact n-gram check is a weak proxy; review motifs and rights.
- **Generalization:** a falling training loss alone does NOT prove good new music.
