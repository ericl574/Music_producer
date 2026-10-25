import numpy as np
import pretty_midi
import torch
from scipy.io import wavfile

from musicgen.tokens import VOCAB, BAR, BOS, EOS, PAD
from musicgen.midi_data import encode_midi, decode_midi, classify_role, save_record, load_record
from musicgen.dataset import TokenWindows
from musicgen.model import MusicTransformer, ModelConfig, save_model, load_model
from musicgen.generation import demo_motif, generate_motif
from musicgen.arrangement import arrange, SECTION_PLAN
from musicgen.synth import render


def test_fixed_vocabulary_and_role_detection():
    assert len(set(VOCAB)) == len(VOCAB)
    assert classify_role('Bass (MIDI)') == 'bass'
    assert classify_role('Vocals (MIDI)') == 'lead'
    assert classify_role('Chord Synth') == 'chords'


def test_midi_encoding_roundtrip(tmp_path):
    m = pretty_midi.PrettyMIDI(initial_tempo=120)
    lead = pretty_midi.Instrument(program=80, name='Lead (MIDI)')
    lead.notes.append(pretty_midi.Note(velocity=95, pitch=69, start=0, end=.25))
    lead.notes.append(pretty_midi.Note(velocity=100, pitch=72, start=.5, end=.75))
    m.instruments.append(lead)
    path = tmp_path / 'mini.mid'
    m.write(str(path))
    record = encode_midi(path)
    assert record['n_notes'] == 2
    assert record['tokens'][0] == BOS
    assert record['tokens'][-1] == EOS
    assert record['tokens'].count(BAR) == 1
    output = decode_midi(record['tokens'], bpm=120)
    assert len(output.instruments) == 1
    assert [n.pitch for n in output.instruments[0].notes] == [69, 72]
    save_record(record, tmp_path / 'mini.json.gz')
    assert load_record(tmp_path / 'mini.json.gz') == record


def test_training_windows():
    dataset = TokenWindows([{'tokens': [BOS, BAR, 5, 6, EOS]}], context=8, stride=4)
    x, y = dataset[0]
    assert x.shape == y.shape == (8,)
    assert int(x[0]) == BOS and int(y[0]) == BAR
    assert int(y[3]) == EOS
    assert int(y[-1]) == PAD


def test_model_logits_and_causality(tmp_path):
    torch.manual_seed(27)
    model = MusicTransformer(ModelConfig(context=16, width=32, heads=4, layers=1, dropout=0.0))
    model.eval()
    a = torch.tensor([[BOS, BAR, 5, 6, 7]])
    b = torch.tensor([[BOS, BAR, 5, 6, 8]])
    with torch.no_grad():
        la = model(a)
        lb = model(b)
    assert la.shape == (1, 5, len(VOCAB))
    torch.testing.assert_close(la[:, :4], lb[:, :4], atol=1e-5, rtol=1e-5)
    ckpt = tmp_path / 'model.pt'
    save_model(model, ckpt)
    loaded, _ = load_model(ckpt, device=torch.device('cpu'))
    loaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(a), la)


def test_generate_small_valid_motif():
    torch.manual_seed(27)
    model = MusicTransformer(ModelConfig(context=32, width=32, heads=4, layers=1, dropout=0.0))
    tokens, midi = generate_motif(model, bars=1, notes_per_bar=3, bpm=128)
    assert tokens.count(BAR) == 1
    assert sum(len(i.notes) for i in midi.instruments) >= 2


def test_arrangement_and_audio_render(tmp_path):
    motif = demo_motif(bpm=128)
    song = arrange(motif, bpm=128)
    assert sum(bars for _, bars in SECTION_PLAN) == 128
    names = [inst.name for inst in song.instruments]
    assert 'lead' in names and 'pad' in names and 'drums' in names
    assert 235 < song.get_end_time() <= 241
    # Use short MIDI so unit test doesn't render a four-minute audio file.
    short = pretty_midi.PrettyMIDI(initial_tempo=128)
    inst = pretty_midi.Instrument(program=80, name='lead')
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=69, start=0, end=.5))
    short.instruments.append(inst)
    path = tmp_path / 'test.wav'
    info = render(short, path, rate=8000, max_seconds=3)
    rate, data = wavfile.read(str(path))
    assert rate == 8000 and data.ndim == 2
    assert np.max(np.abs(data)) > 0
    assert info['notes_rendered'] == 1
