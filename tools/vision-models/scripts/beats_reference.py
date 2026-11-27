"""Run Beat This!'s small model under PyTorch on a drawn piece of music,
and save what the vision crate's beats test checks burn against.

    uv run --with "beat-this @ git+https://github.com/CPJKU/beat_this@REV" \
        --with numpy --with safetensors \
        python -I scripts/beats_reference.py small0.ckpt OUT.safetensors

The piece is 40 seconds at 22,050 Hz, longer than the 30 seconds the model
reads at once, so the joining of two readings is checked too. It is drawn
here: a kick on every beat at 120 a minute, louder on the first of each
four, a bass note that changes with the bar, and a hat between beats.

Saved: the piece, its spectrogram as the published code makes it, the
model's beat and downbeat answers for every frame, and the times the
published peak picking finds.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from beat_this.inference import load_model, split_predict_aggregate
from beat_this.model.postprocessor import Postprocessor
from beat_this.preprocessing import LogMelSpect
from safetensors.numpy import save_file

ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
RATE = 22050
SECONDS = 40
BEAT = 0.5  # 120 beats a minute

t = np.arange(RATE * SECONDS, dtype=np.float64) / RATE
signal = np.zeros_like(t)
rng = np.random.default_rng(7)
noise = rng.standard_normal(len(t))
BASS = [55.0, 65.41, 73.42, 49.0]
for beat in range(int(SECONDS / BEAT)):
    at = beat * BEAT
    on = t >= at
    # Time since the beat; zero before it, where nothing of it sounds.
    since = np.clip(t - at, 0, None)
    first = beat % 4 == 0
    # A kick: a tone that falls from 120 Hz to 50 Hz and dies in a tenth
    # of a second.
    sweep = 2 * np.pi * (50 * since + 70 * 0.03 * (1 - np.exp(-since / 0.03)))
    kick = np.sin(sweep) * np.exp(-since / 0.09)
    signal += np.where(on, kick, 0.0) * (0.9 if first else 0.55)
    # A hat halfway to the next beat: a short burst of noise.
    half = t - at - BEAT / 2
    hat = noise * np.exp(-np.clip(half, 0, None) / 0.02)
    signal += np.where(half >= 0, hat, 0.0) * 0.08
    # The bar's bass note, held for the beat.
    note = BASS[(beat // 4) % len(BASS)]
    held = on & (since < BEAT)
    signal += np.where(held, np.sin(2 * np.pi * note * since) * np.exp(-since / 0.4), 0.0) * 0.3
signal = (signal / np.abs(signal).max() * 0.8).astype(np.float32)

model = load_model(str(ckpt), "cpu")
with torch.inference_mode():
    spect = LogMelSpect()(torch.from_numpy(signal))
    answer = split_predict_aggregate(
        spect=spect, chunk_size=1500, border_size=6, overlap_mode="keep_first", model=model
    )
beat, downbeat = answer["beat"].float(), answer["downbeat"].float()
beat_times, downbeat_times = Postprocessor(type="minimal", fps=50)(beat, downbeat)

out.parent.mkdir(parents=True, exist_ok=True)
save_file(
    {
        "signal": signal,
        "spect": np.ascontiguousarray(spect.numpy().astype(np.float32)),
        "beat": np.ascontiguousarray(beat.numpy().astype(np.float32)),
        "downbeat": np.ascontiguousarray(downbeat.numpy().astype(np.float32)),
        "beat_times": np.asarray(beat_times, dtype=np.float64),
        "downbeat_times": np.asarray(downbeat_times, dtype=np.float64),
    },
    str(out),
)
print(
    f"{out}: {len(signal)} samples, {tuple(spect.shape)} spectrogram, "
    f"{len(beat_times)} beats, {len(downbeat_times)} downbeats"
)
