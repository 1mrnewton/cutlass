"""Run FastDVDnet under PyTorch on a noisy clip, and save what the vision
crate's denoise test checks burn against.

    uv run --with torch --with numpy --with safetensors \
        python -I scripts/denoise_reference.py WEIGHTS.pth OUT.safetensors

The clip is a moving pattern with clipped Gaussian noise, 70 by 46 so the
frame is padded to the model's grid, seven frames so both ends mirror.

The model is written out here as FastDVDnet's `models.py` defines it (MIT,
Matias Tassano), batch norms and all, so the test also checks the crate's
folding of them; the loop is its `denoise_seq_fastdvdnet`. Nothing from
its repository runs, and the weights load with `weights_only`.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from safetensors.numpy import save_file


def conv(i, o, stride=1, groups=1):
    return nn.Conv2d(i, o, 3, stride=stride, padding=1, groups=groups, bias=False)


def cv(i, o):
    return nn.Sequential(conv(i, o), nn.BatchNorm2d(o), nn.ReLU(), conv(o, o), nn.BatchNorm2d(o), nn.ReLU())


class Wrap(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.convblock = nn.Sequential(*layers)

    def forward(self, x):
        return self.convblock(x)


class Stage(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = Wrap(conv(12, 90, groups=3), nn.BatchNorm2d(90), nn.ReLU(), conv(90, 32), nn.BatchNorm2d(32), nn.ReLU())
        self.downc0 = Wrap(conv(32, 64, stride=2), nn.BatchNorm2d(64), nn.ReLU(), Wrap(*cv(64, 64)))
        self.downc1 = Wrap(conv(64, 128, stride=2), nn.BatchNorm2d(128), nn.ReLU(), Wrap(*cv(128, 128)))
        self.upc2 = Wrap(Wrap(*cv(128, 128)), conv(128, 256), nn.PixelShuffle(2))
        self.upc1 = Wrap(Wrap(*cv(64, 64)), conv(64, 128), nn.PixelShuffle(2))
        self.outc = Wrap(conv(32, 32), nn.BatchNorm2d(32), nn.ReLU(), conv(32, 3))

    def forward(self, a, b, c, noise):
        x0 = self.inc(torch.cat((a, noise, b, noise, c, noise), dim=1))
        x1 = self.downc0(x0)
        x2 = self.upc2(self.downc1(x1))
        x1 = self.upc1(x1 + x2)
        return b - self.outc(x0 + x1)


class FastDvd(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp1 = Stage()
        self.temp2 = Stage()

    def forward(self, x, noise):
        f = [x[:, 3 * m : 3 * m + 3] for m in range(5)]
        a = self.temp1(f[0], f[1], f[2], noise)
        b = self.temp1(f[1], f[2], f[3], noise)
        c = self.temp1(f[2], f[3], f[4], noise)
        return self.temp2(a, b, c, noise)


def denoise(model, seq, sigma):
    """FastDVDnet's own loop: five frames, mirrored at the ends, each frame
    padded to a multiple of four by reflection."""
    n, _, h, w = seq.shape
    ph, pw = (-h) % 4, (-w) % 4
    out = []
    for t in range(n):
        idx = []
        for k in range(t - 2, t + 3):
            k = abs(k)
            if k > n - 1:
                k = 2 * (n - 1) - k
            idx.append(k)
        x = seq[idx].reshape(1, 15, h, w)
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        noise = torch.full((1, 1, h + ph, w + pw), sigma)
        y = torch.clamp(model(x, noise), 0.0, 1.0)[:, :, :h, :w]
        out.append(y[0])
    return torch.stack(out)


def pattern(width, height, k):
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    channels = [((x + k) * (3 + c) + y * (7 - c) + (x * y) % (11 + c)) % 256 for c in range(3)]
    return np.stack(channels, axis=-1).astype(np.float64)


weights, out = Path(sys.argv[1]), Path(sys.argv[2])
level = 25.0
frames, width, height = 7, 70, 46
rng = np.random.default_rng(7)
noisy = np.stack(
    [np.clip(np.round(pattern(width, height, k) + rng.normal(0, level, (height, width, 3))), 0, 255) for k in range(frames)]
).astype(np.uint8)

model = FastDvd()
state = torch.load(weights, map_location="cpu", weights_only=True)
state = {k.removeprefix("module."): v for k, v in state.items()}
model.load_state_dict(state, strict=True)
model.eval()
with torch.no_grad():
    seq = torch.from_numpy(noisy.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    y = denoise(model, seq, level / 255.0).permute(0, 2, 3, 1).numpy()
denoised = np.ascontiguousarray(np.round(y * 255.0).astype(np.uint8))
out.parent.mkdir(parents=True, exist_ok=True)
save_file(
    {
        "noisy": np.ascontiguousarray(noisy),
        "denoised": denoised,
        "noise": np.array([level], dtype=np.float32),
    },
    str(out),
)
print(f"{out}: {noisy.shape}, noise {level}")
