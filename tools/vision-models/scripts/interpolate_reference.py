"""Run RIFE v4.26 under PyTorch between two frames, and save what the
vision crate's interpolate test checks burn against.

    uv run --with torch --with numpy --with safetensors \
        python -I scripts/interpolate_reference.py WEIGHTS.pkl OUT.safetensors

The frames are a pattern moved three pixels across with a square moved
six pixels down, 100 by 70 so they are padded to the model's grid. The
frames a quarter and half of the way between them are saved.

The model is written out here as vs-rife's `IFNet_HDv3_v4_26.py` and
`warplayer.py` define it (MIT, after hzwer's Practical-RIFE), and padded
and timed as its `__init__.py` does. Nothing from either repository runs,
and the weights load with `weights_only`.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from safetensors.numpy import save_file


def warp(x, flow, div, grid):
    flow = torch.cat([flow[:, 0:1] / div[0], flow[:, 1:2] / div[1]], 1)
    g = (grid + flow).permute(0, 2, 3, 1)
    return F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True)


def conv(i, o, stride=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=True), nn.LeakyReLU(0.2, True))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x.clamp(0.0, 1.0)))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, inputs, c):
        super().__init__()
        self.conv0 = nn.Sequential(conv(inputs, c // 2, 2), conv(c // 2, c, 2))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear") / scale
            x = torch.cat((x, flow), 1)
        feat = self.convblock(self.conv0(x))
        tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear")
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 8, 192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, 128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, 96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, 64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, 32)
        self.encode = Head()
        self.scales = [16, 8, 4, 2, 1]

    def forward(self, img0, img1, timestep, div, grid):
        f0, f1 = self.encode(img0), self.encode(img1)
        w0, w1 = img0, img1
        flow = mask = feat = None
        for i, block in enumerate([self.block0, self.block1, self.block2, self.block3, self.block4]):
            if flow is None:
                flow, mask, feat = block(torch.cat((img0, img1, f0, f1, timestep), 1), None, self.scales[i])
            else:
                wf0 = warp(f0, flow[:, :2], div, grid)
                wf1 = warp(f1, flow[:, 2:4], div, grid)
                fd, mask, feat = block(torch.cat((w0, w1, wf0, wf1, timestep, mask, feat), 1), flow, self.scales[i])
                flow = flow + fd
            w0 = warp(img0, flow[:, :2], div, grid)
            w1 = warp(img1, flow[:, 2:4], div, grid)
        mask = torch.sigmoid(mask)
        return w0 * mask + w1 * (1 - mask)


def pattern(width, height, shift):
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    x = x + shift
    channels = [(x * (3 + c) + y * (7 - c) + (x * y) % (11 + c)) % 256 for c in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


weights, out = Path(sys.argv[1]), Path(sys.argv[2])
w, h = 100, 70
a = pattern(w, h, 0)
b = pattern(w, h, 3)
a[20:40, 30:50] = (250, 240, 30)
b[26:46, 30:50] = (250, 240, 30)

state = torch.load(weights, map_location="cpu", weights_only=True)
state = {k.removeprefix("module."): v for k, v in state.items() if k.startswith("module.")}
model = IFNet()
missing, unexpected = model.load_state_dict(state, strict=False)
assert not missing, missing
assert all(k.startswith(("teacher.", "caltime.")) for k in unexpected), unexpected
model.eval()

pw, ph = math.ceil(w / 64) * 64, math.ceil(h / 64) * 64
div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0])
across = torch.linspace(-1.0, 1.0, pw).view(1, 1, 1, pw).expand(-1, -1, ph, -1)
down = torch.linspace(-1.0, 1.0, ph).view(1, 1, ph, 1).expand(-1, -1, -1, pw)
grid = torch.cat([across, down], 1)


def tensor(rgb):
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    return F.pad(x, (0, pw - w, 0, ph - h))


saved = {"a": a, "b": b}
with torch.no_grad():
    for name, t in [("quarter", 0.25), ("half", 0.5)]:
        y = model(tensor(a), tensor(b), torch.full([1, 1, ph, pw], t), div, grid)
        y = y[0, :, :h, :w].clamp(0, 1).permute(1, 2, 0).numpy()
        saved[name] = np.round(y * 255.0).astype(np.uint8)
out.parent.mkdir(parents=True, exist_ok=True)
save_file({k: np.ascontiguousarray(v) for k, v in saved.items()}, str(out))
print(f"{out}: {a.shape}, at 0.25 and 0.5")
