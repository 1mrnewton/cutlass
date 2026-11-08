"""Run Real-ESRGAN's realesr-general-x4v3 under PyTorch on one frame, and
save what the vision crate's enhance test checks burn against.

    uv run --with torch --with numpy --with safetensors --with pillow \
        python -I scripts/enhance_reference.py WEIGHTS.pth OUT.safetensors [FRAME.png]

With no frame, a 48 by 32 pattern the test also draws. The model is
written out here, as Real-ESRGAN's SRVGGNetCompact, so nothing from its
package runs; the weights load with `weights_only`.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from safetensors.numpy import save_file


class Compact(nn.Module):
    def __init__(self, features=64, convs=32, scale=4):
        super().__init__()
        body = [nn.Conv2d(3, features, 3, 1, 1), nn.PReLU(features)]
        for _ in range(convs):
            body += [nn.Conv2d(features, features, 3, 1, 1), nn.PReLU(features)]
        body.append(nn.Conv2d(features, 3 * scale * scale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(scale)
        self.scale = scale

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")


def pattern(width, height):
    """The test's `frame`: every pixel differs."""
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    channels = [(x * (3 + k) + y * (7 - k) + (x * y) % (11 + k)) % 256 for k in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


weights, out = Path(sys.argv[1]), Path(sys.argv[2])
if len(sys.argv) > 3:
    from PIL import Image

    rgb = np.asarray(Image.open(sys.argv[3]).convert("RGB"))
else:
    rgb = pattern(48, 32)

model = Compact()
state = torch.load(weights, map_location="cpu", weights_only=True)["params"]
model.load_state_dict(state, strict=True)
model.eval()
with torch.no_grad():
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    # Contiguous, or safetensors writes the permuted view's bytes as they lie.
    y = np.ascontiguousarray(model(x)[0].clamp(0, 1).permute(1, 2, 0).numpy())
enhanced = np.ascontiguousarray(np.round(y * 255.0).astype(np.uint8))
out.parent.mkdir(parents=True, exist_ok=True)
save_file({"rgb": np.ascontiguousarray(rgb), "enhanced": enhanced, "float": y}, str(out))
print(f"{out}: {rgb.shape} to {enhanced.shape}")
