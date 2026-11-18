"""Run BiRefNet_lite under PyTorch, with the model code published beside its
weights, on one frame at two sizes, and save what the vision crate's
matting test checks burn against.

    uv run --with torch --with torchvision --with transformers --with timm \
        --with kornia --with einops --with numpy --with safetensors \
        python -I scripts/matting_reference.py MODEL_DIR OUT.safetensors

MODEL_DIR holds the published config.json, BiRefNet_config.py, birefnet.py
and model.safetensors. The model reads a square; the frame is drawn at 512
and at 1024 pixels a side, sizes the crate reads as is, so its resize is
not part of the comparison. Both sizes pad the backbone's windows, and 512
keeps the CPU test quick. It is a figure beside a vase on a table, in
front of a striped wall. The frame is normalized as the crate does, so
both read the same input.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForImageSegmentation

model_dir, out = Path(sys.argv[1]), Path(sys.argv[2])


def draw(side):
    s = side / 512.0
    y, x = np.mgrid[0:side, 0:side].astype(np.float32) / s
    rgb = np.zeros((side, side, 3), np.float32)
    rgb[:] = (0.55 + 0.1 * np.sin(x / 9.0))[..., None] * np.array([0.75, 0.8, 0.85])
    table = y > 400
    rgb[table] = [0.45, 0.32, 0.2]
    # A figure: body, neck, head and hair.
    body = ((x - 200) / 120) ** 2 + ((y - 520) / 200) ** 2 < 1
    rgb[body] = [0.2, 0.3, 0.55]
    neck = (np.abs(x - 200) < 25) & (y > 250) & (y < 330)
    head = ((x - 200) / 60) ** 2 + ((y - 210) / 75) ** 2 < 1
    rgb[neck | head] = [0.9, 0.7, 0.58]
    hair = ((((x - 200) / 68) ** 2 + ((y - 195) / 80) ** 2 < 1) & (y < 180)
            & (np.sin(x * 0.9) > -0.6))
    rgb[hair] = [0.18, 0.12, 0.08]
    # A vase with a shaded body and a narrow neck, standing on the table.
    vase = (((x - 400) / 45) ** 2 + ((y - 360) / 55) ** 2 < 1) | (
        (np.abs(x - 400) < 14) & (y > 270) & (y < 320))
    shade = np.clip(0.5 + (400 - x) / 90, 0.2, 1)[..., None]
    rgb[vase] = (shade * np.array([0.85, 0.85, 0.3]))[vase]
    return np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


model = AutoModelForImageSegmentation.from_pretrained(str(model_dir), trust_remote_code=True)
model.eval()
torch.set_float32_matmul_precision("highest")
mean = np.array([0.485, 0.456, 0.406], np.float32)
spread = np.array([0.229, 0.224, 0.225], np.float32)
tensors = {}
for side in (512, 1024):
    pixels = draw(side)
    normed = (pixels.astype(np.float32) / 255.0 - mean) / spread
    tensor = torch.from_numpy(normed).permute(2, 0, 1)[None].contiguous()
    with torch.no_grad():
        matte = model(tensor)[-1].sigmoid()[0, 0].numpy().astype(np.float32)
    tensors[f"rgb_{side}"] = pixels
    tensors[f"matte_{side}"] = np.ascontiguousarray(matte)
    print(f"{side}: matte from {matte.min():.3f} to {matte.max():.3f}, "
          f"{np.mean(matte > 0.5):.3f} kept")

out.parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(out))
print(out)
