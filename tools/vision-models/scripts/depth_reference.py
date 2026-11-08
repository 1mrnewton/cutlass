"""Run Depth Anything V2 Small under transformers on one frame, and save
what the vision crate's depth test checks burn against.

    uv run --with torch --with transformers --with numpy --with safetensors \
        python -I scripts/depth_reference.py MODEL_DIR OUT.safetensors

MODEL_DIR holds the published config.json and model.safetensors. The
frame is 686 by 518, whole patches that are not square, so the position
embeddings are resampled as they are for a real clip. It is a lit scene
drawn here: a floor that recedes, a ball and a box in front of a wall.
The frame is normalized as the crate does, so both read the same input.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import DepthAnythingForDepthEstimation

model_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
w, h = 686, 518

y, x = np.mgrid[0:h, 0:w].astype(np.float32)
rgb = np.zeros((h, w, 3), np.float32)
# A wall, then a floor whose stripes close up towards the horizon.
horizon = h * 0.45
wall = y < horizon
rgb[wall] = [0.55, 0.6, 0.7]
depth_floor = np.where(wall, 1.0, (y - horizon) / (h - horizon) + 1e-3)
stripes = (np.floor(x / (8 + 60 * depth_floor)) % 2).astype(np.float32)
floor = ~wall
rgb[floor] = (0.3 + 0.35 * stripes[floor, None]) * np.array([0.8, 0.7, 0.55])
# A shaded ball.
cx, cy, r = 170.0, 250.0, 70.0
d2 = ((x - cx) ** 2 + (y - cy) ** 2) / r**2
ball = d2 < 1
nz = np.sqrt(np.clip(1 - d2, 0, 1))
shade = np.clip(0.25 + 0.75 * (nz * 0.7 + (cx - x) / r * 0.3 + (cy - y) / r * 0.3), 0, 1)
rgb[ball] = shade[ball, None] * np.array([0.9, 0.2, 0.15])
# A box nearer still, its top lighter than its front.
rgb[260:370, 330:470] = [0.2, 0.45, 0.25]
rgb[230:260, 345:485] = [0.35, 0.7, 0.4]
pixels = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)

mean = np.array([0.485, 0.456, 0.406], np.float32)
spread = np.array([0.229, 0.224, 0.225], np.float32)
normed = (pixels.astype(np.float32) / 255.0 - mean) / spread
tensor = torch.from_numpy(normed).permute(2, 0, 1)[None].contiguous()

model = DepthAnythingForDepthEstimation.from_pretrained(str(model_dir))
model.eval()
with torch.no_grad():
    depth = model(pixel_values=tensor).predicted_depth[0].numpy()

out.parent.mkdir(parents=True, exist_ok=True)
save_file({"rgb": pixels, "depth": np.ascontiguousarray(depth.astype(np.float32))}, str(out))
print(f"{out}: {pixels.shape}, depth {depth.shape} from {depth.min():.3f} to {depth.max():.3f}")
