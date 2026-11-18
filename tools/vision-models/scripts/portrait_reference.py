"""Run MODNet under onnxruntime on one frame, and save what the vision
crate's portrait test checks burn against.

    uv run --with onnxruntime --with numpy --with safetensors \
        python -I scripts/portrait_reference.py model.onnx OUT.safetensors

The frame is 512 by 640, a size the model reads as is (each side a whole
number of 32 pixels, the short side 512), so the crate's resize is not part
of the comparison. It is drawn here: a figure, a head and shoulders with
a darker fringe of hair, in front of a striped wall. The frame is
normalized as the crate does, so both read the same input.
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from safetensors.numpy import save_file

model, out = Path(sys.argv[1]), Path(sys.argv[2])
w, h = 512, 640

y, x = np.mgrid[0:h, 0:w].astype(np.float32)
rgb = np.zeros((h, w, 3), np.float32)
# A wall of soft vertical stripes.
rgb[:] = (0.55 + 0.1 * np.sin(x / 9.0))[..., None] * np.array([0.75, 0.8, 0.85])
# Shoulders and a body: half an ellipse rising from the bottom.
body = ((x - 256) / 210) ** 2 + ((y - 690) / 260) ** 2 < 1
rgb[body] = [0.2, 0.3, 0.55]
# A neck and a head, skin toned, shaded from the left.
neck = (np.abs(x - 256) < 42) & (y > 330) & (y < 450)
head = ((x - 256) / 95) ** 2 + ((y - 270) / 120) ** 2 < 1
skin = np.clip(0.75 + (256 - x) / 600, 0, 1)[..., None] * np.array([0.92, 0.72, 0.6])
rgb[neck | head] = skin[neck | head]
# Hair over the top of the head, with strands that break its edge.
hair = (((x - 256) / 105) ** 2 + ((y - 245) / 115) ** 2 < 1) & (y < 230)
strands = hair | ((((x - 256) / 112) ** 2 + ((y - 245) / 122) ** 2 < 1) & (y < 220)
                  & (np.sin(x * 0.9) > 0.3))
rgb[strands] = [0.18, 0.12, 0.08]
# Eyes and a mouth.
for ex in (220, 292):
    rgb[((x - ex) / 12) ** 2 + ((y - 275) / 7) ** 2 < 1] = [0.1, 0.1, 0.1]
rgb[(np.abs(x - 256) < 28) & (np.abs(y - 330) < 5)] = [0.6, 0.25, 0.25]
pixels = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)

normed = (pixels.astype(np.float32) / 255.0 - 0.5) / 0.5
tensor = np.ascontiguousarray(normed.transpose(2, 0, 1)[None])

session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
(matte,) = session.run(None, {"input": tensor})
matte = matte[0, 0].astype(np.float32)

out.parent.mkdir(parents=True, exist_ok=True)
save_file({"rgb": pixels, "matte": np.ascontiguousarray(matte)}, str(out))
print(f"{out}: {pixels.shape}, matte {matte.shape} from {matte.min():.3f} to "
      f"{matte.max():.3f}, {np.mean(matte > 0.5):.3f} kept")
