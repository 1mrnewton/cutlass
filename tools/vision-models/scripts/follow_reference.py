"""Run SAM 2.1 hiera tiny under transformers' Sam2VideoModel, streaming one
frame at a time, and save what the vision crate's follow test checks burn
against.

    uv run --with torch --with transformers --with numpy --with safetensors \
        python -I scripts/follow_reference.py MODEL_DIR OUT.safetensors

MODEL_DIR holds the published config.json, the processor's configs and
model.safetensors. Three frames are drawn at 1024 pixels a side, the size the
model reads, so no resize is part of the comparison: a figure walking past a
vase in front of a striped wall. The frames are normalized as the crate does
and handed to the model as they are.

Three runs are saved, each the low-resolution logits (256 a side) and the
object score of every frame:

- a: one click on the figure's body on frame 0, then frames 1 and 2 followed
  from memory.
- b: the figure painted as a mask on frame 0, then frame 1 followed.
- c: a click kept on the head and one taken out on the body, frame 0 only,
  which picks its mask by stability rather than by score.

The image features of frame 0 are saved too, to find where a difference
starts.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import Sam2VideoModel, Sam2VideoProcessor

model_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
SIDE = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
SPREAD = np.array([0.229, 0.224, 0.225], np.float32)


def figure(step):
    """Where the figure stands on frame `step`: its body's centre."""
    return 380 + 36 * step, 600 + 6 * step


def draw(step):
    y, x = np.mgrid[0:SIDE, 0:SIDE].astype(np.float32)
    rgb = np.zeros((SIDE, SIDE, 3), np.float32)
    rgb[:] = (0.55 + 0.1 * np.sin(x / 18.0))[..., None] * np.array([0.75, 0.8, 0.85])
    rgb[y > 800] = [0.45, 0.32, 0.2]
    vase = (((x - 780) / 90) ** 2 + ((y - 700) / 110) ** 2 < 1) | (
        (np.abs(x - 780) < 28) & (y > 520) & (y < 620))
    shade = np.clip(0.5 + (780 - x) / 180, 0.2, 1)[..., None]
    rgb[vase] = (shade * np.array([0.85, 0.85, 0.3]))[vase]
    body, head = person(step, x, y)
    rgb[body] = [0.2, 0.3, 0.55]
    rgb[head] = [0.9, 0.7, 0.58]
    return np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def person(step, x, y):
    cx, cy = figure(step)
    body = ((x - cx) / 110) ** 2 + ((y - cy) / 190) ** 2 < 1
    head = ((x - cx) / 62) ** 2 + ((y - (cy - 270)) / 76) ** 2 < 1
    return body, head


def normalized(rgb):
    pixels = (rgb.astype(np.float32) / 255.0 - MEAN) / SPREAD
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


torch.set_float32_matmul_precision("highest")
model = Sam2VideoModel.from_pretrained(str(model_dir), attn_implementation="eager").eval()
processor = Sam2VideoProcessor.from_pretrained(str(model_dir))
frames = [draw(step) for step in range(3)]
tensors = {f"rgb_{i}": frame for i, frame in enumerate(frames)}

with torch.no_grad():
    features = model.get_image_features(normalized(frames[0])[None]).fpn_hidden_states
for name, level in zip(["s0", "s1", "top"], features):
    tensors[f"{name}_0"] = np.ascontiguousarray(level[:, 0].numpy())


def run(name, steps, prompt):
    session = processor.init_video_session(inference_device="cpu", dtype=torch.float32)
    for step in range(steps):
        if step == 0:
            prompt(session)
        with torch.no_grad():
            output = model(inference_session=session, frame=normalized(frames[step]))
        logits = output.pred_masks[0, 0].numpy().astype(np.float32)
        score = output.object_score_logits.reshape(-1).numpy().astype(np.float32)
        tensors[f"{name}_logits_{step}"] = np.ascontiguousarray(logits)
        tensors[f"{name}_score_{step}"] = score
        print(f"{name} {step}: logits from {logits.min():.2f} to {logits.max():.2f}, "
              f"{np.mean(logits > 0):.3f} kept, score {score[0]:.2f}")


def clicks(name, points):
    tensors[f"clicks_{name}"] = np.array(points, np.float32)

    def add(session):
        processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=0, obj_ids=1,
            input_points=[[[[p[0], p[1]] for p in points]]],
            input_labels=[[[int(p[2]) for p in points]]],
            original_size=(SIDE, SIDE))
    return add


cx, cy = figure(0)
run("a", 3, clicks("a", [[cx, cy, 1]]))

y, x = np.mgrid[0:SIDE, 0:SIDE].astype(np.float32)
body, head = person(0, x, y)
painted = (body | head).astype(np.uint8)
tensors["mask"] = painted * 255


def paint(session):
    processor.add_inputs_to_inference_session(
        inference_session=session, frame_idx=0, obj_ids=1, input_masks=painted.astype(bool))


run("b", 2, paint)

run("c", 1, clicks("c", [[cx, cy - 270, 1], [cx, cy + 60, 0]]))

out.parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(out))
print(out)
