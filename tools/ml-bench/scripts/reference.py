"""Run MobileSAM under onnxruntime on one frame and one click, and save what
the bench checks candle and burn against.

    python -I scripts/reference.py models frame.png X Y out/reference.safetensors

Needs numpy, pillow, onnxruntime and safetensors. X and Y are the click in
the frame's own pixels.
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from safetensors.numpy import save_file

models, image = Path(sys.argv[1]), Path(sys.argv[2])
cx, cy, out = float(sys.argv[3]), float(sys.argv[4]), Path(sys.argv[5])
cpu = ["CPUExecutionProvider"]
enc = ort.InferenceSession(str(models / "mobile_sam_image_encoder.onnx"), providers=cpu)
dec = ort.InferenceSession(str(models / "sam_mask_decoder_lowres.onnx"), providers=cpu)

img = Image.open(image).convert("RGB")
w, h = img.size
scale = 1024 / max(w, h)
size = (int(w * scale + 0.5), int(h * scale + 0.5))
resized = np.asarray(img.resize(size, Image.BILINEAR)).astype(np.float32)
(emb,) = enc.run(None, {"input_image": resized})
coords = np.array([[[cx * scale, cy * scale], [0.0, 0.0]]], dtype=np.float32)
labels = np.array([[1.0, -1.0]], dtype=np.float32)
iou, low = dec.run(None, {
    "image_embeddings": emb, "point_coords": coords, "point_labels": labels,
    "mask_input": np.zeros((1, 1, 256, 256), np.float32),
    "has_mask_input": np.zeros((1,), np.float32),
})
out.parent.mkdir(parents=True, exist_ok=True)
save_file({"resized": resized, "embeddings": emb, "point_coords": coords,
           "point_labels": labels, "iou": iou, "low_res_masks": low}, str(out))
print(f"{out}: scores {iou[0].round(4).tolist()}")
