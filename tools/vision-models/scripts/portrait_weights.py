"""Read MODNet's weights out of the published ONNX file and save them as
safetensors, named as the vision crate's portrait module names them.

    uv run --with onnx --with numpy --with safetensors \
        python -I scripts/portrait_weights.py model.onnx OUT.safetensors

There is no PyTorch checkpoint of MODNet on Hugging Face, only this ONNX
export. Its MobileNetV2 backbone has every batch norm folded into the
convolution before it, and those weights lost their names, so they are
named here by their place in the graph: the stem, seventeen inverted
residual blocks (the first with no expansion), and the last 1x1
convolution. The branches kept their PyTorch names, which map one to one.
Weights stay in PyTorch's layout; the crate's converter turns them into
burn's.

Every constant the crate builds into the model instead of reading (the
ReLU6 clips, the resize scales) is checked here, so a different export
fails loudly instead of converting into something wrong.
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

src, out = Path(sys.argv[1]), Path(sys.argv[2])
graph = onnx.load(str(src)).graph
inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
constants = {}
for node in graph.node:
    if node.op_type == "Constant":
        constants[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)

# MobileNetV2's blocks: (expansion, channels, repeats, first stride).
SETTINGS = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
            (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
names = ["backbone.stem"]
block = 0
for expansion, _, repeats, _ in SETTINGS:
    for _ in range(repeats):
        parts = (["expand"] if expansion != 1 else []) + ["depthwise", "project"]
        names += [f"backbone.blocks.{block}.{part}" for part in parts]
        block += 1
names.append("backbone.last")
assert block == 17 and len(names) == 52, (block, len(names))

tensors = {}
convs = [n for n in graph.node if n.op_type == "Conv"]
for name, node in zip(names, convs):
    weight, bias = node.input[1], node.input[2]
    assert weight in inits and not weight.startswith(("lr_", "hr_", "f_")), weight
    tensors[f"{name}.weight"] = inits[weight]
    tensors[f"{name}.bias"] = inits[bias]

# Every ReLU6 is a Clip from 0 to 6.
for node in graph.node:
    if node.op_type == "Clip":
        low, high = (float(constants[x]) for x in node.input[1:3])
        assert (low, high) == (0.0, 6.0), (node.name, low, high)

# The squeeze-and-excitation block: two bias-free linears, x @ W in ONNX.
matmuls = [n for n in graph.node if n.op_type == "MatMul"]
assert len(matmuls) == 2
tensors["se.fc1.weight"] = inits[matmuls[0].input[1]].T
tensors["se.fc2.weight"] = inits[matmuls[1].input[1]].T

# Every resize is bilinear with half-pixel centres, by a power of two.
for node in graph.node:
    if node.op_type == "Resize":
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        assert attrs["mode"] == b"linear", attrs
        assert attrs["coordinate_transformation_mode"] == b"pytorch_half_pixel", attrs
        scales = inits[node.input[2]]
        assert list(scales[:2]) == [1.0, 1.0] and scales[2] == scales[3], scales
        assert scales[2] in (0.25, 0.5, 2.0), scales

# Instance norms have no learned scale or shift.
for node in graph.node:
    if node.op_type == "InstanceNormalization":
        scale, shift = (constants.get(x, inits.get(x)) for x in node.input[1:3])
        assert np.all(scale == 1) and np.all(shift == 0), node.name

for name, value in inits.items():
    if not name.startswith(("lr_branch.", "hr_branch.", "f_branch.")):
        continue
    if "num_batches_tracked" in name:
        continue
    key = name.replace(".layers.0.", ".conv.").replace(".layers.1.bnorm.", ".bnorm.")
    key = key.replace("f_branch.conv_f.0.", "f_branch.conv_f.")
    key = key.replace("f_branch.conv_f.1.conv.", "f_branch.out.")
    key = key.replace("lr_branch.", "lr.").replace("hr_branch.", "hr.").replace("f_branch.", "fusion.")
    tensors[key] = value

used = {node.input[i] for node in convs[:52] for i in (1, 2)}
used |= {m.input[1] for m in matmuls}
left = [n for n in inits if n not in used and not n.startswith(("lr_", "hr_", "f_"))]
# What is left is shape plumbing: concat sizes and resize scales.
for name in left:
    assert inits[name].size <= 4, (name, inits[name].shape)

out.parent.mkdir(parents=True, exist_ok=True)
save_file({k: np.ascontiguousarray(v.astype(np.float32)) for k, v in tensors.items()}, str(out))
print(f"{out}: {len(tensors)} tensors")
