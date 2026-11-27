"""Read Beat This!'s small checkpoint and save its weights as safetensors,
named as the vision crate's beats module names them.

    uv run --with torch --with numpy --with safetensors \
        python -I scripts/beats_weights.py small0.ckpt OUT.safetensors

The checkpoint is a PyTorch Lightning file. It is read with
`weights_only=True`, so nothing in it runs. Weights stay in PyTorch's
layout; the crate's converter turns them into burn's.

Everything the crate builds into the model instead of reading (its sizes
and the rotary position frequencies) is checked here, so a different
checkpoint fails loudly instead of converting into something wrong.
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

src, out = Path(sys.argv[1]), Path(sys.argv[2])
checkpoint = torch.load(str(src), map_location="cpu", weights_only=True)

# The small model: the main transformer 128 wide, everything else as the
# large one.
EXPECTED = {
    "spect_dim": 128,
    "transformer_dim": 128,
    "ff_mult": 4,
    "n_layers": 6,
    "head_dim": 32,
    "stem_dim": 32,
    "sum_head": True,
    "partial_transformers": True,
}
hyper = checkpoint["hyper_parameters"]
for key, value in EXPECTED.items():
    assert hyper[key] == value, (key, hyper[key], value)

# The rotary position frequencies, which the crate works out itself.
FREQS = 1.0 / (10000 ** (np.arange(0, 32, 2, dtype=np.float32) / 32))

RENAMES = [
    (r"^model\.", ""),
    (r"\.attnF\.", ".attn_f."),
    (r"\.attnT\.", ".attn_t."),
    (r"\.ffF\.", ".ff_f."),
    (r"\.ffT\.", ".ff_t."),
    (r"^transformer_blocks\.layers\.(\d+)\.0\.", r"transformer.layers.\1.attn."),
    (r"^transformer_blocks\.layers\.(\d+)\.1\.", r"transformer.layers.\1.ff."),
    (r"^transformer_blocks\.norm\.", "transformer.norm."),
    (r"\.to_out\.0\.", ".to_out."),
    (r"\.net\.0\.", ".norm."),
    (r"\.net\.1\.", ".inner."),
    (r"\.net\.4\.", ".out."),
    (r"^task_heads\.beat_downbeat_lin\.", "head.linear."),
]

tensors = {}
for name, value in checkpoint["state_dict"].items():
    value = value.numpy()
    if name.endswith("num_batches_tracked"):
        continue
    if name.endswith("rotary_embed.freqs"):
        assert np.allclose(value, FREQS, rtol=1e-6, atol=0), name
        continue
    key = name
    for pattern, to in RENAMES:
        key = re.sub(pattern, to, key)
    assert key not in tensors, key
    tensors[key] = np.ascontiguousarray(value.astype(np.float32))

weights = sum(value.size for value in tensors.values())
out.parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(out))
print(f"{out}: {len(tensors)} tensors, {weights} weights")
