"""Cut the stock SAM decoder down to the low-res masks and scores.

The stock export also resizes the masks to the frame's own size. candle's
decoder stops at the 256x256 masks, so the bench compares that much on both.
"""
import sys
import onnx.utils

src, dst = sys.argv[1:3]
onnx.utils.extract_model(
    src,
    dst,
    input_names=["image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input"],
    output_names=["iou_predictions", "low_res_masks"],
)
