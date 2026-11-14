"""Reference for the vision crate's subject detectors.

For each frame: OpenCV's own detectors (FaceDetectorYN, and the OpenCV Zoo
MediaPipe person and palm classes) give the boxes, onnxruntime gives the raw
outputs for the exact input tensors, and a numpy decode of those outputs is
checked against OpenCV before anything is written.

usage: detect_reference.py MODELS_DIR OUT.safetensors FRAME.png...
"""
import sys, pathlib
import numpy as np, cv2 as cv, onnxruntime as ort
from safetensors.numpy import save_file

models = pathlib.Path(sys.argv[1]); out = sys.argv[2]; frames = sys.argv[3:]

def anchors(size, layers):
    out = []
    for stride, n in layers:
        f = -(-size // stride)
        for y in range(f):
            for x in range(f):
                out += [((x + 0.5) / f, (y + 0.5) / f)] * n
    return np.array(out)

class SSD:
    """The OpenCV Zoo MediaPipe person and palm detectors, as their demos run them."""
    def __init__(self, path, size, layers, nchw, signed, kps):
        self.size = np.array([size, size]); self.anchors = anchors(size, layers)
        self.nchw, self.signed, self.kps = nchw, signed, kps
        self.model = cv.dnn.readNet(path)
    def _preprocess(self, image):
        pad_bias = np.array([0., 0.])
        ratio = min(self.size / image.shape[:2])
        ratio_size = (np.array(image.shape[:2]) * ratio).astype(np.int32)
        image = cv.resize(image, (ratio_size[1], ratio_size[0]))
        pad_h, pad_w = self.size[0] - ratio_size[0], self.size[1] - ratio_size[1]
        pad_bias[0] = left = pad_w // 2; pad_bias[1] = top = pad_h // 2
        image = cv.copyMakeBorder(image, top, pad_h - top, left, pad_w - left, cv.BORDER_CONSTANT, None, (0, 0, 0))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.signed: image = (image - 0.5) * 2
        pad_bias = (pad_bias / ratio).astype(np.int32)
        blob = image.transpose(2, 0, 1)[None] if self.nchw else image[None]
        return blob, pad_bias
    def _postprocess(self, out, shape, pad):
        score = 1 / (1 + np.exp(-np.clip(out[1][0, :, 0].astype(np.float64), -100, 100)))
        d = out[0][0]; scale = max(shape)
        c, wh = d[:, :2] / self.size, d[:, 2:4] / self.size
        boxes = np.c_[(c - wh / 2 + self.anchors) * scale, (c + wh / 2 + self.anchors) * scale] - [pad[0], pad[1], pad[0], pad[1]]
        keep = cv.dnn.NMSBoxes(boxes, score, 0.5, 0.3, top_k=5000)
        if len(keep) == 0: return np.empty((0, 5 + 2 * self.kps))
        keep = np.array(keep).reshape(-1)
        lm = d[keep, 4:].reshape(-1, self.kps, 2) / self.size + self.anchors[keep][:, None, :]
        lm = lm * scale - pad
        return np.c_[boxes[keep], lm.reshape(len(keep), -1), score[keep]]

yunet_onnx = str(models / "face_detection_yunet_2023mar.onnx")
person_onnx = str(models / "person_detection_mediapipe_2023mar.onnx")
palm_onnx = str(models / "palm_detection_mediapipe_2023feb.onnx")
persondet = SSD(person_onnx, 224, [(8, 2), (16, 2), (32, 6)], True, True, 4)
palmdet = SSD(palm_onnx, 192, [(8, 2), (16, 6)], False, False, 7)
sess = {n: ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for n, p in
        [("face", yunet_onnx), ("person", person_onnx), ("palm", palm_onnx)]}

def nms_order(boxes, scores, thr, nms):
    idx = cv.dnn.NMSBoxes([list(map(float, [b[0], b[1], b[2]-b[0], b[3]-b[1]])) for b in boxes], list(map(float, scores)), thr, nms)
    return np.array(idx).reshape(-1)

FACE_LONG = 640
tensors = {}
for k, path in enumerate(frames):
    bgr = cv.imread(path, cv.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    tensors[f"{k}.rgb"] = np.ascontiguousarray(bgr[:, :, ::-1])
    # Face: fit the long side to 640, pad right and bottom to 32.
    s = FACE_LONG / max(w, h); fw, fh = round(w * s), round(h * s)
    small = cv.resize(bgr, (fw, fh))
    # The ONNX file takes 640 by 640: the picture sits top left on black.
    pw = ph = FACE_LONG
    padded = np.zeros((ph, pw, 3), np.uint8); padded[:fh, :fw] = small
    yn = cv.FaceDetectorYN.create(yunet_onnx, "", (pw, ph), 0.6, 0.3, 5000)
    _, faces = yn.detect(padded)
    faces = np.zeros((0, 15), np.float32) if faces is None else faces
    blob = padded.transpose(2, 0, 1)[None].astype(np.float32)
    names = [o.name for o in sess["face"].get_outputs()]
    raw = dict(zip(names, sess["face"].run(None, {sess["face"].get_inputs()[0].name: blob})))
    boxes, scores = [], []
    for stride in (8, 16, 32):
        cols, rows = pw // stride, ph // stride
        cls = np.clip(raw[f"cls_{stride}"].reshape(-1), 0, 1); obj = np.clip(raw[f"obj_{stride}"].reshape(-1), 0, 1)
        bb = raw[f"bbox_{stride}"].reshape(-1, 4)
        for i in range(rows * cols):
            sc = np.sqrt(cls[i] * obj[i])
            if sc < 0.6: continue
            r, c = divmod(i, cols)
            cx, cy = (c + bb[i, 0]) * stride, (r + bb[i, 1]) * stride
            bw, bh = np.exp(bb[i, 2]) * stride, np.exp(bb[i, 3]) * stride
            boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]); scores.append(sc)
    keep = nms_order(boxes, scores, 0.6, 0.3) if boxes else []
    mine = np.array([[*boxes[i], scores[i]] for i in keep], np.float32).reshape(-1, 5)
    cvb = np.c_[faces[:, 0], faces[:, 1], faces[:, 0] + faces[:, 2], faces[:, 1] + faces[:, 3], faces[:, 14]].astype(np.float32)
    assert mine.shape == cvb.shape and np.allclose(mine, cvb, atol=0.5), (path, mine, cvb)
    print(path, "faces", (cvb[:, :4] / s).round(1).tolist())
    tensors[f"{k}.face.input"] = blob
    for n in names: tensors[f"{k}.face.{n}"] = raw[n]
    tensors[f"{k}.face.boxes"] = np.c_[cvb[:, :4] / s, cvb[:, 4:]].astype(np.float32)
    # Person and palm: the OpenCV Zoo classes, and onnxruntime on their input.
    for name, det in [("person", persondet), ("palm", palmdet)]:
        blob, pad = det._preprocess(bgr)
        blob = blob.astype(np.float32)
        # Boxes then scores, told apart by their width.
        order = lambda outs: sorted(outs, key=lambda o: -o.shape[-1])
        raw = order(sess[name].run(None, {sess[name].get_inputs()[0].name: blob}))
        det.model.setInput(blob)
        cvraw = order(det.model.forward(det.model.getUnconnectedOutLayersNames()))
        print(name, [o.shape for o in raw])
        res = det._postprocess(raw, np.array([w, h]), pad)
        res_cv = det._postprocess(cvraw, np.array([w, h]), pad)
        assert np.allclose(res, res_cv, atol=1.0), (name, res, res_cv)
        print(path, name, res[:, :4].round(1).tolist(), "kps", res[:, 4:-1].round(1).tolist(), "score", res[:, -1].round(3).tolist())
        tensors[f"{k}.{name}.input"] = blob
        tensors[f"{k}.{name}.boxes_out"] = raw[0].astype(np.float32)
        tensors[f"{k}.{name}.scores_out"] = raw[1].astype(np.float32)
        tensors[f"{k}.{name}.found"] = res.astype(np.float32)
# safetensors writes the buffer as it lies: a transposed array must be laid out anew.
save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, out)
print("wrote", out)
