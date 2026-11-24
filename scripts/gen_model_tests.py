"""Generate model-level test data: expected outputs from onnxruntime.

Inputs per model (at least 5):
* the Model Zoo sample inputs (test_data_set_*/input_0.pb from the .tar.gz,
  see scripts/fetch_models.py);
* ImageNet models: four scikit-image sample photos (astronaut and rocket:
  public domain; chelsea and coffee: CC0), resized so the short side is
  256 (PIL bilinear), centre-cropped to 224x224 and stored as uint8 CHW
  tensors in testdata/images/. The MoonBit test normalizes them with
  ImageNet mean/std in float32 exactly as done here;
* MNIST: digits 0-9 drawn with PIL's default font (generated here, no
  third-party content), stored as float32 [1,1,28,28] in testdata/mnist/.

Expected outputs: onnxruntime 1.31.1, CPUExecutionProvider, one thread,
graph optimizations disabled. As a sanity check the outputs with all
optimizations enabled are computed too; their difference is recorded.

Outputs: testdata/expected/<model>/<case>.pb and tests/models/manifest_wbtest.mbt.
Run: .cache/venv/Scripts/python scripts/gen_model_tests.py
"""

import glob
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from PIL import Image, ImageDraw, ImageFont
import skimage.data

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, ROOT, fnv1a, mbt_string, write_generated  # noqa: E402

MODELS = ["mnist-12", "squeezenet1.1-7", "mobilenetv2-12", "resnet18-v1-7"]
IMAGES = ["astronaut", "chelsea", "coffee", "rocket"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
TD = os.path.join(ROOT, "testdata")


def model_path(name):
    if name == "mnist-12":
        return os.path.join(TD, "models", "mnist-12.onnx")
    return os.path.join(CACHE, "models", name + ".onnx")


def crop_224(img: np.ndarray) -> np.ndarray:
    im = Image.fromarray(img).convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
    w, h = im.size
    left, top = (w - 224) // 2, (h - 224) // 2
    im = im.crop((left, top, left + 224, top + 224))
    return np.asarray(im, np.uint8).transpose(2, 0, 1).copy()  # CHW


def normalize(chw_u8: np.ndarray) -> np.ndarray:
    """float32 ops in the same order as the MoonBit side:
    ((x / 255) - mean[c]) / std[c]."""
    x = chw_u8.astype(np.float32) / np.float32(255.0)
    x = (x - MEAN[:, None, None]) / STD[:, None, None]
    return x[None].astype(np.float32)


def draw_digit(d: int) -> np.ndarray:
    im = Image.new("L", (28, 28), 0)
    font = ImageFont.load_default(size=22)
    draw = ImageDraw.Draw(im)
    l, t, r, b = draw.textbbox((0, 0), str(d), font=font)
    draw.text(((28 - (r - l)) / 2 - l, (28 - (b - t)) / 2 - t), str(d), fill=255, font=font)
    return np.asarray(im, np.float32)[None, None]  # 0..255 like the zoo inputs


def tensor_hash(a: np.ndarray) -> str:
    return f"{fnv1a(np.ascontiguousarray(a.astype('<f4')).tobytes()):08x}"


def sessions(path):
    def make(level):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = level
        return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return make(ort.GraphOptimizationLevel.ORT_DISABLE_ALL), make(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)


def save_pb(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(numpy_helper.from_array(arr).SerializeToString())


def main():
    os.makedirs(os.path.join(TD, "models"), exist_ok=True)
    src = os.path.join(CACHE, "models", "mnist-12.onnx")
    with open(src, "rb") as f, open(model_path("mnist-12"), "wb") as g:
        g.write(f.read())
    for name in IMAGES:
        save_pb(crop_224(getattr(skimage.data, name)()), os.path.join(TD, "images", name + ".pb"))
    digits = [3, 5, 7, 8]
    for d in digits:
        save_pb(draw_digit(d), os.path.join(TD, "mnist", f"digit_{d}.pb"))

    rows = []
    report = []
    for m in MODELS:
        path = model_path(m)
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing: run scripts/fetch_models.py")
        plain, opt = sessions(path)
        in_name = plain.get_inputs()[0].name
        cases = []
        for d in sorted(glob.glob(os.path.join(CACHE, "models", m + "-data", "test_data_set_*"))):
            rel = os.path.relpath(os.path.join(d, "input_0.pb"), ROOT).replace("\\", "/")
            x = numpy_helper.to_array(onnx.load_tensor(os.path.join(d, "input_0.pb")))
            cases.append((f"zoo_{os.path.basename(d)[-1]}", "pb", rel, x))
        if m == "mnist-12":
            for d in digits:
                rel = f"testdata/mnist/digit_{d}.pb"
                cases.append((f"digit_{d}", "pb", rel, numpy_helper.to_array(onnx.load_tensor(os.path.join(ROOT, rel)))))
        else:
            for name in IMAGES:
                rel = f"testdata/images/{name}.pb"
                u8 = numpy_helper.to_array(onnx.load_tensor(os.path.join(ROOT, rel)))
                cases.append((name, "image", rel, normalize(u8)))
        for case, kind, rel, x in cases:
            y = plain.run(None, {in_name: x})[0]
            y_opt = opt.run(None, {in_name: x})[0]
            opt_diff = float(np.max(np.abs(y - y_opt)))
            exp_rel = f"testdata/expected/{m}/{case}.pb"
            save_pb(y, os.path.join(ROOT, exp_rel))
            top5 = np.argsort(-y.reshape(-1), kind="stable")[:5].tolist()
            rows.append((m, case, kind, rel, exp_rel, tensor_hash(x), top5))
            report.append(f"{m:16s} {case:10s} top5={top5} ort_opt_vs_noopt_maxdiff={opt_diff:.2e}")
    body = [
        "///|\n/// (model, case, input kind, input file, expected output file,\n"
        "///  FNV-1a hash of the float32 input fed to onnxruntime, expected top-5)\n",
        "let model_cases : Array[(String, String, String, String, String, String, Array[Int])] = [\n",
    ]
    for m, case, kind, rel, exp, h, top5 in rows:
        body.append(
            f"  ({mbt_string(m)}, {mbt_string(case)}, {mbt_string(kind)}, {mbt_string(rel)}, "
            f"{mbt_string(exp)}, {mbt_string(h)}, {top5}),\n"
        )
    body.append("]\n")
    write_generated(
        os.path.join(ROOT, "tests", "models", "manifest_wbtest.mbt"),
        "scripts/gen_model_tests.py",
        "onnxruntime 1.31.1 (CPU, 1 thread, optimizations disabled); ONNX Model Zoo inputs; scikit-image sample photos",
        ".cache/venv/Scripts/python scripts/gen_model_tests.py",
        "".join(body),
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
