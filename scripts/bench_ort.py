"""Reference timing: onnxruntime 1.31.1, CPU, 1 thread, all optimizations.
Run: .cache/venv/Scripts/python scripts/bench_ort.py"""
import time
import numpy as np
import onnxruntime as ort

for name, path, shape in [
    ("mnist-12", "testdata/models/mnist-12.onnx", (1, 1, 28, 28)),
    ("squeezenet1.1-7", ".cache/models/squeezenet1.1-7.onnx", (1, 3, 224, 224)),
    ("mobilenetv2-12", ".cache/models/mobilenetv2-12.onnx", (1, 3, 224, 224)),
    ("resnet18-v1-7", ".cache/models/resnet18-v1-7.onnx", (1, 3, 224, 224)),
]:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    feed = {s.get_inputs()[0].name: x}
    s.run(None, feed)
    ts = []
    for _ in range(20):
        t = time.perf_counter()
        s.run(None, feed)
        ts.append((time.perf_counter() - t) * 1000)
    print(f"onnxruntime | {name} | {np.mean(ts):.1f} ms average, best {min(ts):.1f} ms")
