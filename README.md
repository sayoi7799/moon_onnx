# moon_onnx

A general-purpose ONNX inference runtime written in pure MoonBit. It reads
standard `.onnx` files and has no external dependencies, and the same code
runs on the **js**, **wasm-gc** and **native** backends.

- Includes its own protobuf decoder, tensors with NumPy broadcasting,
  graph execution, and 46 operators.
- Runs real image-classification models: MNIST, SqueezeNet 1.1,
  MobileNetV2 and ResNet-18. On all three backends the outputs agree with
  onnxruntime within tolerance and the top-5 classes are identical.
- Operator coverage is measured with the official ONNX backend tests
  (onnx v1.22.0). All **317** in-scope node tests pass, and all **118**
  out-of-scope tests (unsupported dtypes, training mode, and so on) are
  correctly rejected at load time. All **69** official pytorch-converted
  tests pass as well.
- Anything unsupported (operator, operator version, attribute, data type)
  is reported **while the model is loading**, naming the node. The runtime
  never silently computes a wrong result.

## How it differs from existing MoonBit projects

| Project | What it is | How moon_onnx differs |
|---|---|---|
| [HuLunTunTao/moon-ort](https://github.com/HuLunTunTao/moon-ort) | Bindings to the official ONNX Runtime C library; native only | moon_onnx is pure MoonBit with no external runtime library, and runs on js, wasm-gc and native |
| [OJII3/onnxruntime_web](https://github.com/OJII3/onnxruntime_web) | Bindings to onnxruntime-web; browser only | Same as above; no JavaScript or WASM runtime to ship alongside |
| [QUHONGYE-gif/modelhub](https://github.com/QUHONGYE-gif/modelhub) | Contains a pure-MoonBit ONNX executor with 11 operators (Gemm, MatMul, Add, Relu, ...) for small fully-connected models | 46 operators including convolution, pooling, normalization and shape operators. Runs CNNs such as ResNet-18. Handles operator versions from opset 1 to 25, and is measured against the official backend tests |
| soehi/moonlm, majikxu/laya, ... | Inference hand-written for one specific model architecture | Reads any ONNX file whose operators are in the supported set |

## Usage

```moonbit
let session = @runtime.Session::from_bytes(model_bytes)   // raises if unsupported
for spec in session.inputs() {
  println(spec.to_string())                               // e.g. "data: float32[N,3,224,224]"
}
let input = @tensor.Tensor::f32([1, 3, 224, 224], pixels) // or @tensor.from_proto_bytes(pb)
let outputs = session.run([("data", input)])              // [(name, Tensor)] in graph order
```

Errors are `@graph.OnnxError`: `DecodeFailed` (malformed protobuf, with the
byte offset), `Unsupported` / `InvalidModel` (at load time, naming the node),
`InvalidInput` (wrong name, dtype, rank or dimension) and `RuntimeFailure`
(naming the node), for example:

```
unsupported: node #0 'erf_node' (Erf): operator 'Erf' is not supported
unsupported: node #0 'bn' (BatchNormalization-15): BatchNormalization with training_mode=1 is not supported
invalid input: input 'Input3' dimension 2 must be 28, got 27 (shape [1,1,27,28])
decode error: protobuf decode error at byte 3 (model.graph): length-delimited field of 9 bytes exceeds remaining 2 bytes
```

### Command-line example

```bash
moon run cmd/onnx_run --target native --release -- .cache/models/resnet18-v1-7.onnx examples/chelsea.ppm --repeat 5
```

```
model    .cache/models/resnet18-v1-7.onnx (opset 8, 69 nodes, producer '')
operators Add-7 Conv-1 Gemm-7 Relu-6 Flatten-1 MaxPool-8 GlobalAveragePool-1 BatchNormalization-7
input    data: float32[N,3,224,224]
output   resnetv15_dense0_fwd: float32[N,1000]
tensor   float32[1,3,224,224] from examples/chelsea.ppm

top-5:
  #1  class 285  score 15.851795  p=55.9%
  #2  class 281  score 15.005581  p=23.98%
  ...
load     118 ms
infer    1092.2 ms average over 5 runs, best 1083 ms
```

`INPUT` can be a binary PPM or BMP image, which is resized and normalized
according to the model's input shape, or a serialized `TensorProto` (`.pb`),
which is fed to the model unchanged. Options: `--labels FILE` (one class
name per line), `--repeat N`, `--top K`, `--invert` (for MNIST-style models
when the image has a dark digit on light paper). To convert other image
formats:
`python -c "from PIL import Image; Image.open('a.jpg').convert('RGB').save('a.ppm')"`.

## Supported operators

Versions are listed as `since_version`. The version used by a node is the
largest since_version that is ≤ the model's opset. Versions older than the
ones listed here use legacy semantics and are rejected at load time.

| Group | Operators (oldest supported version) | Types |
|---|---|---|
| Convolution / pooling | Conv (1): groups, depthwise, dilations, `auto_pad`, 1-D/2-D/3-D · MaxPool (1): ceil_mode, dilations, Indices output · AveragePool (1): count_include_pad, ceil_mode · GlobalAveragePool (1) | float32 (MaxPool also uint8) |
| Normalization / activation | BatchNormalization (6, inference only) · Relu (6) · LeakyRelu (6) · Clip (6) · Sigmoid (6) · Tanh (6) · Softmax (1) · LogSoftmax (1) · Dropout (7, inference; training mode only with ratio 0) | float32/float64; Relu, Clip and Neg also accept integers |
| Linear algebra | Gemm (6) · MatMul (1) | float32 |
| Element-wise | Add, Sub, Mul, Div (6, NumPy broadcasting, including opset-6 `broadcast`/`axis`) · Pow (7) · Sqrt, Exp, Log, Neg, Abs (6) · Min, Max (6) | float32, float64, int32, int64, uint8 |
| Shape | Reshape (5) · Flatten (1) · Transpose (1) · Concat (4) · Squeeze, Unsqueeze (1) · Slice (1) · Gather (1) · Shape (1) · Cast (6) · Pad (2: constant/reflect/edge/wrap) · Identity (1) · Constant (1) · ConstantOfShape (9) · Expand (8) | all 6 MVP types |
| Reduction | ReduceMean, ReduceSum, ReduceMax (1, axes as attribute or input) · ArgMax (1) | numeric (ReduceMax also bool) |

Tensor element types: float32, float64, int32, int64, uint8, bool. Not
supported (all rejected at load time): control flow (If/Loop/Scan), sequences
and maps, quantized operators, string tensors, float16 and other dtypes,
external data files, training mode, and non-default operator domains.

## Official ONNX backend tests

Source: `onnx/backend/test/data/node` at onnx **v1.22.0**, the last release
that commits this data. A test is **in MVP scope** when all of its types and
operators are supported; this label comes from the test's contents (see
`scripts/gen_node_tests.py`), not from our results. An out-of-scope test
counts as **correctly rejected** only when loading or running fails with a
"not supported" error. Function-expanded tests (`*_expanded`) are counted
under the operator they expand. Expected outputs are the official `.pb`
files, compared with the official runner's tolerance (rtol 1e-3, atol 1e-7).
Results are identical on js, wasm-gc and native.

<details><summary>Per-operator results (node tests)</summary>

| operator | official tests | in MVP scope | passed | correctly rejected |
|---|---:|---:|---:|---:|
| Abs | 1 | 1 | 1 | 0 |
| Add | 8 | 3 | 3 | 5 |
| ArgMax | 16 | 16 | 16 | 0 |
| AveragePool | 20 | 20 | 20 | 0 |
| BatchNormalization | 4 | 2 | 2 | 2 |
| Cast | 60 | 2 | 2 | 58 |
| Clip | 24 | 10 | 10 | 14 |
| Concat | 12 | 12 | 12 | 0 |
| Constant | 1 | 1 | 1 | 0 |
| ConstantOfShape | 3 | 3 | 3 | 0 |
| Conv | 6 | 6 | 6 | 0 |
| Div | 10 | 5 | 5 | 5 |
| Dropout | 12 | 8 | 8 | 4 |
| Exp | 2 | 2 | 2 | 0 |
| Expand | 2 | 2 | 2 | 0 |
| Flatten | 9 | 9 | 9 | 0 |
| Gather | 4 | 4 | 4 | 0 |
| Gemm | 11 | 11 | 11 | 0 |
| GlobalAveragePool | 2 | 2 | 2 | 0 |
| Identity | 3 | 1 | 1 | 2 |
| LeakyRelu | 6 | 3 | 3 | 3 |
| Log | 2 | 2 | 2 | 0 |
| LogSoftmax | 21 | 21 | 21 | 0 |
| MatMul | 7 | 7 | 7 | 0 |
| Max | 14 | 8 | 8 | 6 |
| MaxPool | 19 | 19 | 19 | 0 |
| Min | 14 | 8 | 8 | 6 |
| Mul | 9 | 4 | 4 | 5 |
| Neg | 2 | 2 | 2 | 0 |
| Pad | 6 | 6 | 6 | 0 |
| Pow | 12 | 10 | 10 | 2 |
| ReduceMax | 10 | 10 | 10 | 0 |
| ReduceMean | 8 | 8 | 8 | 0 |
| ReduceSum | 12 | 12 | 12 | 0 |
| Relu | 2 | 1 | 1 | 1 |
| Reshape | 10 | 10 | 10 | 0 |
| Shape | 11 | 11 | 11 | 0 |
| Sigmoid | 2 | 2 | 2 | 0 |
| Slice | 8 | 8 | 8 | 0 |
| Softmax | 21 | 21 | 21 | 0 |
| Sqrt | 2 | 2 | 2 | 0 |
| Squeeze | 2 | 2 | 2 | 0 |
| Sub | 9 | 4 | 4 | 5 |
| Tanh | 2 | 2 | 2 | 0 |
| Transpose | 7 | 7 | 7 | 0 |
| Unsqueeze | 7 | 7 | 7 | 0 |
| **total** | **435** | **317** | **317** | **118** |

</details>

pytorch-converted tests (`onnx/backend/test/data/pytorch-converted`): the 69
tests that use only MVP operators. They cover Conv1d/2d/3d, grouped,
depthwise, dilated and strided convolutions, as well as pooling, padding and
BatchNorm.

<details><summary>Results (pytorch-converted)</summary>

| defining operator | official tests | in MVP scope | passed | correctly rejected |
|---|---:|---:|---:|---:|
| Abs | 1 | 1 | 1 | 0 |
| AveragePool | 7 | 7 | 7 | 0 |
| BatchNormalization | 5 | 5 | 5 | 0 |
| Conv | 26 | 26 | 26 | 0 |
| Exp | 1 | 1 | 1 | 0 |
| Gather | 2 | 2 | 2 | 0 |
| Gemm | 1 | 1 | 1 | 0 |
| LeakyRelu | 2 | 2 | 2 | 0 |
| LogSoftmax | 3 | 3 | 3 | 0 |
| MaxPool | 8 | 8 | 8 | 0 |
| Neg | 1 | 1 | 1 | 0 |
| Pad | 4 | 4 | 4 | 0 |
| Relu | 1 | 1 | 1 | 0 |
| Sigmoid | 1 | 1 | 1 | 0 |
| Softmax | 3 | 3 | 3 | 0 |
| Tanh | 1 | 1 | 1 | 0 |
| Transpose | 2 | 2 | 2 | 0 |
| **total** | **69** | **69** | **69** | **0** |

</details>

## Models

| Model | File (ONNX Model Zoo) | opset | Producer | License |
|---|---|---|---|---|
| MNIST | mnist-12.onnx | 12 | CNTK 2.5.1 | MIT |
| SqueezeNet 1.1 | squeezenet1.1-7.onnx | 7 | MXNet | Apache-2.0 |
| MobileNetV2 | mobilenetv2-12.onnx | 12 | PyTorch 1.8 | Apache-2.0 |
| ResNet-18 v1 | resnet18-v1-7.onnx | 8 | MXNet | Apache-2.0 |

Agreement with onnxruntime 1.31.1 (CPU, 1 thread, graph optimizations
disabled), with 5 inputs per model. The tolerance is
|ours − ort| ≤ 1e-4 + 1e-3·|ort|, and the top-5 classes must match exactly,
in the same order. All four models give identical results on js, wasm-gc and
native:

| Model | Inputs | Top-1 same | Top-5 same (incl. order) | Max abs error | Max rel error |
|---|---:|---:|---:|---:|---:|
| mnist-12 | 5 | 5/5 | 5/5 | 1.5e-3 (logits are on the order of 10³) | 2.5e-6 |
| squeezenet1.1-7 | 5 | 5/5 | 5/5 | 2.3e-5 | 7.7e-6 |
| mobilenetv2-12 | 5 | 5/5 | 5/5 | 1.5e-5 | 5.2e-4 |
| resnet18-v1-7 | 5 | 5/5 | 5/5 | 1.1e-5 | 4.7e-4 |

For comparison, onnxruntime's own optimized and unoptimized outputs for these
inputs differ by up to 1.5e-3 (MNIST) and 1.3e-5 (ImageNet models).

**Why this tolerance:** a float32 accumulation over about 20 layers picks up
about 1e-5 relative error from summation order alone (onnxruntime's MLAS
kernels use FMA and a different blocking). 1e-3 relative / 1e-4 absolute is
two orders of magnitude above that, yet far below any change that could
reorder classes. The top-5 check enforces that directly.

## Speed

Single thread, release builds, one 224×224 image (MNIST: one 28×28 image),
averaged over 5 runs on a Windows 11 laptop:

| Model | GMACs | native | wasm-gc | js (Node 24) | onnxruntime 1.31.1 (1 thread, MLAS) |
|---|---:|---:|---:|---:|---:|
| mnist-12 | 0.0005 | 0.8 ms | 1.4 ms | 8.6 ms | < 0.1 ms |
| squeezenet1.1-7 | 0.35 | 236 ms | 214 ms | 802 ms | 5.0 ms |
| mobilenetv2-12 | 0.30 | 211 ms | 208 ms | 841 ms | 5.8 ms |
| resnet18-v1-7 | 1.8 | 1092 ms | 950 ms | 3178 ms | 26.3 ms |

Timer resolution is 1 ms, so the MNIST times are only approximate. All
three backends predict the same class for the example image (285, chelsea).
Reproduce with `bash scripts/bench.sh 5` and `python scripts/bench_ort.py`.

Convolution uses im2col followed by a cache-blocked float32 GEMM
(i-k-j loop order). 1×1 convolutions skip im2col, and depthwise
convolutions use a direct loop. There is no SIMD or multithreading, which
explains most of the gap to onnxruntime. Raising performance is future
work; the MVP prioritizes correctness.

## Architecture

| Package | Role |
|---|---|
| `proto` | Protobuf wire-format decoder and ONNX messages (Model/Graph/Node/Attribute/Tensor/ValueInfo) |
| `tensor` | Tensor type (one contiguous buffer per dtype), shapes, strides, broadcasting, TensorProto → Tensor |
| `ops` | Operator kernels and the registry: version resolution, attribute and type validation at load time |
| `graph` | Model loading, topological sort, Constant folding, liveness-based release, execution |
| `runtime` | Public `Session` API |
| `vision` | Image decoding (PPM/BMP), resize/crop, ImageNet normalization, top-k (used by the examples) |
| `fileio` | File reading for tests and the CLI (native/wasm-gc: moonbitlang/x/fs; js: Node fs) |
| `cmd/onnx_run` | Command-line classifier |
| `tests/node`, `tests/models`, `tests/edge` | Official backend tests, whole-model tests, edge cases |

## Tests

`moon test --target js|wasm-gc|native` runs:

1. **protobuf / tensor**: decoder output compared with the `onnx` Python
   package for mnist-12, official test files, a hand-built model covering
   every attribute and storage form, truncated prefixes and malformed
   messages; TensorProto conversion compared with `onnx.numpy_helper`;
   broadcasting compared with NumPy.
2. **operators**: all official backend node tests and pytorch-converted
   tests, as above.
3. **models**: the four models against onnxruntime. mnist-12 is committed;
   the others are skipped with a message until `scripts/fetch_models.py`
   has downloaded them. CI downloads them and fails if any test is skipped.
4. **cross-backend**: each backend is checked against the same
   onnxruntime/ONNX reference values.
5. **edge cases**: truncated and corrupted model files (every prefix up to
   2 KB), unsupported operators, domains, attributes, dtypes and opsets,
   cycles and undefined values, input dtype/rank/dimension/symbolic-dimension
   mismatches, runtime errors that name the node, and empty tensors.

Test data is embedded as `.mbt` files where practical, so it runs on every
backend. The larger models are read from disk at test time; `fileio` makes
this work on wasm-gc as well, through moonrun.

## Regenerating the test data

All expected values come from the official ONNX test data and from
onnxruntime; none are written by hand. Versions are pinned in
`scripts/requirements.txt` (onnx 1.22.0, onnxruntime 1.31.1, numpy 2.3.5,
Python 3.12).

```bash
py -3.12 -m venv .cache/venv
.cache/venv/Scripts/python -m pip install -r scripts/requirements.txt
.cache/venv/Scripts/python scripts/fetch_test_data.py    # onnx v1.22.0 backend tests (git SHA-1 verified)
.cache/venv/Scripts/python scripts/fetch_models.py       # Model Zoo models + sample data (sha256 verified)
.cache/venv/Scripts/python scripts/gen_proto_fixtures.py # proto/fixtures_wbtest.mbt
.cache/venv/Scripts/python scripts/gen_tensor_fixtures.py
.cache/venv/Scripts/python scripts/gen_op_schemas.py     # ops/schemas.mbt from onnx.defs
.cache/venv/Scripts/python scripts/gen_node_tests.py     # tests/node/data_*_wbtest.mbt
.cache/venv/Scripts/python scripts/gen_model_tests.py    # testdata/expected, tests/models/manifest_wbtest.mbt
.cache/venv/Scripts/python scripts/gen_edge_fixtures.py
bash scripts/bench.sh 5 && .cache/venv/Scripts/python scripts/bench_ort.py
```

On Linux/macOS, use `.cache/venv/bin/python` instead.

## Sources and licenses

moon_onnx is licensed under Apache-2.0 (see `LICENSE`). It is an independent
implementation in MoonBit. Only the specifications and observable behavior
were consulted:

- [ONNX specification and operator documentation](https://onnx.ai/onnx/operators/)
  and `onnx.proto` (Apache-2.0) define the semantics. `ops/schemas.mbt` is
  generated from the `onnx.defs` operator schemas (names, versions,
  attribute lists).
- onnxruntime (MIT) behavior is used as the reference for model outputs.
  None of its code is used.

Third-party data in this repository or downloaded by the scripts:

| Data | Source | License |
|---|---|---|
| `tests/node/data_*_wbtest.mbt` (embedded unchanged) | onnx v1.22.0 `onnx/backend/test/data/{node,pytorch-converted}` | Apache-2.0 |
| `testdata/models/mnist-12.onnx`; downloaded squeezenet1.1-7, mobilenetv2-12, resnet18-v1-7 and their sample data | [ONNX Model Zoo](https://github.com/onnx/models) commit c5cae0f | MIT (MNIST), Apache-2.0 (others) |
| `testdata/images/*.pb`, `examples/chelsea.ppm`, `examples/coffee.bmp` | scikit-image sample data: astronaut and rocket (public domain), chelsea and coffee (CC0) | public domain / CC0 |
| `testdata/mnist/digit_*.pb`, `examples/digit3.ppm` | drawn with PIL's default font by `scripts/gen_model_tests.py` | (generated) |
| `testdata/expected/` | outputs of onnxruntime 1.31.1 | (generated) |
