"""Generate tensor fixtures: tensor/fixtures_wbtest.mbt.

* TensorProto -> array conversions checked against onnx.numpy_helper.to_array
  for every MVP dtype in both raw_data and typed-field storage, the
  initializers of mnist-12, and official node-test .pb files.
* Malformed / unsupported TensorProtos that must be rejected.
* NumPy broadcasting: np.broadcast_shapes plus the actual element mapping
  (a + b on index-valued arrays).

Run: .cache/venv/Scripts/python scripts/gen_tensor_fixtures.py
"""

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, ROOT, fnv1a, mbt_base64, mbt_string, write_generated  # noqa: E402

DTYPE_NAME = {
    np.dtype(np.float32): "float32",
    np.dtype(np.float64): "float64",
    np.dtype(np.int32): "int32",
    np.dtype(np.int64): "int64",
    np.dtype(np.uint8): "uint8",
    np.dtype(np.bool_): "bool",
}


def describe(arr: np.ndarray) -> str:
    le = arr.astype(arr.dtype.newbyteorder("<"), copy=False)
    data = np.ascontiguousarray(le).tobytes()
    shape = "[" + ",".join(str(d) for d in arr.shape) + "]"
    return f"{DTYPE_NAME[arr.dtype]}{shape} n={arr.size} h={fnv1a(data):08x}"


def node_pb(test: str, file: str) -> bytes:
    with open(os.path.join(CACHE, "node", test, "test_data_set_0", file), "rb") as f:
        return f.read()


def main():
    accepted = []  # (name, bytes)

    # Typed-field storage for each dtype, including edge values.
    accepted += [
        ("typed_f32", helper.make_tensor("t", TensorProto.FLOAT, [2, 2], [1.5, -0.0, float("inf"), 3.4e38])),
        ("typed_f32_nan", helper.make_tensor("t", TensorProto.FLOAT, [1], [float("nan")])),
        ("typed_f64", helper.make_tensor("t", TensorProto.DOUBLE, [3], [0.1, -1e300, 5e-324])),
        ("typed_i32", helper.make_tensor("t", TensorProto.INT32, [3], [-2147483648, 0, 2147483647])),
        ("typed_i64", helper.make_tensor("t", TensorProto.INT64, [2], [-(2**63), 2**63 - 1])),
        ("typed_u8", helper.make_tensor("t", TensorProto.UINT8, [3], [0, 128, 255])),
        ("typed_bool", helper.make_tensor("t", TensorProto.BOOL, [3], [True, False, True])),
        ("typed_scalar", helper.make_tensor("t", TensorProto.FLOAT, [], [42.0])),
        ("typed_empty", helper.make_tensor("t", TensorProto.FLOAT, [0, 4], [])),
    ]
    # raw_data storage for each dtype.
    rng = np.random.default_rng(0)
    for dt in [np.float32, np.float64, np.int32, np.int64, np.uint8, np.bool_]:
        if dt == np.bool_:
            a = rng.integers(0, 2, (2, 3)).astype(np.bool_)
        elif np.issubdtype(dt, np.integer):
            info = np.iinfo(dt)
            a = rng.integers(info.min, info.max, (2, 3), dtype=dt, endpoint=True)
        else:
            a = rng.standard_normal((2, 3)).astype(dt)
        accepted.append((f"raw_{np.dtype(dt).name}", numpy_helper.from_array(a, "r")))
    accepted.append(("raw_empty", numpy_helper.from_array(np.zeros((3, 0), np.int64), "e")))
    accepted.append(("raw_scalar", numpy_helper.from_array(np.array(7, np.int64), "s")))

    # mnist-12 initializers (typed float_data / int64_data).
    mnist = onnx.load(os.path.join(CACHE, "models", "mnist-12.onnx"))
    for t in mnist.graph.initializer:
        accepted.append((f"mnist_{t.name}", t))

    # Official node-test data (raw_data).
    for test, file in [
        ("test_conv_with_strides_padding", "input_0.pb"),
        ("test_argmax_keepdims_example", "output_0.pb"),
        ("test_maxpool_2d_uint8", "input_0.pb"),
        ("test_reduce_max_bool_inputs", "input_0.pb"),
        ("test_cast_DOUBLE_to_FLOAT", "input_0.pb"),
        ("test_div_int32_trunc", "input_0.pb"),
        ("test_gather_0", "input_1.pb"),
        ("test_constantofshape_int_shape_zero", "output_0.pb"),
    ]:
        accepted.append((f"node_{test[5:]}_{file[:-3]}", TensorProto.FromString(node_pb(test, file))))

    rejected = []
    rejected.append(("float16", numpy_helper.from_array(np.ones(2, np.float16), "h"), "data type FLOAT16 is not supported"))
    rejected.append(("int8", numpy_helper.from_array(np.ones(2, np.int8), "i8"), "data type INT8 is not supported"))
    rejected.append(("string", helper.make_tensor("s", TensorProto.STRING, [1], [b"x"]), "data type STRING is not supported"))
    ext = TensorProto(name="ext", data_type=TensorProto.FLOAT, dims=[2], data_location=TensorProto.EXTERNAL)
    rejected.append(("external", ext, "external data is not supported"))
    short = numpy_helper.from_array(np.ones((2, 2), np.float32), "short")
    short.raw_data = short.raw_data[:-4]
    rejected.append(("raw_too_short", short, "raw_data has 12 bytes"))
    few = TensorProto(name="few", data_type=TensorProto.INT64, dims=[3], int64_data=[1, 2])
    rejected.append(("typed_too_few", few, "int64_data has 2 values, shape needs 3"))
    neg = TensorProto(name="neg", data_type=TensorProto.FLOAT, dims=[-1])
    rejected.append(("negative_dim", neg, "invalid dimension -1"))
    missing = TensorProto(name="missing", data_type=TensorProto.FLOAT, dims=[2])
    rejected.append(("no_data", missing, "float_data has 0 values, shape needs 2"))

    body = ["///|\nlet conversions : Array[(String, String, String)] = [\n"]
    for name, t in accepted:
        expected = describe(numpy_helper.to_array(t))
        body.append(f"  (\n    {mbt_string(name)},\n    {mbt_base64(t.SerializeToString(), '    ')},\n    {mbt_string(expected)},\n  ),\n")
    body.append("]\n\n")
    body.append("///|\n/// (name, TensorProto bytes, expected error substring)\n")
    body.append("let rejections : Array[(String, String, String)] = [\n")
    for name, t, err in rejected:
        body.append(f"  ({mbt_string(name)}, {mbt_string(__import__('base64').b64encode(t.SerializeToString()).decode())}, {mbt_string(err)}),\n")
    body.append("]\n\n")

    # Broadcasting.
    pairs = [
        ([], []), ([], [3]), ([3], []), ([1], [4]), ([4], [1]), ([3], [3]),
        ([2, 3], [3]), ([2, 3], [2, 1]), ([2, 1], [1, 3]), ([1, 3], [2, 1]),
        ([2, 3, 4], [4]), ([2, 3, 4], [3, 1]), ([2, 1, 4], [3, 1]), ([1, 1, 1], [2, 3, 4]),
        ([5, 1, 3, 1], [4, 1, 2]), ([2, 3, 4, 5], [1, 3, 1, 5]), ([1, 2, 1, 2, 1], [2, 1, 2, 1, 2]),
        ([0], [1]), ([0, 3], [1, 3]), ([2, 0], [2, 1]), ([1], [0]),
        ([2, 3], [4]), ([3], [4]), ([2, 3], [3, 2]), ([0], [2]),
        ([1, 4, 1, 1], [1, 4, 3, 5]), ([1, 10], [1, 10]),
    ]
    body.append("///|\n/// (a shape, b shape, broadcast shape or \"error\", flattened a+b for index-valued inputs)\n")
    body.append("let broadcasts : Array[(Array[Int], Array[Int], String, Array[Int])] = [\n")
    for sa, sb in pairs:
        try:
            out = np.broadcast_shapes(tuple(sa), tuple(sb))
            a = np.arange(int(np.prod(sa)), dtype=np.int64).reshape(sa)
            b = np.arange(int(np.prod(sb)), dtype=np.int64).reshape(sb) * 1000
            vals = (a + b).reshape(-1).tolist()
            res = "[" + ",".join(str(d) for d in out) + "]"
        except ValueError:
            res, vals = "error", []
        body.append(f"  ({sa}, {sb}, {mbt_string(res)}, {vals}),\n")
    body.append("]\n")
    write_generated(
        os.path.join(ROOT, "tensor", "fixtures_wbtest.mbt"),
        "scripts/gen_tensor_fixtures.py",
        "onnx.numpy_helper (onnx 1.22.0), numpy 2.3.5, mnist-12.onnx, onnx v1.22.0 node tests",
        ".cache/venv/Scripts/python scripts/gen_tensor_fixtures.py",
        "".join(body),
    )


if __name__ == "__main__":
    main()
