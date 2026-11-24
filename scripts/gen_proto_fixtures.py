"""Generate proto decoder fixtures: proto/fixtures_wbtest.mbt.

Each fixture is a serialized ONNX message plus the summary that the official
`onnx` package produces for it (see proto_summary.py). Fixtures:

* mnist-12.onnx from the ONNX Model Zoo (real model, typed float_data storage)
* several official node-test models and tensors from onnx v1.22.0
* a hand-built model exercising every attribute type, every tensor storage
  form, non-tensor value types, symbolic dims, external data, sparse
  initializers, functions and unknown fields
* truncated prefixes of the above: when protobuf rejects a prefix our decoder
  must fail too; when protobuf accepts it (cut at a field boundary) the
  summaries must still agree.

Run: .cache/venv/Scripts/python scripts/gen_proto_fixtures.py
"""

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from google.protobuf.message import DecodeError

sys.path.insert(0, os.path.dirname(__file__))
import proto_summary  # noqa: E402
from common import CACHE, ROOT, mbt_base64, mbt_string, write_generated  # noqa: E402


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def handcrafted_model() -> bytes:
    sub = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "subgraph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    raw_f = numpy_helper.from_array(np.array([[1.5, -2.25], [3.0, 1e-30]], np.float32), "raw_f")
    typed_i32 = helper.make_tensor("typed_i32", TensorProto.INT32, [3], [-1, 0, 2147483647])
    typed_i64 = helper.make_tensor("typed_i64", TensorProto.INT64, [2], [-(2**63), 2**63 - 1])
    typed_f64 = helper.make_tensor("typed_f64", TensorProto.DOUBLE, [2], [0.1, -1e300])
    typed_u8 = helper.make_tensor("typed_u8", TensorProto.UINT8, [2], [0, 255])
    typed_bool = helper.make_tensor("typed_bool", TensorProto.BOOL, [2], [True, False])
    typed_u64 = helper.make_tensor("typed_u64", TensorProto.UINT64, [1], [2**64 - 1])
    raw_i64 = numpy_helper.from_array(np.array([7, -8], np.int64), "raw_i64")
    empty = numpy_helper.from_array(np.zeros((0, 3), np.float32), "empty")
    scalar = numpy_helper.from_array(np.array(42.0, np.float32), "scalar")
    strings = helper.make_tensor("strings", TensorProto.STRING, [2], [b"a", b"bc"])
    external = TensorProto(name="external", data_type=TensorProto.FLOAT, dims=[4])
    external.data_location = TensorProto.EXTERNAL
    e = external.external_data.add()
    e.key, e.value = "location", "weights.bin"
    sparse = helper.make_sparse_tensor(
        helper.make_tensor("vals", TensorProto.FLOAT, [1], [1.0]),
        helper.make_tensor("idx", TensorProto.INT64, [1], [0]),
        [4],
    )
    node = helper.make_node(
        "Custom",
        ["raw_f", "", "typed_i32"],
        ["out"],
        name="node_with_everything",
        domain="test.domain",
        f=0.5,
        i=-3,
        s=b"\x00\xffbytes",
        t=typed_f64,
        g=sub,
        floats=[1.0, -0.0, float("inf")],
        ints=[1, -1, 2**40],
        strings=[b"x", "中文".encode()],
        tensors=[typed_u8, typed_bool],
        graphs=[sub],
    )
    sparse_attr = helper.make_attribute("sp", sparse)
    node.attribute.append(sparse_attr)
    ref = AttributeProto_ref("ref_attr", "outer_name")
    node.attribute.append(ref)
    node2 = helper.make_node("Unicodeé", ["out"], ["final文"], name="n2")
    inputs = [
        helper.make_tensor_value_info("in_dyn", TensorProto.FLOAT, ["N", 3, None]),
        helper.make_tensor_value_info("in_noshape", TensorProto.INT64, None),
        helper.make_tensor_sequence_value_info("in_seq", TensorProto.FLOAT, [2]),
        helper.make_value_info("in_map", helper.make_map_type_proto(TensorProto.INT64, helper.make_tensor_type_proto(TensorProto.FLOAT, [1]))),
        helper.make_value_info("in_opt", helper.make_optional_type_proto(helper.make_tensor_type_proto(TensorProto.FLOAT, [1]))),
        helper.make_value_info("in_sparse", helper.make_sparse_tensor_type_proto(TensorProto.FLOAT, [4])),
        onnx.ValueInfoProto(name="in_notype"),
    ]
    graph = helper.make_graph(
        [node, node2],
        "everything",
        inputs,
        [helper.make_tensor_value_info("final文", TensorProto.FLOAT, [])],
        initializer=[raw_f, typed_i32, typed_i64, typed_f64, typed_u8, typed_bool, typed_u64, raw_i64, empty, scalar, strings, external],
        value_info=[helper.make_tensor_value_info("out", TensorProto.DOUBLE, [2, "M"])],
        sparse_initializer=[sparse],
    )
    func = helper.make_function("test.domain", "Fn", ["a"], ["b"], [helper.make_node("Identity", ["a"], ["b"])], [helper.make_opsetid("", 18)])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("test.domain", 1)],
        producer_name="moon_onnx fixture",
        producer_version="1.0",
        functions=[func],
    )
    model.domain = "org.example"
    model.model_version = 9
    data = model.SerializeToString()
    # Unknown fields that a conforming decoder must skip: field 99 of each wire type.
    data += varint((99 << 3) | 0) + varint(12345)
    data += varint((98 << 3) | 1) + b"\x01" * 8
    data += varint((97 << 3) | 2) + varint(3) + b"abc"
    data += varint((96 << 3) | 5) + b"\x02" * 4
    # An unknown group (with a nested group) and known fields with the wrong
    # wire type: protobuf skips both as unknown fields.
    data += varint((95 << 3) | 3) + varint((1 << 3) | 0) + varint(7)
    data += varint((94 << 3) | 3) + varint((94 << 3) | 4) + varint((95 << 3) | 4)
    data += varint((7 << 3) | 0) + varint(5)  # graph as varint
    data += varint((2 << 3) | 5) + b"\x00" * 4  # producer_name as fixed32
    # Re-parse to make sure protobuf accepts it.
    onnx.ModelProto.FromString(data)
    return data


def AttributeProto_ref(name, ref_name):
    a = onnx.AttributeProto(name=name, ref_attr_name=ref_name, type=onnx.AttributeProto.FLOAT)
    return a


def unpacked_tensor() -> bytes:
    """A TensorProto whose repeated fields are written *unpacked*.

    protobuf always writes [packed=true] fields packed, so encode by hand;
    protobuf parsers must accept both encodings.
    """
    out = bytearray()
    for d in (2, 3):
        out += varint((1 << 3) | 0) + varint(d)
    out += varint((2 << 3) | 0) + varint(TensorProto.INT32)
    for v in (-5, 0, 7, 1, 2, 3):
        out += varint((5 << 3) | 0) + varint(v & 0xFFFFFFFFFFFFFFFF)
    name = b"unpacked"
    out += varint((8 << 3) | 2) + varint(len(name)) + name
    data = bytes(out)
    TensorProto.FromString(data)
    return data


def node_test(name: str, file: str) -> bytes:
    with open(os.path.join(CACHE, "node", name, file), "rb") as f:
        return f.read()


def summarize(kind: str, data: bytes) -> str:
    if kind == "model":
        return proto_summary.model(onnx.ModelProto.FromString(data))
    return proto_summary.tensor(TensorProto.FromString(data)) + "\n"


def try_summarize(kind: str, data: bytes):
    try:
        return summarize(kind, data)
    except DecodeError:
        return None
    except Exception:  # protobuf can raise RuntimeWarning-like errors too
        return None


def main():
    with open(os.path.join(CACHE, "models", "mnist-12.onnx"), "rb") as f:
        mnist = f.read()
    fixtures = [
        ("mnist_12", "model", mnist),
        ("handcrafted", "model", handcrafted_model()),
        ("node_conv", "model", node_test("test_conv_with_strides_padding", "model.onnx")),
        ("node_constant", "model", node_test("test_constant", "model.onnx")),
        ("node_cast", "model", node_test("test_cast_DOUBLE_to_FLOAT", "model.onnx")),
        ("node_maxpool_uint8_input", "tensor", node_test("test_maxpool_2d_uint8", "test_data_set_0/input_0.pb")),
        ("node_argmax_output_i64", "tensor", node_test("test_argmax_keepdims_example", "test_data_set_0/output_0.pb")),
        ("node_reduce_bool_input", "tensor", node_test("test_reduce_max_bool_inputs", "test_data_set_0/input_0.pb")),
        ("node_cast_f64_input", "tensor", node_test("test_cast_DOUBLE_to_FLOAT", "test_data_set_0/input_0.pb")),
        ("node_float_output", "tensor", node_test("test_conv_with_strides_padding", "test_data_set_0/output_0.pb")),
        ("unpacked_tensor", "tensor", unpacked_tensor()),
    ]
    body = []
    body.append("///|\nlet fixtures : Array[(String, String, String, String)] = [\n")
    for name, kind, data in fixtures:
        body.append(f"  (\n    {mbt_string(name)},\n    {mbt_string(kind)},\n")
        body.append("    " + mbt_base64(data, "    ") + ",\n")
        body.append(f"    {mbt_string(summarize(kind, data))},\n  ),\n")
    body.append("]\n\n")

    # Truncated prefixes of the handcrafted model and of mnist (small set of cut points).
    body.append("///|\n/// (fixture name, prefix length, expected summary or \"\" if protobuf rejects it)\n")
    body.append("let truncations : Array[(String, Int, String)] = [\n")
    rejected = accepted = 0
    for name, kind, data in fixtures:
        if name not in ("handcrafted", "node_conv", "node_float_output"):
            continue
        cuts = sorted({1, 2, 3, 5, 8, 13, len(data) // 3, len(data) // 2, len(data) - 1, len(data) - 2}
                      | set(range(0, len(data), max(1, len(data) // 40))))
        for cut in cuts:
            if cut <= 0 or cut >= len(data):
                continue
            s = try_summarize(kind, data[:cut])
            if s is None:
                rejected += 1
            else:
                accepted += 1
            body.append(f"  ({mbt_string(name)}, {cut}, {mbt_string(s or '')}),\n")
    body.append("]\n\n")
    print(f"truncations: {rejected} rejected by protobuf, {accepted} accepted")

    # Malformed messages: each must be rejected by protobuf, and by us.
    malformed = [
        ("varint_11_bytes", varint((1 << 3) | 0) + b"\xff" * 10 + b"\x01"),
        ("group_unterminated", varint((9 << 3) | 3) + varint((1 << 3) | 0) + varint(1)),
        ("group_mismatched_end", varint((9 << 3) | 3) + varint((10 << 3) | 4)),
        ("end_group_alone", varint((9 << 3) | 4)),
        ("wire_type_7", varint((1 << 3) | 7) + b"\x00"),
        ("length_overflow", varint((7 << 3) | 2) + varint(1 << 40)),
        ("length_past_end", varint((2 << 3) | 2) + varint(10) + b"abc"),
        ("field_zero", varint(0) + b"\x00"),
        ("nested_truncated", varint((7 << 3) | 2) + varint(4) + varint((1 << 3) | 2) + varint(9) + b"ab"),
    ]
    body.append("///|\n/// Malformed ModelProto bytes, all rejected by the protobuf Python package.\n")
    body.append("let malformed : Array[(String, String)] = [\n")
    for name, data in malformed:
        assert try_summarize("model", data) is None, name
        body.append(f"  ({mbt_string(name)}, {mbt_string(data.hex())}),\n")
    body.append("]\n")
    write_generated(
        os.path.join(ROOT, "proto", "fixtures_wbtest.mbt"),
        "scripts/gen_proto_fixtures.py",
        "ONNX Model Zoo mnist-12.onnx; onnx v1.22.0 backend node tests; hand-built messages",
        ".cache/venv/Scripts/python scripts/gen_proto_fixtures.py",
        "".join(body),
    )


if __name__ == "__main__":
    main()
