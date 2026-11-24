"""Generate edge-case models for tests/edge/fixtures_wbtest.mbt.

Small hand-built models (onnx.helper, not passed through the checker so
invalid graphs are possible): unsupported operators/domains/attributes/
dtypes/opsets, structurally invalid graphs, runtime failures and empty
tensors. The expected behaviour (which error, naming which node) is part of
the MoonBit test; for the cases that do run, the expected values are
computed here with onnxruntime.

Run: .cache/venv/Scripts/python scripts/gen_edge_fixtures.py
"""

import base64
import os
import sys

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, mbt_string, write_generated  # noqa: E402

F = TensorProto.FLOAT
I64 = TensorProto.INT64


def model(nodes, inputs, outputs, inits=(), opset=17, domains=()):
    g = helper.make_graph(nodes, "edge", inputs, outputs, initializer=list(inits))
    ops = [helper.make_opsetid("", opset)] + [helper.make_opsetid(d, 1) for d in domains]
    m = helper.make_model(g, opset_imports=ops, producer_name="moon_onnx edge fixtures")
    m.ir_version = 8
    return m.SerializeToString()


def vi(name, t, shape):
    return helper.make_tensor_value_info(name, t, shape)


def main():
    cases = {}
    x = vi("x", F, [2, 3])
    y = vi("y", F, [2, 3])
    cases["unsupported_op"] = model([helper.make_node("Erf", ["x"], ["y"], name="erf_node")], [x], [y])
    cases["custom_domain"] = model(
        [helper.make_node("FusedThing", ["x"], ["y"], name="fused", domain="com.example")], [x], [y], domains=["com.example"]
    )
    cases["unknown_attribute"] = model(
        [helper.make_node("Relu", ["x"], ["y"], name="relu_bad", flavour=3)], [x], [y]
    )
    cases["float16_input"] = model([helper.make_node("Relu", ["h"], ["y"], name="r")], [vi("h", TensorProto.FLOAT16, [2])], [vi("y", TensorProto.FLOAT16, [2])])
    cases["opset_too_new"] = model([helper.make_node("Relu", ["x"], ["y"], name="r")], [x], [y], opset=30)
    cases["legacy_version"] = model([helper.make_node("Relu", ["x"], ["y"], name="old_relu")], [x], [y], opset=1)
    cases["bad_auto_pad"] = model(
        [helper.make_node("MaxPool", ["x"], ["y"], name="pool", kernel_shape=[2], auto_pad="WEIRD")], [vi("x", F, [1, 1, 4])], [vi("y", F, None)]
    )
    cases["bn_training"] = model(
        [helper.make_node("BatchNormalization", ["x", "s", "b", "m", "v"], ["y"], name="bn", training_mode=1)],
        [vi("x", F, [1, 2, 2])],
        [vi("y", F, None)],
        inits=[numpy_helper.from_array(np.ones(2, np.float32), n) for n in "sbmv"],
        opset=15,
    )
    cases["undefined_input"] = model([helper.make_node("Relu", ["nowhere"], ["y"], name="lost")], [x], [y])
    cases["cycle"] = model(
        [helper.make_node("Add", ["x", "b"], ["a"], name="n1"), helper.make_node("Relu", ["a"], ["b"], name="n2")],
        [x],
        [vi("a", F, None)],
    )
    cases["duplicate_output"] = model(
        [helper.make_node("Relu", ["x"], ["y"], name="first"), helper.make_node("Neg", ["x"], ["y"], name="second")], [x], [y]
    )
    cases["output_not_produced"] = model([helper.make_node("Relu", ["x"], ["y"], name="r")], [x], [vi("z", F, None)])
    cases["sequence_input"] = model(
        [helper.make_node("Identity", ["s"], ["y"], name="id")],
        [helper.make_tensor_sequence_value_info("s", F, [2])],
        [helper.make_tensor_sequence_value_info("y", F, [2])],
    )
    # runtime failures
    cases["reshape_runtime"] = model(
        [helper.make_node("Reshape", ["x", "shape"], ["y"], name="reshape_bad")],
        [vi("x", F, ["N", 3]), vi("shape", I64, [2])],
        [vi("y", F, None)],
    )
    cases["gather_out_of_range"] = model(
        [helper.make_node("Gather", ["x", "idx"], ["y"], name="gather_bad", axis=0)],
        [vi("x", F, [2, 3]), vi("idx", I64, [1])],
        [vi("y", F, None)],
    )
    cases["int_div_zero"] = model(
        [helper.make_node("Div", ["a", "b"], ["c"], name="div0")],
        [vi("a", TensorProto.INT32, [2]), vi("b", TensorProto.INT32, [2])],
        [vi("c", TensorProto.INT32, [2])],
    )
    # symbolic dims shared by two inputs
    cases["symbolic_pair"] = model(
        [helper.make_node("Add", ["p", "q"], ["r"], name="add")],
        [vi("p", F, ["N", 3]), vi("q", F, ["N", 3])],
        [vi("r", F, ["N", 3])],
    )
    # empty tensors flowing through several ops
    empty_graph = model(
        [
            helper.make_node("Add", ["e", "e"], ["s"], name="add_empty"),
            helper.make_node("Relu", ["s"], ["r"], name="relu_empty"),
            helper.make_node("ReduceSum", ["r", "axes"], ["sum"], name="sum_empty", keepdims=0),
            helper.make_node("Transpose", ["r"], ["t"], name="transpose_empty"),
        ],
        [vi("e", F, [0, 3])],
        [vi("sum", F, None), vi("t", F, None)],
        inits=[numpy_helper.from_array(np.array([0], np.int64), "axes")],
    )
    cases["empty_tensors"] = empty_graph
    sess = ort.InferenceSession(empty_graph, providers=["CPUExecutionProvider"])
    s, t = sess.run(None, {"e": np.zeros((0, 3), np.float32)})
    empty_expect = (list(s.shape), [float(v) for v in s.tolist()], list(t.shape))

    body = ["///|\nlet edge_models : Map[String, String] = {\n"]
    for k, v in cases.items():
        body.append(f"  {mbt_string(k)}: {mbt_string(base64.b64encode(v).decode())},\n")
    body.append("}\n\n")
    sh, vals, tsh = empty_expect
    body.append(
        "///|\n/// onnxruntime result for empty_tensors: (sum shape, sum values, transpose shape)\n"
        f"let empty_expected : (Array[Int], Array[Double], Array[Int]) = ({sh}, {vals}, {tsh})\n"
    )
    write_generated(
        os.path.join(ROOT, "tests", "edge", "fixtures_wbtest.mbt"),
        "scripts/gen_edge_fixtures.py",
        "hand-built models (onnx.helper); expected values from onnxruntime 1.31.1",
        ".cache/venv/Scripts/python scripts/gen_edge_fixtures.py",
        "".join(body),
    )
    print("empty_tensors via onnxruntime:", empty_expect)


if __name__ == "__main__":
    main()
