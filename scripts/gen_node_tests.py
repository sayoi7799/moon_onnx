"""Generate tests/node/*.mbt from the official ONNX backend tests.

Input: .cache/node and .cache/pytorch-converted (scripts/fetch_test_data.py).
For every test: the model, its input .pb files and the expected output .pb
files are embedded (base64) unchanged, so all three backends run them.

Each test is labelled from its content, not from our results:
  "pass"            all types and operators are in the MVP scope;
  "reject:<reason>" it needs something outside the scope (a dtype such as
                    float16/int8, a non-tensor type, an operator outside the
                    MVP set, training-mode BatchNormalization or random
                    training-mode Dropout). Such a test passes when the model
                    is rejected with an error instead of producing a result.

Run: .cache/venv/Scripts/python scripts/gen_node_tests.py
"""

import glob
import json
import os
import re
import sys

import onnx
from onnx import TensorProto, numpy_helper

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, ROOT, mbt_base64, mbt_string, write_generated  # noqa: E402
from mvp import MVP_DTYPES, MVP_OPS  # noqa: E402

OUT = os.path.join(ROOT, "tests", "node")


def type_issues(tp, where, issues):
    w = tp.WhichOneof("value")
    if w == "tensor_type":
        if tp.tensor_type.elem_type not in MVP_DTYPES:
            issues.append(f"dtype {TensorProto.DataType.Name(tp.tensor_type.elem_type)}")
    elif w is not None:
        issues.append(f"{w.replace('_type', '')} type")


def classify(test_dir):
    m = onnx.load(os.path.join(test_dir, "model.onnx"))
    g = m.graph
    issues = []
    for v in list(g.input) + list(g.output):
        type_issues(v.type, v.name, issues)
    for t in g.initializer:
        if t.data_type not in MVP_DTYPES:
            issues.append(f"dtype {TensorProto.DataType.Name(t.data_type)}")
    for n in g.node:
        if n.op_type not in MVP_OPS or n.domain not in ("", "ai.onnx"):
            issues.append(f"operator {n.op_type}")
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type not in MVP_DTYPES:
                issues.append(f"dtype {TensorProto.DataType.Name(a.t.data_type)}")
            if n.op_type == "Cast" and a.name == "to" and a.i not in MVP_DTYPES:
                issues.append(f"dtype {TensorProto.DataType.Name(a.i)}")
            if n.op_type == "BatchNormalization" and a.name == "training_mode" and a.i == 1:
                issues.append("training-mode BatchNormalization")
        if n.op_type == "Dropout" and len(n.input) >= 3 and n.input[2]:
            # training_mode is a graph input here; read its test value
            idx = [v.name for v in g.input].index(n.input[2])
            tm = numpy_helper.to_array(onnx.load_tensor(os.path.join(test_dir, "test_data_set_0", f"input_{idx}.pb")))
            ratio = 0.5
            if len(n.input) >= 2 and n.input[1]:
                ridx = [v.name for v in g.input].index(n.input[1])
                ratio = float(numpy_helper.to_array(onnx.load_tensor(os.path.join(test_dir, "test_data_set_0", f"input_{ridx}.pb"))))
            if bool(tm) and ratio != 0.0:
                issues.append("random training-mode Dropout")
    issues = sorted(set(issues))
    return "pass" if not issues else "reject:" + ", ".join(issues)


def load_case(kind, test, family):
    d = os.path.join(CACHE, kind, test)
    with open(os.path.join(d, "model.onnx"), "rb") as f:
        model = f.read()
    sets = sorted(glob.glob(os.path.join(d, "test_data_set_*")))
    data_set = sets[0]
    def pbs(prefix):
        files = sorted(glob.glob(os.path.join(data_set, prefix + "_*.pb")), key=lambda p: int(re.findall(r"_(\d+)\.pb$", p)[0]))
        return [open(p, "rb").read() for p in files]
    return {
        "name": test,
        "family": family,
        "expect": classify(d),
        "model": model,
        "inputs": pbs("input"),
        "outputs": pbs("output"),
        "sets": len(sets),
    }


def pytorch_family(test_dir):
    m = onnx.load(os.path.join(test_dir, "model.onnx"))
    ops = [n.op_type for n in m.graph.node]
    # The defining op of a PyTorch module test: the first "heavy" op.
    for op in ops:
        if op not in ("Constant", "Identity", "Reshape", "Unsqueeze", "Squeeze", "Shape", "Gather", "Concat", "Cast"):
            return op
    return ops[0] if ops else "?"


def emit(cases, var):
    # One top-level block per case keeps each text segment small.
    out = []
    for i, c in enumerate(cases):
        out.append(f"///|\nlet {var}_{i} : NodeCase = {{\n")
        out.append(f"  name: {mbt_string(c['name'])},\n")
        out.append(f"  family: {mbt_string(c['family'])},\n")
        out.append(f"  expect: {mbt_string(c['expect'])},\n")
        out.append(f"  model: {mbt_base64(c['model'], '  ', 1000)},\n")
        out.append("  inputs: [\n")
        for b in c["inputs"]:
            out.append("    " + mbt_base64(b, "    ", 1000) + ",\n")
        out.append("  ],\n  outputs: [\n")
        for b in c["outputs"]:
            out.append("    " + mbt_base64(b, "    ", 1000) + ",\n")
        out.append("  ],\n}\n\n")
    names = ", ".join(f"{var}_{i}" for i in range(len(cases)))
    out.append(f"///|\nlet {var} : Array[NodeCase] = [{names}]\n")
    return "".join(out)


def snake(op):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", op).lower()


def main():
    with open(os.path.join(CACHE, "node_ops.json")) as f:
        families = json.load(f)
    by_family = {}
    for test, fam in sorted(families.items()):
        by_family.setdefault(fam, []).append(load_case("node", test, fam))
    pt = []
    for d in sorted(glob.glob(os.path.join(CACHE, "pytorch-converted", "test_*"))):
        test = os.path.basename(d)
        fam = pytorch_family(d)
        c = load_case("pytorch-converted", test, fam)
        if c["expect"].startswith("reject:") and "operator" in c["expect"]:
            continue  # uses an operator outside the MVP set: not part of the MVP suite
        pt.append(c)
    for f in glob.glob(os.path.join(OUT, "data_*.mbt")):
        os.remove(f)
    names = []
    cmd = ".cache/venv/Scripts/python scripts/gen_node_tests.py"
    src = "onnx v1.22.0 onnx/backend/test/data/node (Apache-2.0)"
    for fam, cases in sorted(by_family.items()):
        var = "node_" + snake(fam)
        names.append(var)
        write_generated(os.path.join(OUT, f"data_{snake(fam)}_wbtest.mbt"), "scripts/gen_node_tests.py", src, cmd, emit(cases, var))
    # pytorch-converted, split into a few files
    chunk = 12
    for i in range(0, len(pt), chunk):
        var = f"pytorch_{i // chunk}"
        names.append(var)
        write_generated(
            os.path.join(OUT, f"data_pytorch_{i // chunk}_wbtest.mbt"),
            "scripts/gen_node_tests.py",
            "onnx v1.22.0 onnx/backend/test/data/pytorch-converted (Apache-2.0)",
            cmd,
            emit(pt[i : i + chunk], var),
        )
    index = "///|\nlet node_suites : Array[Array[NodeCase]] = [\n" + "".join(f"  {n},\n" for n in names if n.startswith("node_")) + "]\n\n"
    index += "///|\nlet pytorch_suites : Array[Array[NodeCase]] = [\n" + "".join(f"  {n},\n" for n in names if n.startswith("pytorch_")) + "]\n"
    write_generated(os.path.join(OUT, "data_index_wbtest.mbt"), "scripts/gen_node_tests.py", src, cmd, index)
    n_node = sum(len(v) for v in by_family.values())
    n_pass = sum(1 for v in by_family.values() for c in v if c["expect"] == "pass")
    print(f"node tests: {n_node} ({n_pass} in scope), pytorch-converted: {len(pt)} ({sum(1 for c in pt if c['expect'] == 'pass')} in scope)")
    multi = [c["name"] for v in by_family.values() for c in v if c["sets"] > 1]
    if multi:
        print("note: tests with more than one data set (only set 0 used):", multi)


if __name__ == "__main__":
    main()
