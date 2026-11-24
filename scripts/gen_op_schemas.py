"""Generate ops/schemas.mbt from the official ONNX operator schemas.

For every MVP operator and every since_version up to opset 25 it records the
allowed attribute names, required attributes and input/output arity. The
runtime uses this table to resolve which version of an operator a model uses
(the largest since_version <= the model's opset) and to reject attributes the
spec does not define.

Run: .cache/venv/Scripts/python scripts/gen_op_schemas.py
"""

import os
import sys

from onnx import defs

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, mbt_string, write_generated  # noqa: E402

MAX_OPSET = 25

OPS = (
    "Conv MaxPool AveragePool GlobalAveragePool BatchNormalization "
    "Relu LeakyRelu Clip Sigmoid Tanh Softmax LogSoftmax Gemm MatMul "
    "Add Sub Mul Div Pow Sqrt Exp Log Neg Abs Min Max "
    "Reshape Flatten Transpose Concat Squeeze Unsqueeze Slice Gather Shape Cast Pad "
    "Identity Constant ConstantOfShape Expand "
    "ReduceMean ReduceSum ReduceMax ArgMax Dropout"
).split()

VARIADIC = 1 << 20


def arity(params):
    lo = hi = 0
    for p in params:
        if p.option == p.option.Single:
            lo += 1
            hi += 1
        elif p.option == p.option.Optional:
            hi += 1
        else:  # variadic
            lo += p.min_arity
            hi = VARIADIC
    return lo, hi


def main():
    body = [
        "///|\n/// (op_type, since_version, attribute names, required attributes,\n"
        "///  min inputs, max inputs, min outputs, max outputs)\n",
        "let schema_table : Array[(String, Int, Array[String], Array[String], Int, Int, Int, Int)] = [\n",
    ]
    count = 0
    for op in OPS:
        seen = set()
        for v in range(1, MAX_OPSET + 1):
            try:
                s = defs.get_schema(op, v)
            except Exception:
                continue
            if s.since_version in seen:
                continue
            seen.add(s.since_version)
            attrs = sorted(s.attributes)
            req = sorted(a for a in attrs if s.attributes[a].required)
            ilo, ihi = arity(s.inputs)
            olo, ohi = arity(s.outputs)
            body.append(
                f"  ({mbt_string(op)}, {s.since_version}, [{', '.join(mbt_string(a) for a in attrs)}], "
                f"[{', '.join(mbt_string(a) for a in req)}], {ilo}, {ihi}, {olo}, {ohi}),\n"
            )
            count += 1
    body.append("]\n\n")
    body.append(f"///|\npub let max_supported_opset : Int = {MAX_OPSET}\n")
    write_generated(
        os.path.join(ROOT, "ops", "schemas.mbt"),
        "scripts/gen_op_schemas.py",
        "onnx 1.22.0 onnx.defs operator schemas",
        ".cache/venv/Scripts/python scripts/gen_op_schemas.py",
        "".join(body),
    )
    print(count, "schema versions")


if __name__ == "__main__":
    main()
