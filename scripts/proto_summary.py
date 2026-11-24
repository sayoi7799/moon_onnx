"""Canonical text summary of a decoded ONNX protobuf message.

The MoonBit test suite (proto/summary_test.mbt) produces exactly the same
format from its own decoder, so comparing the two strings checks every
decoded field against the official `onnx` / `protobuf` Python packages.

Floating-point values are printed as IEEE bit patterns so the comparison is
exact. Bulk data is summarised as (element count, FNV-1a hash of the
little-endian bytes).
"""

import struct

from onnx import AttributeProto, TensorProto

from common import fnv1a


def _hex(b: bytes) -> str:
    return b.hex()


def _bytes_field(b: bytes) -> str:
    if len(b) <= 32:
        return "x" + _hex(b)
    return f"{len(b)}:{fnv1a(b):08x}"


def _packed(fmt: str, values) -> str:
    data = b"".join(struct.pack("<" + fmt, v) for v in values)
    return f"{len(values)}:{fnv1a(data):08x}"


def tensor(t: TensorProto) -> str:
    parts = [
        f"name={t.name}",
        f"dt={t.data_type}",
        "dims=[" + ",".join(str(d) for d in t.dims) + "]",
    ]
    if t.HasField("raw_data"):
        parts.append(f"raw={len(t.raw_data)}:{fnv1a(t.raw_data):08x}")
    if len(t.float_data):
        parts.append("float=" + _packed("f", t.float_data))
    if len(t.int32_data):
        parts.append("int32=" + _packed("i", t.int32_data))
    if len(t.int64_data):
        parts.append("int64=" + _packed("q", t.int64_data))
    if len(t.double_data):
        parts.append("double=" + _packed("d", t.double_data))
    if len(t.uint64_data):
        parts.append("uint64=" + _packed("Q", t.uint64_data))
    if len(t.string_data):
        parts.append(f"str={len(t.string_data)}")
    if t.data_location:
        parts.append(f"loc={t.data_location}")
    if len(t.external_data):
        parts.append(f"ext={len(t.external_data)}")
    if t.HasField("segment"):
        parts.append("seg")
    return "tensor(" + " ".join(parts) + ")"


def _dim(d) -> str:
    w = d.WhichOneof("value")
    if w == "dim_value":
        return str(d.dim_value)
    if w == "dim_param":
        return "$" + d.dim_param
    return "?"


def type_proto(tp) -> str:
    w = tp.WhichOneof("value")
    if w == "tensor_type":
        tt = tp.tensor_type
        if tt.HasField("shape"):
            shape = "[" + ",".join(_dim(d) for d in tt.shape.dim) + "]"
        else:
            shape = "none"
        return f"tensor({tt.elem_type},{shape})"
    return {
        None: "notype",
        "sequence_type": "sequence",
        "map_type": "map",
        "optional_type": "optional",
        "sparse_tensor_type": "sparse",
        "opaque_type": "opaque",
    }[w]


def _infer_attr_type(a: AttributeProto) -> int:
    if a.type:
        return a.type
    for field, t in [
        ("f", AttributeProto.FLOAT),
        ("i", AttributeProto.INT),
        ("s", AttributeProto.STRING),
        ("t", AttributeProto.TENSOR),
        ("g", AttributeProto.GRAPH),
    ]:
        if a.HasField(field):
            return t
    for field, t in [
        ("floats", AttributeProto.FLOATS),
        ("ints", AttributeProto.INTS),
        ("strings", AttributeProto.STRINGS),
        ("tensors", AttributeProto.TENSORS),
        ("graphs", AttributeProto.GRAPHS),
    ]:
        if len(getattr(a, field)):
            return t
    return 0


def attribute(a: AttributeProto, ind: str, out: list):
    head = f"{ind}attr name={a.name} type={_infer_attr_type(a)}"
    if a.ref_attr_name:
        head += f" ref={a.ref_attr_name}"
    vals = []
    if a.HasField("f"):
        vals.append(f"f={struct.unpack('<I', struct.pack('<f', a.f))[0]:08x}")
    if a.HasField("i"):
        vals.append(f"i={a.i}")
    if a.HasField("s"):
        vals.append("s=" + _bytes_field(a.s))
    if len(a.floats):
        vals.append(
            "floats=["
            + ",".join(f"{struct.unpack('<I', struct.pack('<f', x))[0]:08x}" for x in a.floats)
            + "]"
        )
    if len(a.ints):
        vals.append("ints=[" + ",".join(str(x) for x in a.ints) + "]")
    if len(a.strings):
        vals.append("strings=[" + ",".join(_bytes_field(x) for x in a.strings) + "]")
    if a.HasField("sparse_tensor") or len(a.sparse_tensors):
        vals.append(f"sparse={int(a.HasField('sparse_tensor')) + len(a.sparse_tensors)}")
    if a.HasField("tp") or len(a.type_protos):
        vals.append(f"tp={int(a.HasField('tp')) + len(a.type_protos)}")
    out.append(head + (" " + " ".join(vals) if vals else ""))
    if a.HasField("t"):
        out.append(f"{ind}  t=" + tensor(a.t))
    for t in a.tensors:
        out.append(f"{ind}  ts=" + tensor(t))
    if a.HasField("g"):
        out.append(f"{ind}  g:")
        graph(a.g, ind + "    ", out)
    for g in a.graphs:
        out.append(f"{ind}  gs:")
        graph(g, ind + "    ", out)


def graph(g, ind: str, out: list):
    out.append(
        f"{ind}graph name={g.name} nodes={len(g.node)} init={len(g.initializer)}"
        f" sparse_init={len(g.sparse_initializer)}"
    )
    for kind, vis in [("input", g.input), ("output", g.output), ("value_info", g.value_info)]:
        for v in vis:
            out.append(f"{ind}{kind} {v.name} {type_proto(v.type)}")
    for t in g.initializer:
        out.append(f"{ind}init " + tensor(t))
    for i, n in enumerate(g.node):
        out.append(
            f"{ind}node[{i}] op={n.op_type} dom={n.domain} name={n.name}"
            f" in={','.join(n.input)} out={','.join(n.output)}"
        )
        for a in n.attribute:
            attribute(a, ind + "  ", out)


def model(m) -> str:
    out = [
        f"model ir={m.ir_version} producer={m.producer_name}/{m.producer_version}"
        f" domain={m.domain} version={m.model_version}"
        f" functions={len(m.functions)} training={len(m.training_info)}"
    ]
    for o in m.opset_import:
        out.append(f"opset {o.domain}={o.version}")
    if m.HasField("graph"):
        graph(m.graph, "", out)
    return "\n".join(out) + "\n"
