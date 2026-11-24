// Canonical text summary of decoded messages; must match scripts/proto_summary.py.

///|
fn fnv1a(bytes : BytesView) -> UInt32 {
  let mut h : UInt32 = 0x811C9DC5
  for b in bytes {
    h = h ^ b.to_uint()
    h = h * 0x01000193
  }
  h
}

///|
fn hex_digit(n : Int32) -> Char {
  "0123456789abcdef".get_char(n).unwrap()
}

///|
fn hex8(u : UInt32) -> String {
  let buf = StringBuilder()
  for i = 7; i >= 0; i = i - 1 {
    buf.write_char(hex_digit(((u >> (i * 4)) & 0xF).reinterpret_as_int()))
  }
  buf.to_string()
}

///|
fn hex_bytes(b : BytesView) -> String {
  let buf = StringBuilder()
  for x in b {
    buf.write_char(hex_digit(x.to_int() >> 4))
    buf.write_char(hex_digit(x.to_int() & 0xF))
  }
  buf.to_string()
}

///|
fn bytes_field(b : Bytes) -> String {
  if b.length() <= 32 {
    "x" + hex_bytes(b[:])
  } else {
    "\{b.length()}:\{hex8(fnv1a(b[:]))}"
  }
}

///|
fn packed(n : Int32, bytes : Bytes) -> String {
  "\{n}:\{hex8(fnv1a(bytes[:]))}"
}

///|
fn le_bytes(width : Int32, n : Int32, get : (Int32) -> UInt64) -> Bytes {
  Bytes::makei(width * n, fn(j) {
    ((get(j / width) >> (8 * (j % width))) & 0xFF).to_byte()
  })
}

///|
fn tensor_summary(t : TensorProto) -> String {
  let parts = [
    "name=\{t.name}",
    "dt=\{t.data_type}",
    "dims=[" + t.dims.map(fn(d) { d.to_string() }).join(",") + "]",
  ]
  if t.raw_data is Some(raw) {
    parts.push("raw=\{raw.length()}:\{hex8(fnv1a(raw))}")
  }
  if t.float_data.length() > 0 {
    let n = t.float_data.length()
    parts.push(
      "float=" +
      packed(
        n,
        le_bytes(4, n, fn(i) {
          t.float_data[i].reinterpret_as_uint().to_uint64()
        }),
      ),
    )
  }
  if t.int32_data.length() > 0 {
    let n = t.int32_data.length()
    parts.push(
      "int32=" +
      packed(
        n,
        le_bytes(4, n, fn(i) {
          t.int32_data[i].reinterpret_as_uint().to_uint64()
        }),
      ),
    )
  }
  if t.int64_data.length() > 0 {
    let n = t.int64_data.length()
    parts.push(
      "int64=" +
      packed(
        n,
        le_bytes(8, n, fn(i) { t.int64_data[i].reinterpret_as_uint64() }),
      ),
    )
  }
  if t.double_data.length() > 0 {
    let n = t.double_data.length()
    parts.push(
      "double=" +
      packed(
        n,
        le_bytes(8, n, fn(i) { t.double_data[i].reinterpret_as_uint64() }),
      ),
    )
  }
  if t.uint64_data.length() > 0 {
    let n = t.uint64_data.length()
    parts.push(
      "uint64=" + packed(n, le_bytes(8, n, fn(i) { t.uint64_data[i] })),
    )
  }
  if t.string_data_count > 0 {
    parts.push("str=\{t.string_data_count}")
  }
  if t.data_location != 0 {
    parts.push("loc=\{t.data_location}")
  }
  if t.external_data_count > 0 {
    parts.push("ext=\{t.external_data_count}")
  }
  if t.has_segment {
    parts.push("seg")
  }
  "tensor(" + parts.join(" ") + ")"
}

///|
fn type_summary(tp : TypeProto) -> String {
  match tp {
    TensorType(elem_type~, shape~) => {
      let s = match shape {
        None => "none"
        Some(dims) =>
          "[" +
          dims
          .map(fn(d) {
            match d {
              DimValue(v) => v.to_string()
              DimParam(p) => "$" + p
              DimUnknown => "?"
            }
          })
          .join(",") +
          "]"
      }
      "tensor(\{elem_type},\{s})"
    }
    SequenceType => "sequence"
    MapType => "map"
    OptionalType => "optional"
    SparseTensorType => "sparse"
    OpaqueType => "opaque"
    NoType => "notype"
  }
}

///|
fn attr_summary(a : AttributeProto, ind : String, out : Array[String]) -> Unit {
  let mut head = "\{ind}attr name=\{a.name} type=\{a.attr_type}"
  if a.ref_attr_name != "" {
    head = head + " ref=\{a.ref_attr_name}"
  }
  let vals = []
  if a.f is Some(f) {
    vals.push("f=" + hex8(f.reinterpret_as_uint()))
  }
  if a.i is Some(i) {
    vals.push("i=\{i}")
  }
  if a.s is Some(s) {
    vals.push("s=" + bytes_field(s))
  }
  if a.floats.length() > 0 {
    vals.push(
      "floats=[" +
      a.floats.map(fn(x) { hex8(x.reinterpret_as_uint()) }).join(",") +
      "]",
    )
  }
  if a.ints.length() > 0 {
    vals.push("ints=[" + a.ints.map(fn(x) { x.to_string() }).join(",") + "]")
  }
  if a.strings.length() > 0 {
    vals.push("strings=[" + a.strings.map(bytes_field).join(",") + "]")
  }
  if a.sparse_tensor_count > 0 {
    vals.push("sparse=\{a.sparse_tensor_count}")
  }
  if a.type_proto_count > 0 {
    vals.push("tp=\{a.type_proto_count}")
  }
  out.push(if vals.length() > 0 { head + " " + vals.join(" ") } else { head })
  if a.t is Some(t) {
    out.push("\{ind}  t=" + tensor_summary(t))
  }
  for t in a.tensors {
    out.push("\{ind}  ts=" + tensor_summary(t))
  }
  if a.g is Some(g) {
    out.push("\{ind}  g:")
    graph_summary(g, ind + "    ", out)
  }
  for g in a.graphs {
    out.push("\{ind}  gs:")
    graph_summary(g, ind + "    ", out)
  }
}

///|
fn graph_summary(g : GraphProto, ind : String, out : Array[String]) -> Unit {
  out.push(
    "\{ind}graph name=\{g.name} nodes=\{g.nodes.length()} init=\{g.initializers.length()} sparse_init=\{g.sparse_initializer_count}",
  )
  for
    kv in [
      ("input", g.inputs),
      ("output", g.outputs),
      ("value_info", g.value_infos),
    ] {
    for v in kv.1 {
      out.push("\{ind}\{kv.0} \{v.name} \{type_summary(v.value_type)}")
    }
  }
  for t in g.initializers {
    out.push("\{ind}init " + tensor_summary(t))
  }
  for i, n in g.nodes {
    out.push(
      "\{ind}node[\{i}] op=\{n.op_type} dom=\{n.domain} name=\{n.name} in=\{n.inputs.join(",")} out=\{n.outputs.join(",")}",
    )
    for a in n.attributes {
      attr_summary(a, ind + "  ", out)
    }
  }
}

///|
fn model_summary(m : ModelProto) -> String {
  let out = [
    "model ir=\{m.ir_version} producer=\{m.producer_name}/\{m.producer_version} domain=\{m.domain} version=\{m.model_version} functions=\{m.function_count} training=\{m.training_info_count}",
  ]
  for o in m.opset_imports {
    out.push("opset \{o.domain}=\{o.version}")
  }
  if m.graph is Some(g) {
    graph_summary(g, "", out)
  }
  out.join("\n") + "\n"
}

///|
fn decode_b64(s : String) -> Bytes {
  @base64.decode(s, ignore_whitespace=true) catch {
    _ => abort("bad base64 fixture")
  }
}

///|
fn summarize(kind : String, bytes : BytesView) -> String raise DecodeError {
  if kind == "model" {
    model_summary(decode_model(bytes))
  } else {
    tensor_summary(decode_tensor(bytes)) + "\n"
  }
}

///|
test "decoded fixtures match the onnx Python package" {
  for fx in fixtures {
    let (name, kind, b64, expected) = fx
    let bytes = decode_b64(b64)
    let got = summarize(kind, bytes[:]) catch {
      e => fail("\{name}: unexpected error \{e}")
    }
    if got != expected {
      fail(
        "\{name}: summary mismatch\n--- got\n\{got}\n--- expected\n\{expected}",
      )
    }
  }
}

///|
test "truncated inputs fail exactly when protobuf rejects them" {
  let by_name : Map[String, (String, Bytes)] = Map([])
  for fx in fixtures {
    by_name[fx.0] = (fx.1, decode_b64(fx.2))
  }
  let mut rejected = 0
  for tr in truncations {
    let (name, cut, expected) = tr
    let (kind, bytes) = by_name[name]
    let prefix = bytes[0:cut]
    let result : Result[String, DecodeError] = Ok(summarize(kind, prefix)) catch {
      e => Err(e)
    }
    match (result, expected) {
      (Err(DecodeError(offset~, ..)), "") => {
        assert_true(
          offset >= 0 && offset <= cut,
          msg="\{name}@\{cut}: offset \{offset}",
        )
        rejected += 1
      }
      (Ok(got), exp) if exp != "" =>
        assert_eq(got, exp, msg="\{name}@\{cut}: accepted prefix differs")
      (Ok(_), _) =>
        fail("\{name}@\{cut}: protobuf rejects this prefix, we accepted it")
      (Err(e), _) =>
        fail("\{name}@\{cut}: protobuf accepts this prefix, we raised \{e}")
    }
  }
  assert_true(rejected > 100)
}

///|
fn from_hex(s : String) -> Bytes {
  let digit = fn(c : Char) -> Int32 {
    let i = c.to_int()
    if i >= '0'.to_int() && i <= '9'.to_int() {
      i - '0'.to_int()
    } else {
      i - 'a'.to_int() + 10
    }
  }
  let chars = s.to_array()
  Bytes::makei(chars.length() / 2, fn(i) {
    (digit(chars[2 * i]) * 16 + digit(chars[2 * i + 1])).to_byte()
  })
}

///|
test "malformed messages are rejected with a position" {
  for m in malformed {
    let (name, hex) = m
    let bytes = from_hex(hex)
    let result : Result[ModelProto, DecodeError] = Ok(decode_model(bytes[:])) catch {
      e => Err(e)
    }
    guard result is Err(DecodeError(offset~, path~, ..)) else {
      fail("\{name}: accepted malformed input")
    }
    assert_true(offset >= 0 && offset <= bytes.length(), msg=name)
    assert_true(path.has_prefix("model"), msg=name)
  }
}

///|
test "error message names field path and byte offset" {
  // graph (field 7) -> node (field 1) whose length runs past the end
  let bytes = from_hex("3a040a096162")
  let msg = try {
    ignore(decode_model(bytes[:]))
    ""
  } catch {
    e => Show::to_string(e)
  }
  inspect(
    msg,
    content="protobuf decode error at byte 3 (model.graph): length-delimited field of 9 bytes exceeds remaining 2 bytes",
  )
}
