///|
fn fnv1a_bytes(n : Int32, width : Int32, get : (Int32) -> UInt64) -> UInt32 {
  let mut h : UInt32 = 0x811C9DC5
  for i in 0..<n {
    let v = get(i)
    for k in 0..<width {
      h = h ^ ((v >> (8 * k)) & 0xFF).to_uint()
      h = h * 0x01000193
    }
  }
  h
}

///|
fn hex8(u : UInt32) -> String {
  let digits = "0123456789abcdef"
  let buf = StringBuilder()
  for i = 7; i >= 0; i = i - 1 {
    buf.write_char(
      digits.get_char(((u >> (i * 4)) & 0xF).reinterpret_as_int()).unwrap(),
    )
  }
  buf.to_string()
}

///|
/// Same format as `describe` in scripts/gen_tensor_fixtures.py.
fn describe(t : Tensor) -> String {
  let n = t.size()
  let h = match t.data {
    F32(a) =>
      fnv1a_bytes(n, 4, fn(i) { a[i].reinterpret_as_uint().to_uint64() })
    F64(a) => fnv1a_bytes(n, 8, fn(i) { a[i].reinterpret_as_uint64() })
    I32(a) =>
      fnv1a_bytes(n, 4, fn(i) { a[i].reinterpret_as_uint().to_uint64() })
    I64(a) => fnv1a_bytes(n, 8, fn(i) { a[i].reinterpret_as_uint64() })
    U8(a) => fnv1a_bytes(n, 1, fn(i) { a[i].to_uint64() })
    Bool(a) => fnv1a_bytes(n, 1, fn(i) { if a[i] { 1 } else { 0 } })
  }
  "\{t.dtype()}\{shape_to_string(t.shape)} n=\{n} h=\{hex8(h)}"
}

///|
fn b64(s : String) -> Bytes {
  @base64.decode(s, ignore_whitespace=true) catch {
    _ => abort("bad base64 fixture")
  }
}

///|
test "TensorProto conversion matches onnx.numpy_helper.to_array" {
  for c in conversions {
    let (name, data, expected) = c
    let t = from_proto_bytes(b64(data)[:]) catch { e => fail("\{name}: \{e}") }
    assert_eq(describe(t), expected, msg=name)
  }
}

///|
test "unsupported or malformed TensorProtos are rejected" {
  for r in rejections {
    let (name, data, expected) = r
    let msg = try {
      ignore(from_proto_bytes(b64(data)[:]))
      fail("\{name}: accepted")
    } catch {
      e => Show::to_string(e)
    }
    assert_true(
      msg.contains(expected),
      msg="\{name}: '\{msg}' lacks '\{expected}'",
    )
  }
}

///|
test "broadcasting matches NumPy" {
  for c in broadcasts {
    let (sa, sb, expected, values) = c
    let a = FixedArray::from_array(sa)
    let b = FixedArray::from_array(sb)
    let ctx = shape_to_string(a) + " vs " + shape_to_string(b)
    let out = broadcast_shapes(a, b) catch {
      _ => {
        assert_eq(expected, "error", msg=ctx)
        continue
      }
    }
    assert_eq(shape_to_string(out), expected, msg=ctx)
    // Element mapping: out = a_index + 1000 * b_index, as computed by NumPy.
    let n = numel(out)
    let got = FixedArray::make(n, -1)
    let as_ = broadcast_strides(a, out)
    let bs = broadcast_strides(b, out)
    for_each_broadcast2(out, as_, bs, fn(o, ia, ib) { got[o] = ia + 1000 * ib })
    assert_eq(got, FixedArray::from_array(values), msg=ctx)
  }
}

///|
test "multi-way broadcast and helpers" {
  let s = broadcast_all([[2, 1, 3], [4, 1], [1]])
  assert_eq(shape_to_string(s), "[2,4,3]")
  assert_eq(strides([2, 3, 4]), ([12, 4, 1] : FixedArray[Int32]))
  assert_eq(strides([]).length(), 0)
  assert_eq(normalize_axis(-1L, 4), 3)
  let r : Result[Int32, TensorError] = Ok(normalize_axis(4L, 4)) catch {
    e => Err(e)
  }
  assert_true(r is Err(_))
  let big : Result[Int32, TensorError] = Ok(numel([65536, 65536])) catch {
    e => Err(e)
  }
  assert_true(big is Err(_))
}

///|
test "tensor constructors check sizes" {
  let t = Tensor::f32([2, 2], [1.0, 2.0, 3.0, 4.0])
  assert_eq(t.to_string_brief(), "float32[2,2]")
  let bad : Result[Tensor, TensorError] = Ok(Tensor::f32([3], [1.0])) catch {
    e => Err(e)
  }
  assert_true(bad is Err(_))
  let r = t.reshaped([4])
  assert_eq(r.rank(), 1)
  assert_eq(
    Tensor::i32([2], [1, 2]).to_i64_array(),
    ([1L, 2L] : FixedArray[Int64]),
  )
}
