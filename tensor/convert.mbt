// TensorProto -> Tensor.
//
// ONNX stores tensor contents in one of two ways:
//   * raw_data: little-endian, densely packed bytes (bool = 1 byte);
//   * a typed repeated field: float_data (FLOAT), double_data (DOUBLE),
//     int64_data (INT64), and int32_data for the narrow types
//     (INT32, UINT8, BOOL, ...).
// External data, segments, strings and sparse tensors are rejected.
// Typed-field conversion follows onnx.numpy_helper.to_array (C casts).

///|
pub fn from_proto(t : @proto.TensorProto) -> Tensor raise TensorError {
  let what = if t.name == "" { "tensor" } else { "tensor '\{t.name}'" }
  if t.data_location == 1 || t.external_data_count > 0 {
    raise TensorError("\{what}: external data is not supported")
  }
  if t.has_segment {
    raise TensorError("\{what}: segmented tensors are not supported")
  }
  guard DType::from_onnx_code(t.data_type) is Some(dtype) else {
    raise TensorError(
      "\{what}: data type \{@proto.data_type_name(t.data_type)} is not supported",
    )
  }
  let shape = FixedArray::make(t.dims.length(), 0)
  for i, d in t.dims {
    if d < 0L || d > max_elements {
      raise TensorError("\{what}: invalid dimension \{d}")
    }
    shape[i] = d.to_int()
  }
  let n = numel(shape) catch {
    TensorError(m) => raise TensorError("\{what}: \{m}")
  }
  let data = match t.raw_data {
    Some(raw) => from_raw(what, dtype, n, raw)
    None => from_typed(what, dtype, n, t)
  }
  { shape, data, }
}

///|
/// Decodes a serialized TensorProto (an ONNX test-data `.pb` file).
pub fn from_proto_bytes(bytes : BytesView) -> Tensor raise Error {
  from_proto(@proto.decode_tensor(bytes))
}

///|
fn elem_size(dtype : DType) -> Int32 {
  match dtype {
    F32 | I32 => 4
    F64 | I64 => 8
    U8 | Bool => 1
  }
}

///|
fn le32(b : BytesView, p : Int32) -> UInt32 {
  b[p].to_uint() |
  (b[p + 1].to_uint() << 8) |
  (b[p + 2].to_uint() << 16) |
  (b[p + 3].to_uint() << 24)
}

///|
fn le64(b : BytesView, p : Int32) -> UInt64 {
  le32(b, p).to_uint64() | (le32(b, p + 4).to_uint64() << 32)
}

///|
fn from_raw(
  what : String,
  dtype : DType,
  n : Int32,
  raw : BytesView,
) -> Data raise TensorError {
  let need = n.to_int64() * elem_size(dtype).to_int64()
  if raw.length().to_int64() != need {
    raise TensorError(
      "\{what}: raw_data has \{raw.length()} bytes, \{dtype} x \{n} needs \{need}",
    )
  }
  match dtype {
    F32 =>
      F32(
        FixedArray::makei(n, fn(i) {
          Float::reinterpret_from_uint(le32(raw, 4 * i))
        }),
      )
    F64 =>
      F64(
        FixedArray::makei(n, fn(i) { le64(raw, 8 * i).reinterpret_as_double() }),
      )
    I32 =>
      I32(FixedArray::makei(n, fn(i) { le32(raw, 4 * i).reinterpret_as_int() }))
    I64 =>
      I64(
        FixedArray::makei(n, fn(i) { le64(raw, 8 * i).reinterpret_as_int64() }),
      )
    U8 => U8(FixedArray::makei(n, fn(i) { raw[i] }))
    Bool => Bool(FixedArray::makei(n, fn(i) { raw[i] != b'\x00' }))
  }
}

///|
fn from_typed(
  what : String,
  dtype : DType,
  n : Int32,
  t : @proto.TensorProto,
) -> Data raise TensorError {
  let (field, count) = match dtype {
    F32 => ("float_data", t.float_data.length())
    F64 => ("double_data", t.double_data.length())
    I64 => ("int64_data", t.int64_data.length())
    I32 | U8 | Bool => ("int32_data", t.int32_data.length())
  }
  if count != n {
    raise TensorError("\{what}: \{field} has \{count} values, shape needs \{n}")
  }
  match dtype {
    F32 => F32(FixedArray::from_array(t.float_data))
    F64 => F64(FixedArray::from_array(t.double_data))
    I64 => I64(FixedArray::from_array(t.int64_data))
    I32 => I32(FixedArray::from_array(t.int32_data))
    U8 => U8(FixedArray::makei(n, fn(i) { (t.int32_data[i] & 0xFF).to_byte() }))
    Bool => Bool(FixedArray::makei(n, fn(i) { t.int32_data[i] != 0 }))
  }
}
