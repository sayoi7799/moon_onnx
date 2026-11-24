// Dense tensors.
//
// A tensor is a shape plus one flat, row-major (C order) buffer whose element
// type is fixed by the `Data` variant. Tensors are always contiguous: ops
// that change layout (Transpose, Slice, ...) produce a new buffer. Strides
// are derived from the shape whenever an op needs them.

///|
/// Element types supported by the MVP.
#derive(Eq, Debug)
pub(all) enum DType {
  F32
  F64
  I32
  I64
  U8
  Bool
}

///|
pub impl Show for DType with fn output(self, logger) {
  logger.write_string(self.name())
}

///|
/// ONNX TensorProto.DataType name.
pub fn DType::name(self : DType) -> String {
  match self {
    F32 => "float32"
    F64 => "float64"
    I32 => "int32"
    I64 => "int64"
    U8 => "uint8"
    Bool => "bool"
  }
}

///|
/// The TensorProto.DataType code.
pub fn DType::onnx_code(self : DType) -> Int32 {
  match self {
    F32 => 1
    U8 => 2
    I32 => 6
    I64 => 7
    Bool => 9
    F64 => 11
  }
}

///|
/// Maps a TensorProto.DataType code to a supported dtype.
pub fn DType::from_onnx_code(code : Int32) -> DType? {
  match code {
    1 => Some(F32)
    2 => Some(U8)
    6 => Some(I32)
    7 => Some(I64)
    9 => Some(Bool)
    11 => Some(F64)
    _ => None
  }
}

///|
pub fn DType::is_float(self : DType) -> Bool {
  self is (F32 | F64)
}

///|
pub fn DType::is_integer(self : DType) -> Bool {
  self is (I32 | I64 | U8)
}

///|
pub(all) enum Data {
  F32(FixedArray[Float])
  F64(FixedArray[Double])
  I32(FixedArray[Int32])
  I64(FixedArray[Int64])
  U8(FixedArray[Byte])
  Bool(FixedArray[Bool])
}

///|
pub fn Data::dtype(self : Data) -> DType {
  match self {
    F32(_) => F32
    F64(_) => F64
    I32(_) => I32
    I64(_) => I64
    U8(_) => U8
    Bool(_) => Bool
  }
}

///|
pub fn Data::length(self : Data) -> Int32 {
  match self {
    F32(a) => a.length()
    F64(a) => a.length()
    I32(a) => a.length()
    I64(a) => a.length()
    U8(a) => a.length()
    Bool(a) => a.length()
  }
}

///|
/// Allocates a zero-filled buffer.
pub fn Data::zeros(dtype : DType, n : Int32) -> Data {
  match dtype {
    F32 => F32(FixedArray::make(n, 0.0))
    F64 => F64(FixedArray::make(n, 0.0))
    I32 => I32(FixedArray::make(n, 0))
    I64 => I64(FixedArray::make(n, 0L))
    U8 => U8(FixedArray::make(n, b'\x00'))
    Bool => Bool(FixedArray::make(n, false))
  }
}

///|
pub(all) suberror TensorError {
  TensorError(String)
}

///|
pub impl Show for TensorError with fn output(self, logger) {
  let TensorError(msg) = self
  logger.write_string(msg)
}

///|
pub struct Tensor {
  shape : FixedArray[Int32]
  data : Data
}

///|
/// Creates a tensor, checking that the buffer length matches the shape.
pub fn Tensor::new(
  shape : FixedArray[Int32],
  data : Data,
) -> Tensor raise TensorError {
  let n = numel(shape)
  if n != data.length() {
    raise TensorError(
      "shape \{shape_to_string(shape)} needs \{n} elements, buffer has \{data.length()}",
    )
  }
  { shape, data, }
}

///|
pub fn Tensor::zeros(
  dtype : DType,
  shape : FixedArray[Int32],
) -> Tensor raise TensorError {
  { shape, data: Data::zeros(dtype, numel(shape)), }
}

///|
pub fn Tensor::f32(
  shape : Array[Int32],
  values : Array[Float],
) -> Tensor raise TensorError {
  Tensor::new(
    FixedArray::from_array(shape),
    F32(FixedArray::from_array(values)),
  )
}

///|
pub fn Tensor::f64(
  shape : Array[Int32],
  values : Array[Double],
) -> Tensor raise TensorError {
  Tensor::new(
    FixedArray::from_array(shape),
    F64(FixedArray::from_array(values)),
  )
}

///|
pub fn Tensor::i32(
  shape : Array[Int32],
  values : Array[Int32],
) -> Tensor raise TensorError {
  Tensor::new(
    FixedArray::from_array(shape),
    I32(FixedArray::from_array(values)),
  )
}

///|
pub fn Tensor::i64(
  shape : Array[Int32],
  values : Array[Int64],
) -> Tensor raise TensorError {
  Tensor::new(
    FixedArray::from_array(shape),
    I64(FixedArray::from_array(values)),
  )
}

///|
pub fn Tensor::u8(
  shape : Array[Int32],
  values : Array[Byte],
) -> Tensor raise TensorError {
  Tensor::new(FixedArray::from_array(shape), U8(FixedArray::from_array(values)))
}

///|
pub fn Tensor::bool(
  shape : Array[Int32],
  values : Array[Bool],
) -> Tensor raise TensorError {
  Tensor::new(
    FixedArray::from_array(shape),
    Bool(FixedArray::from_array(values)),
  )
}

///|
/// A rank-0 int64 tensor.
pub fn Tensor::scalar_i64(v : Int64) -> Tensor {
  { shape: [], data: I64([v]), }
}

///|
/// A rank-0 float32 tensor.
pub fn Tensor::scalar_f32(v : Float) -> Tensor {
  { shape: [], data: F32([v]), }
}

///|
pub fn Tensor::dtype(self : Tensor) -> DType {
  self.data.dtype()
}

///|
pub fn Tensor::shape(self : Tensor) -> FixedArray[Int32] {
  self.shape
}

///|
pub fn Tensor::data(self : Tensor) -> Data {
  self.data
}

///|
pub fn Tensor::rank(self : Tensor) -> Int32 {
  self.shape.length()
}

///|
pub fn Tensor::size(self : Tensor) -> Int32 {
  self.data.length()
}

///|
/// Same data, new shape (element count must match). The buffer is shared.
pub fn Tensor::reshaped(
  self : Tensor,
  shape : FixedArray[Int32],
) -> Tensor raise TensorError {
  Tensor::new(shape, self.data)
}

///|
pub fn Tensor::to_f32_array(
  self : Tensor,
) -> FixedArray[Float] raise TensorError {
  match self.data {
    F32(a) => a
    _ => raise TensorError("expected float32 tensor, got \{self.dtype()}")
  }
}

///|
/// Reads an integer tensor (int32 or int64) as Int64 values; used for
/// shape-like inputs (Reshape shape, Slice starts, ...).
pub fn Tensor::to_i64_array(
  self : Tensor,
) -> FixedArray[Int64] raise TensorError {
  match self.data {
    I64(a) => a
    I32(a) => FixedArray::makei(a.length(), fn(i) { a[i].to_int64() })
    _ =>
      raise TensorError(
        "expected an int32 or int64 tensor, got \{self.dtype()}",
      )
  }
}

///|
pub fn Tensor::to_string_brief(self : Tensor) -> String {
  "\{self.dtype()}\{shape_to_string(self.shape)}"
}
