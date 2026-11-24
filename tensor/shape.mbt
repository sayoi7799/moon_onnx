// Shapes, strides and NumPy-style broadcasting.

///|
/// Largest element count we allow for a single tensor (2^31 - 1).
let max_elements : Int64 = 2147483647L

///|
/// Number of elements; raises if the product overflows a 32-bit index.
pub fn numel(shape : FixedArray[Int32]) -> Int32 raise TensorError {
  let mut n = 1L
  for d in shape {
    if d < 0 {
      raise TensorError("negative dimension in shape \{shape_to_string(shape)}")
    }
    n = n * d.to_int64()
    if n > max_elements {
      raise TensorError(
        "shape \{shape_to_string(shape)} has more than \{max_elements} elements",
      )
    }
  }
  n.to_int()
}

///|
pub fn shape_to_string(shape : FixedArray[Int32]) -> String {
  let parts = []
  for d in shape {
    parts.push(d.to_string())
  }
  "[" + parts.join(",") + "]"
}

///|
/// Row-major strides, in elements.
pub fn strides(shape : FixedArray[Int32]) -> FixedArray[Int32] {
  let r = shape.length()
  let s = FixedArray::make(r, 1)
  for i = r - 2; i >= 0; i = i - 1 {
    s[i] = s[i + 1] * shape[i + 1]
  }
  s
}

///|
/// NumPy broadcasting of two shapes: align on the right; each pair of
/// dimensions must be equal or one of them must be 1.
pub fn broadcast_shapes(
  a : FixedArray[Int32],
  b : FixedArray[Int32],
) -> FixedArray[Int32] raise TensorError {
  let r = if a.length() > b.length() { a.length() } else { b.length() }
  let out = FixedArray::make(r, 1)
  for i in 0..<r {
    let da = if i < r - a.length() { 1 } else { a[i - (r - a.length())] }
    let db = if i < r - b.length() { 1 } else { b[i - (r - b.length())] }
    out[i] = if da == db {
      da
    } else if da == 1 {
      db
    } else if db == 1 {
      da
    } else {
      raise TensorError(
        "shapes \{shape_to_string(a)} and \{shape_to_string(b)} cannot be broadcast",
      )
    }
  }
  out
}

///|
/// Broadcast of any number of shapes (Min, Max, Sum, ...).
pub fn broadcast_all(
  shapes : Array[FixedArray[Int32]],
) -> FixedArray[Int32] raise TensorError {
  let mut out : FixedArray[Int32] = []
  for s in shapes {
    out = broadcast_shapes(out, s)
  }
  out
}

///|
/// Element strides of `shape` when it is broadcast to `out_shape`:
/// broadcast (size-1 or missing) dimensions get stride 0.
pub fn broadcast_strides(
  shape : FixedArray[Int32],
  out_shape : FixedArray[Int32],
) -> FixedArray[Int32] {
  let r = out_shape.length()
  let offset = r - shape.length()
  let own = strides(shape)
  FixedArray::makei(r, fn(i) {
    if i < offset || shape[i - offset] == 1 {
      0
    } else {
      own[i - offset]
    }
  })
}

///|
pub fn shapes_equal(a : FixedArray[Int32], b : FixedArray[Int32]) -> Bool {
  if a.length() != b.length() {
    return false
  }
  for i in 0..<a.length() {
    if a[i] != b[i] {
      return false
    }
  }
  true
}

///|
/// Walks every element of `out_shape` in row-major order, tracking the flat
/// offsets into two broadcast inputs. `f(out_index, a_offset, b_offset)`.
///
/// The innermost dimension is a tight loop; outer dimensions advance an
/// odometer, so no division is needed per element.
pub fn for_each_broadcast2(
  out_shape : FixedArray[Int32],
  sa : FixedArray[Int32],
  sb : FixedArray[Int32],
  f : (Int32, Int32, Int32) -> Unit,
) -> Unit {
  let r = out_shape.length()
  let total = out_shape.fold(init=1, fn(acc, d) { acc * d })
  if total == 0 {
    return
  }
  if r == 0 {
    f(0, 0, 0)
    return
  }
  let inner = out_shape[r - 1]
  let ia = sa[r - 1]
  let ib = sb[r - 1]
  let counter = FixedArray::make(r, 0)
  let mut oa = 0
  let mut ob = 0
  let mut o = 0
  while o < total {
    let mut pa = oa
    let mut pb = ob
    for k in 0..<inner {
      f(o + k, pa, pb)
      pa += ia
      pb += ib
    }
    o += inner
    // advance the odometer over dimensions [0, r-1)
    let mut d = r - 2
    while d >= 0 {
      counter[d] += 1
      oa += sa[d]
      ob += sb[d]
      if counter[d] < out_shape[d] {
        break
      }
      oa -= sa[d] * out_shape[d]
      ob -= sb[d] * out_shape[d]
      counter[d] = 0
      d -= 1
    }
  }
}

///|
/// Single-input variant: offsets into one broadcast input.
pub fn for_each_broadcast1(
  out_shape : FixedArray[Int32],
  sa : FixedArray[Int32],
  f : (Int32, Int32) -> Unit,
) -> Unit {
  for_each_broadcast2(out_shape, sa, sa, fn(o, a, _) { f(o, a) })
}

///|
/// Normalizes a possibly negative axis into [0, rank).
pub fn normalize_axis(axis : Int64, rank : Int32) -> Int32 raise TensorError {
  let a = if axis < 0L { axis + rank.to_int64() } else { axis }
  if a < 0L || a >= rank.to_int64() {
    raise TensorError("axis \{axis} is out of range for rank \{rank}")
  }
  a.to_int()
}
