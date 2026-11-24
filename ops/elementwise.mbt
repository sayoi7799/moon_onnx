// Element-wise operators.
//
// float32 transcendental functions are evaluated in double precision and
// rounded once, so results are within 1 ulp of the correctly rounded value
// on every backend.

///|
fn map_f32(a : FixedArray[Float], f : (Float) -> Float) -> FixedArray[Float] {
  let out = FixedArray::make(a.length(), (0.0 : Float))
  for i in 0..<a.length() {
    out[i] = f(a[i])
  }
  out
}

///|
fn map_f64(
  a : FixedArray[Double],
  f : (Double) -> Double,
) -> FixedArray[Double] {
  let out = FixedArray::make(a.length(), 0.0)
  for i in 0..<a.length() {
    out[i] = f(a[i])
  }
  out
}

///|
fn map_i32(a : FixedArray[Int32], f : (Int32) -> Int32) -> FixedArray[Int32] {
  let out = FixedArray::make(a.length(), 0)
  for i in 0..<a.length() {
    out[i] = f(a[i])
  }
  out
}

///|
fn map_i64(a : FixedArray[Int64], f : (Int64) -> Int64) -> FixedArray[Int64] {
  let out = FixedArray::make(a.length(), 0L)
  for i in 0..<a.length() {
    out[i] = f(a[i])
  }
  out
}

///|
/// Applies a real function to a float tensor (computed in double).
fn map_real(t : @tensor.Tensor, f : (Double) -> Double) -> @tensor.Tensor raise {
  match t.data() {
    F32(a) =>
      tensor_new(
        t.shape(),
        F32(map_f32(a, fn(x) { Float::from_double(f(x.to_double())) })),
      )
    F64(a) => tensor_new(t.shape(), F64(map_f64(a, f)))
    _ => raise OpError("expected a float tensor, got \{t.dtype()}")
  }
}

///|
fn sigmoid(x : Double) -> Double {
  if x >= 0.0 {
    1.0 / (1.0 + @math.exp(-x))
  } else {
    let e = @math.exp(x)
    e / (1.0 + e)
  }
}

///|
fn prepare_unary(info : NodeInfo) -> Kernel raise {
  let op = info.op_type
  let allowed = match op {
    "Relu" => if info.since >= 14 { signed_types } else { float_types }
    "Neg" => signed_types
    "Abs" => numeric_types
    _ => float_types
  }
  let t = expect_type(info, 0, allowed)
  let alpha = if op == "LeakyRelu" {
    info.attrs.float("alpha", 0.01)
  } else {
    0.0
  }
  let alpha_d = alpha.to_double()
  Kernel::new([t], fn(inputs) raise {
    let x = arg(inputs, 0)
    let y = match (op, x.data()) {
      ("Relu", F32(a)) =>
        tensor_new(
          x.shape(),
          F32(map_f32(a, fn(v) { if v < 0.0 { 0.0 } else { v } })),
        )
      ("Relu", F64(a)) =>
        tensor_new(
          x.shape(),
          F64(map_f64(a, fn(v) { if v < 0.0 { 0.0 } else { v } })),
        )
      ("Relu", I32(a)) =>
        tensor_new(
          x.shape(),
          I32(map_i32(a, fn(v) { if v < 0 { 0 } else { v } })),
        )
      ("Relu", I64(a)) =>
        tensor_new(
          x.shape(),
          I64(map_i64(a, fn(v) { if v < 0L { 0L } else { v } })),
        )
      ("LeakyRelu", F32(a)) =>
        tensor_new(
          x.shape(),
          F32(map_f32(a, fn(v) { if v < 0.0 { v * alpha } else { v } })),
        )
      ("LeakyRelu", _) =>
        map_real(x, fn(v) { if v < 0.0 { v * alpha_d } else { v } })
      ("Sigmoid", _) => map_real(x, sigmoid)
      ("Tanh", _) => map_real(x, @math.tanh)
      ("Exp", _) => map_real(x, @math.exp)
      ("Log", _) => map_real(x, @math.ln)
      ("Sqrt", _) => map_real(x, fn(v) { v.sqrt() })
      ("Neg", F32(a)) => tensor_new(x.shape(), F32(map_f32(a, fn(v) { -v })))
      ("Neg", F64(a)) => tensor_new(x.shape(), F64(map_f64(a, fn(v) { -v })))
      ("Neg", I32(a)) => tensor_new(x.shape(), I32(map_i32(a, fn(v) { -v })))
      ("Neg", I64(a)) => tensor_new(x.shape(), I64(map_i64(a, fn(v) { -v })))
      ("Abs", F32(a)) =>
        tensor_new(x.shape(), F32(map_f32(a, fn(v) { v.abs() })))
      ("Abs", F64(a)) =>
        tensor_new(x.shape(), F64(map_f64(a, fn(v) { v.abs() })))
      ("Abs", I32(a)) =>
        tensor_new(
          x.shape(),
          I32(map_i32(a, fn(v) { if v < 0 { -v } else { v } })),
        )
      ("Abs", I64(a)) =>
        tensor_new(
          x.shape(),
          I64(map_i64(a, fn(v) { if v < 0L { -v } else { v } })),
        )
      ("Abs", U8(_)) => x
      _ => raise OpError("\{op}: unsupported input type \{x.dtype()}")
    }
    [y]
  })
}

///|
/// Reads a scalar (or one-element) tensor as a Double.
fn scalar_value(t : @tensor.Tensor, what : String) -> Double raise {
  if t.size() != 1 {
    raise OpError("\{what} must be a scalar, got shape \{shape_str(t.shape())}")
  }
  match t.data() {
    F32(a) => a[0].to_double()
    F64(a) => a[0]
    I32(a) => a[0].to_double()
    I64(a) => a[0].to_double()
    U8(a) => a[0].to_int().to_double()
    Bool(a) => if a[0] { 1.0 } else { 0.0 }
  }
}

///|
/// Clip: y = min(max(x, lo), hi). When lo > hi every value becomes hi,
/// as the spec requires. NaN inputs stay NaN.
fn prepare_clip(info : NodeInfo) -> Kernel raise {
  let t = if info.since >= 12 {
    expect_type(info, 0, numeric_types)
  } else {
    expect_type(info, 0, float_types)
  }
  for i in 1..<info.present.length() {
    if info.present[i] {
      same_type(info, i, t)
    }
  }
  let attr_min = if info.since < 11 {
    Some(info.attrs.float("min", @float.min_value).to_double())
  } else {
    None
  }
  let attr_max = if info.since < 11 {
    Some(info.attrs.float("max", @float.max_value).to_double())
  } else {
    None
  }
  Kernel::new([t], fn(inputs) raise {
    let x = arg(inputs, 0)
    let lo_t = opt_arg(inputs, 1)
    let hi_t = opt_arg(inputs, 2)
    let lo = match (attr_min, lo_t) {
      (Some(v), _) => Some(v)
      (None, Some(m)) => Some(scalar_value(m, "Clip min"))
      _ => None
    }
    let hi = match (attr_max, hi_t) {
      (Some(v), _) => Some(v)
      (None, Some(m)) => Some(scalar_value(m, "Clip max"))
      _ => None
    }
    let y : @tensor.Data = match x.data() {
      F32(a) => {
        let l : Float = match lo {
          Some(v) => Float::from_double(v)
          None => @float.neg_infinity
        }
        let h : Float = match hi {
          Some(v) => Float::from_double(v)
          None => @float.infinity
        }
        F32(
          map_f32(a, fn(v) {
            let v1 = if v < l { l } else { v }
            if v1 > h {
              h
            } else {
              v1
            }
          }),
        )
      }
      F64(a) => {
        let l = lo.unwrap_or(@double.neg_infinity)
        let h = hi.unwrap_or(@double.infinity)
        F64(
          map_f64(a, fn(v) {
            let v1 = if v < l { l } else { v }
            if v1 > h {
              h
            } else {
              v1
            }
          }),
        )
      }
      I32(a) => {
        let l = lo.map(fn(v) { v.to_int() }).unwrap_or(-2147483648)
        let h = hi.map(fn(v) { v.to_int() }).unwrap_or(2147483647)
        I32(map_i32(a, fn(v) { if v < l { l } else if v > h { h } else { v } }))
      }
      I64(a) => {
        // read int64 bounds exactly (not through Double)
        let l = match lo_t {
          Some(m) => m.to_i64_array()[0]
          None => -9223372036854775808L
        }
        let h = match hi_t {
          Some(m) => m.to_i64_array()[0]
          None => 9223372036854775807L
        }
        I64(map_i64(a, fn(v) { if v < l { l } else if v > h { h } else { v } }))
      }
      U8(a) => {
        let l = lo.map(fn(v) { v.to_int() }).unwrap_or(0)
        let h = hi.map(fn(v) { v.to_int() }).unwrap_or(255)
        U8(
          FixedArray::makei(a.length(), fn(i) {
            let v = a[i].to_int()
            let c = if v < l { l } else if v > h { h } else { v }
            c.to_byte()
          }),
        )
      }
      Bool(_) => raise OpError("Clip: bool input is not supported")
    }
    [tensor_new(x.shape(), y)]
  })
}

///|
/// Generic broadcasting binary loop.
fn[T, U, V] broadcast_binary(
  a_shape : FixedArray[Int32],
  a : FixedArray[T],
  b_shape : FixedArray[Int32],
  b : FixedArray[U],
  zero : V,
  f : (T, U) -> V raise,
) -> (FixedArray[Int32], FixedArray[V]) raise {
  let out_shape = @tensor.broadcast_shapes(a_shape, b_shape) catch {
    @tensor.TensorError(m) => raise OpError(m)
  }
  let n = numel(out_shape)
  let out = FixedArray::make(n, zero)
  if @tensor.shapes_equal(a_shape, b_shape) {
    for i in 0..<n {
      out[i] = f(a[i], b[i])
    }
  } else if b.length() == 1 {
    let bv = b[0]
    for i in 0..<n {
      out[i] = f(a[i], bv)
    }
  } else if a.length() == 1 {
    let av = a[0]
    for i in 0..<n {
      out[i] = f(av, b[i])
    }
  } else {
    let sa = @tensor.broadcast_strides(a_shape, out_shape)
    let sb = @tensor.broadcast_strides(b_shape, out_shape)
    let mut err : Error? = None
    @tensor.for_each_broadcast2(out_shape, sa, sb, fn(o, ia, ib) {
      if err is None {
        out[o] = f(a[ia], b[ib]) catch {
          e => {
            err = Some(e)
            out[o]
          }
        }
      }
    })
    if err is Some(e) {
      raise e
    }
  }
  (out_shape, out)
}

///|
fn prepare_arith(info : NodeInfo) -> Kernel raise {
  let t = expect_type(info, 0, numeric_types)
  same_type(info, 1, t)
  let op = info.op_type
  // Add/Sub/Mul/Div-6: no implicit broadcasting; with broadcast=1 the second
  // operand is broadcast to the first, aligned at `axis` (default: suffix).
  let legacy = info.since < 7
  let legacy_broadcast = legacy && info.attrs.flag("broadcast", false)
  let legacy_axis = if legacy { info.attrs.int_opt("axis") } else { None }
  Kernel::new([t], fn(inputs) raise {
    let x = arg(inputs, 0)
    let y = arg(inputs, 1)
    let xs = x.shape()
    let ys = if legacy {
      legacy_rhs_shape(op, xs, y.shape(), legacy_broadcast, legacy_axis)
    } else {
      y.shape()
    }
    let (shape, data) : (FixedArray[Int32], @tensor.Data) = match
      (x.data(), y.data()) {
      (F32(a), F32(b)) => {
        let f : (Float, Float) -> Float raise = match op {
          "Add" => fn(p, q) { p + q }
          "Sub" => fn(p, q) { p - q }
          "Mul" => fn(p, q) { p * q }
          _ => fn(p, q) { p / q }
        }
        let (s, o) = broadcast_binary(xs, a, ys, b, (0.0 : Float), f)
        (s, F32(o))
      }
      (F64(a), F64(b)) => {
        let f : (Double, Double) -> Double raise = match op {
          "Add" => fn(p, q) { p + q }
          "Sub" => fn(p, q) { p - q }
          "Mul" => fn(p, q) { p * q }
          _ => fn(p, q) { p / q }
        }
        let (s, o) = broadcast_binary(xs, a, ys, b, 0.0, f)
        (s, F64(o))
      }
      (I32(a), I32(b)) => {
        let f : (Int32, Int32) -> Int32 raise = match op {
          "Add" => fn(p, q) { p + q }
          "Sub" => fn(p, q) { p - q }
          "Mul" => fn(p, q) { p * q }
          _ =>
            fn(p, q) raise {
              if q == 0 {
                raise OpError("Div: integer division by zero")
              }
              p / q
            }
        }
        let (s, o) = broadcast_binary(xs, a, ys, b, 0, f)
        (s, I32(o))
      }
      (I64(a), I64(b)) => {
        let f : (Int64, Int64) -> Int64 raise = match op {
          "Add" => fn(p, q) { p + q }
          "Sub" => fn(p, q) { p - q }
          "Mul" => fn(p, q) { p * q }
          _ =>
            fn(p, q) raise {
              if q == 0L {
                raise OpError("Div: integer division by zero")
              }
              p / q
            }
        }
        let (s, o) = broadcast_binary(xs, a, ys, b, 0L, f)
        (s, I64(o))
      }
      (U8(a), U8(b)) => {
        let f : (Byte, Byte) -> Byte raise = match op {
          "Add" => fn(p, q) { ((p.to_int() + q.to_int()) & 0xFF).to_byte() }
          "Sub" => fn(p, q) { ((p.to_int() - q.to_int()) & 0xFF).to_byte() }
          "Mul" => fn(p, q) { ((p.to_int() * q.to_int()) & 0xFF).to_byte() }
          _ =>
            fn(p, q) raise {
              if q.to_int() == 0 {
                raise OpError("Div: integer division by zero")
              }
              (p.to_int() / q.to_int()).to_byte()
            }
        }
        let (s, o) = broadcast_binary(xs, a, ys, b, b'\x00', f)
        (s, U8(o))
      }
      _ =>
        raise OpError(
          "\{op}: mismatched input types \{x.dtype()} and \{y.dtype()}",
        )
    }
    [tensor_new(shape, data)]
  })
}

///|
/// Shape to use for the second operand of a version-6 binary op so that
/// NumPy broadcasting reproduces the legacy semantics.
fn legacy_rhs_shape(
  op : String,
  a : FixedArray[Int32],
  b : FixedArray[Int32],
  broadcast : Bool,
  axis : Int64?,
) -> FixedArray[Int32] raise {
  if !broadcast {
    if !@tensor.shapes_equal(a, b) {
      raise OpError(
        "\{op}-6 without broadcast=1 needs equal shapes, got \{shape_str(a)} and \{shape_str(b)}",
      )
    }
    return b
  }
  let ra = a.length()
  let rb = b.length()
  let start = match axis {
    Some(ax) => norm_axis(ax, ra)
    None => ra - rb
  }
  if start < 0 || start + rb > ra {
    raise OpError(
      "\{op}-6: cannot broadcast \{shape_str(b)} to \{shape_str(a)}",
    )
  }
  let out = FixedArray::make(ra, 1)
  for i in 0..<rb {
    if b[i] != a[start + i] && b[i] != 1 {
      raise OpError(
        "\{op}-6: cannot broadcast \{shape_str(b)} to \{shape_str(a)}",
      )
    }
    out[start + i] = b[i]
  }
  out
}

///|
/// Reads element i of any numeric buffer as Double.
fn as_double_fn(d : @tensor.Data) -> (Int32) -> Double {
  match d {
    F32(a) => fn(i) { a[i].to_double() }
    F64(a) => fn(i) { a[i] }
    I32(a) => fn(i) { a[i].to_double() }
    I64(a) => fn(i) { a[i].to_double() }
    U8(a) => fn(i) { a[i].to_int().to_double() }
    Bool(a) => fn(i) { if a[i] { 1.0 } else { 0.0 } }
  }
}

///|
fn int_pow(base : Int64, exp : Int64) -> Int64 {
  if exp < 0L {
    // pow(b, -k) truncated towards zero
    return match base {
      1L => 1L
      -1L => if exp % 2L == 0L { 1L } else { -1L }
      _ => 0L
    }
  }
  let mut result = 1L
  let mut b = base
  let mut e = exp
  while e > 0L {
    if e % 2L == 1L {
      result = result * b
    }
    b = b * b
    e = e / 2L
  }
  result
}

///|
/// Pow: output has the base type. Mixed base/exponent types are allowed
/// from Pow-12.
fn prepare_pow(info : NodeInfo) -> Kernel raise {
  let tb = expect_type(info, 0, signed_types)
  let te = if info.since >= 12 {
    expect_type(info, 1, signed_types)
  } else {
    same_type(info, 1, tb)
    tb
  }
  ignore(te)
  Kernel::new([tb], fn(inputs) raise {
    let x = arg(inputs, 0)
    let y = arg(inputs, 1)
    let out_shape = @tensor.broadcast_shapes(x.shape(), y.shape()) catch {
      @tensor.TensorError(m) => raise OpError(m)
    }
    let n = numel(out_shape)
    let sx = @tensor.broadcast_strides(x.shape(), out_shape)
    let sy = @tensor.broadcast_strides(y.shape(), out_shape)
    let ey = as_double_fn(y.data())
    let y_is_int = y.dtype().is_integer()
    let y_ints = if y_is_int { Some(y.to_i64_array()) } else { None }
    let data : @tensor.Data = match x.data() {
      F32(a) => {
        let out = FixedArray::make(n, (0.0 : Float))
        @tensor.for_each_broadcast2(out_shape, sx, sy, fn(o, i, j) {
          out[o] = Float::from_double(@math.pow(a[i].to_double(), ey(j)))
        })
        F32(out)
      }
      F64(a) => {
        let out = FixedArray::make(n, 0.0)
        @tensor.for_each_broadcast2(out_shape, sx, sy, fn(o, i, j) {
          out[o] = @math.pow(a[i], ey(j))
        })
        F64(out)
      }
      I32(a) => {
        let out = FixedArray::make(n, 0)
        @tensor.for_each_broadcast2(out_shape, sx, sy, fn(o, i, j) {
          out[o] = match y_ints {
            Some(e) => int_pow(a[i].to_int64(), e[j]).to_int()
            None => @math.pow(a[i].to_double(), ey(j)).to_int()
          }
        })
        I32(out)
      }
      I64(a) => {
        let out = FixedArray::make(n, 0L)
        @tensor.for_each_broadcast2(out_shape, sx, sy, fn(o, i, j) {
          out[o] = match y_ints {
            Some(e) => int_pow(a[i], e[j])
            None => @math.pow(a[i].to_double(), ey(j)).to_int64()
          }
        })
        I64(out)
      }
      _ => raise OpError("Pow: unsupported base type \{x.dtype()}")
    }
    [tensor_new(out_shape, data)]
  })
}

///|
fn prepare_minmax(info : NodeInfo) -> Kernel raise {
  let allowed = if info.since >= 12 { numeric_types } else { float_types }
  let t = expect_type(info, 0, allowed)
  for i in 1..<info.present.length() {
    same_type(info, i, t)
  }
  let is_max = info.op_type == "Max"
  Kernel::new([t], fn(inputs) raise {
    let mut acc = arg(inputs, 0)
    for i in 1..<inputs.length() {
      let b = arg(inputs, i)
      let (shape, data) : (FixedArray[Int32], @tensor.Data) = match
        (acc.data(), b.data()) {
        (F32(x), F32(y)) => {
          let (s, o) = broadcast_binary(
            acc.shape(),
            x,
            b.shape(),
            y,
            (0.0 : Float),
            fn(p, q) {
              if p.is_nan() || q.is_nan() {
                @float.not_a_number
              } else if is_max == (p > q) {
                p
              } else {
                q
              }
            },
          )
          (s, F32(o))
        }
        (F64(x), F64(y)) => {
          let (s, o) = broadcast_binary(acc.shape(), x, b.shape(), y, 0.0, fn(
            p,
            q,
          ) {
            if p.is_nan() || q.is_nan() {
              @double.not_a_number
            } else if is_max == (p > q) {
              p
            } else {
              q
            }
          })
          (s, F64(o))
        }
        (I32(x), I32(y)) => {
          let (s, o) = broadcast_binary(acc.shape(), x, b.shape(), y, 0, fn(
            p,
            q,
          ) {
            if is_max == (p > q) {
              p
            } else {
              q
            }
          })
          (s, I32(o))
        }
        (I64(x), I64(y)) => {
          let (s, o) = broadcast_binary(acc.shape(), x, b.shape(), y, 0L, fn(
            p,
            q,
          ) {
            if is_max == (p > q) {
              p
            } else {
              q
            }
          })
          (s, I64(o))
        }
        (U8(x), U8(y)) => {
          let (s, o) = broadcast_binary(acc.shape(), x, b.shape(), y, b'\x00', fn(
            p,
            q,
          ) {
            if is_max == (p > q) {
              p
            } else {
              q
            }
          })
          (s, U8(o))
        }
        _ => raise OpError("\{info.op_type}: mismatched input types")
      }
      acc = tensor_new(shape, data)
    }
    [acc]
  })
}

///|
fn prepare_identity(info : NodeInfo) -> Kernel raise {
  let t = expect_type(info, 0, all_types)
  Kernel::new([t], fn(inputs) raise { [arg(inputs, 0)] })
}

///|
/// Dropout in inference: output = input, mask = all true.
/// Training mode (Dropout-12+ `training_mode` input) is deterministic only
/// for ratio 0; any other ratio is rejected when the node runs.
fn prepare_dropout(info : NodeInfo) -> Kernel raise {
  let t = expect_type(info, 0, float_types)
  let mask_type : @tensor.DType = if info.since >= 10 { Bool } else { t }
  if info.since >= 12 {
    if info.has_input(1) {
      ignore(expect_type(info, 1, float_types))
    }
    if info.has_input(2) {
      ignore(expect_type(info, 2, [Bool]))
    }
  }
  Kernel::new([t, mask_type], fn(inputs) raise {
    let x = arg(inputs, 0)
    let training = match opt_arg(inputs, 2) {
      Some(m) => scalar_value(m, "Dropout training_mode") != 0.0
      None => false
    }
    if training {
      let ratio = match opt_arg(inputs, 1) {
        Some(r) => scalar_value(r, "Dropout ratio")
        None => 0.5
      }
      if ratio != 0.0 {
        raise OpError(
          "Dropout in training mode with ratio \{ratio} is random and not supported",
        )
      }
    }
    let n = x.size()
    let mask : @tensor.Data = match mask_type {
      Bool => Bool(FixedArray::make(n, true))
      F64 => F64(FixedArray::make(n, 1.0))
      _ => F32(FixedArray::make(n, 1.0))
    }
    [x, tensor_new(x.shape(), mask)]
  })
}

///|
fn cast_data(d : @tensor.Data, to : @tensor.DType) -> @tensor.Data {
  if d.dtype() == to {
    return d
  }
  let n = d.length()
  match (d, to) {
    (_, Bool) => {
      let get = as_double_fn(d)
      Bool(FixedArray::makei(n, fn(i) { get(i) != 0.0 }))
    }
    (I64(a), I32) => I32(FixedArray::makei(n, fn(i) { a[i].to_int() }))
    (I64(a), U8) => U8(FixedArray::makei(n, fn(i) { (a[i] & 0xFFL).to_byte() }))
    (I64(a), F32) =>
      F32(FixedArray::makei(n, fn(i) { Float::from_double(a[i].to_double()) }))
    (I64(a), F64) => F64(FixedArray::makei(n, fn(i) { a[i].to_double() }))
    (_, I64) => {
      let get = int_or_trunc_fn(d)
      I64(FixedArray::makei(n, get))
    }
    (_, I32) => {
      let get = int_or_trunc_fn(d)
      I32(FixedArray::makei(n, fn(i) { get(i).to_int() }))
    }
    (_, U8) => {
      let get = int_or_trunc_fn(d)
      U8(FixedArray::makei(n, fn(i) { (get(i) & 0xFFL).to_byte() }))
    }
    (_, F32) => {
      let get = as_double_fn(d)
      F32(FixedArray::makei(n, fn(i) { Float::from_double(get(i)) }))
    }
    (_, F64) => {
      let get = as_double_fn(d)
      F64(FixedArray::makei(n, get))
    }
  }
}

///|
/// Element as Int64: integers exactly, floats truncated towards zero.
fn int_or_trunc_fn(d : @tensor.Data) -> (Int32) -> Int64 {
  match d {
    F32(a) => fn(i) { a[i].to_double().to_int64() }
    F64(a) => fn(i) { a[i].to_int64() }
    I32(a) => fn(i) { a[i].to_int64() }
    I64(a) => fn(i) { a[i] }
    U8(a) => fn(i) { a[i].to_int64() }
    Bool(a) => fn(i) { if a[i] { 1L } else { 0L } }
  }
}

///|
fn prepare_cast(info : NodeInfo) -> Kernel raise {
  ignore(expect_type(info, 0, all_types))
  let to_code = info.attrs.int("to", 0L).to_int()
  guard @tensor.DType::from_onnx_code(to_code) is Some(to) else {
    raise OpError("Cast to \{@proto.data_type_name(to_code)} is not supported")
  }
  if info.since >= 24 && info.attrs.has("round_mode") {
    let mode = info.attrs.string("round_mode", "up")
    if mode != "up" {
      // round_mode only affects casts to float8e8m0
      ignore(mode)
    }
  }
  Kernel::new([to], fn(inputs) raise {
    let x = arg(inputs, 0)
    [tensor_new(x.shape(), cast_data(x.data(), to))]
  })
}
