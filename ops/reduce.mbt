// Softmax / LogSoftmax and the reduction operators.

///|
/// Softmax and LogSoftmax. Before opset 13 the input is coerced to 2-D at
/// `axis` (rows = prod(shape[:axis]), cols = prod(shape[axis:])); from 13 the
/// operation runs along the single dimension `axis`.
/// Computed in double precision with the max subtracted for stability.
fn prepare_softmax(info : NodeInfo) -> Kernel raise {
  let t = expect_type(info, 0, float_types)
  let is_log = info.op_type == "LogSoftmax"
  let axis = info.attrs.int("axis", if info.since >= 13 { -1L } else { 1L })
  let legacy = info.since < 13
  Kernel::new([t], fn(inputs) raise {
    let x = arg(inputs, 0)
    let xs = x.shape()
    let r = xs.length()
    let a = if r == 0 && (axis == 0L || axis == -1L) {
      0
    } else {
      norm_axis(axis, r)
    }
    // outer x len x inner decomposition
    let (outer, len, inner) = if r == 0 {
      (1, 1, 1)
    } else if legacy {
      let mut o = 1
      for i in 0..<a {
        o *= xs[i]
      }
      let total = x.size()
      (o, if o == 0 { 0 } else { total / o }, 1)
    } else {
      let mut o = 1
      for i in 0..<a {
        o *= xs[i]
      }
      let mut inn = 1
      for i in (a + 1)..<r {
        inn *= xs[i]
      }
      (o, xs[a], inn)
    }
    let get = as_double_fn(x.data())
    let out = FixedArray::make(x.size(), 0.0)
    for o in 0..<outer {
      for i in 0..<inner {
        let base = o * len * inner + i
        let mut m = @double.neg_infinity
        for j in 0..<len {
          let v = get(base + j * inner)
          if v > m || v.is_nan() {
            m = v
          }
        }
        let mut s = 0.0
        for j in 0..<len {
          s = s + @math.exp(get(base + j * inner) - m)
        }
        let ls = @math.ln(s)
        for j in 0..<len {
          let v = get(base + j * inner) - m
          out[base + j * inner] = if is_log { v - ls } else { @math.exp(v) / s }
        }
      }
    }
    let data : @tensor.Data = match t {
      F32 =>
        F32(
          FixedArray::makei(out.length(), fn(i) { Float::from_double(out[i]) }),
        )
      _ => F64(out)
    }
    [tensor_new(xs, data)]
  })
}

///|
/// Which axes to reduce: from the `axes` attribute (older versions) or the
/// optional `axes` input (ReduceSum-13+, ReduceMean/Max-18+).
fn reduce_axes_from_input(info : NodeInfo) -> Bool {
  match info.op_type {
    "ReduceSum" => info.since >= 13
    _ => info.since >= 18
  }
}

///|
fn prepare_reduce(info : NodeInfo) -> Kernel raise {
  let op = info.op_type
  let allowed : Array[@tensor.DType] = match op {
    "ReduceMean" => [F32, F64, I32, I64]
    "ReduceSum" => numeric_types
    _ => if info.since >= 20 { all_types } else { numeric_types }
  }
  let t = expect_type(info, 0, allowed)
  let keepdims = info.attrs.flag("keepdims", true)
  let from_input = reduce_axes_from_input(info)
  let noop_empty = if from_input {
    info.attrs.flag("noop_with_empty_axes", false)
  } else {
    false
  }
  let attr_axes = if from_input { None } else { info.attrs.ints("axes") }
  if from_input && info.has_input(1) {
    ignore(expect_type(info, 1, [I64]))
  }
  Kernel::new([t], fn(inputs) raise {
    let x = arg(inputs, 0)
    let axes_list : Array[Int64]? = match attr_axes {
      Some(a) => Some(a)
      None =>
        match opt_arg(inputs, 1) {
          Some(at) if from_input => {
            let v = at.to_i64_array()
            Some(Array::makei(v.length(), fn(i) { v[i] }))
          }
          _ => None
        }
    }
    let r = x.rank()
    let reduce = FixedArray::make(r, false)
    match axes_list {
      None =>
        for i in 0..<r {
          reduce[i] = !noop_empty
        }
      Some([]) =>
        for i in 0..<r {
          reduce[i] = !noop_empty
        }
      Some(list) =>
        for ax in list {
          let a = norm_axis(ax, r)
          if reduce[a] {
            raise OpError("\{op}: axis \{ax} given twice")
          }
          reduce[a] = true
        }
    }
    [reduce_tensor(op, x, reduce, keepdims)]
  })
}

///|
fn reduce_tensor(
  op : String,
  x : @tensor.Tensor,
  reduce : FixedArray[Bool],
  keepdims : Bool,
) -> @tensor.Tensor raise {
  let xs = x.shape()
  let r = xs.length()
  let kept = FixedArray::makei(r, fn(i) { if reduce[i] { 1 } else { xs[i] } })
  let out_n = numel(kept)
  // map each input element to its output slot
  let ostr = @tensor.broadcast_strides(kept, xs)
  let count = {
    let mut c = 1
    for i in 0..<r {
      if reduce[i] {
        c *= xs[i]
      }
    }
    c
  }
  let final_shape = if keepdims {
    kept
  } else {
    let dims = []
    for i in 0..<r {
      if !reduce[i] {
        dims.push(xs[i])
      }
    }
    FixedArray::from_array(dims)
  }
  let data : @tensor.Data = match (op, x.data()) {
    ("ReduceMax", Bool(a)) => {
      let out = FixedArray::make(out_n, false)
      @tensor.for_each_broadcast1(xs, ostr, fn(i, o) { out[o] = out[o] || a[i] })
      Bool(out)
    }
    ("ReduceMax", I32(a)) => {
      let out = FixedArray::make(out_n, -2147483648)
      @tensor.for_each_broadcast1(xs, ostr, fn(i, o) {
        if a[i] > out[o] {
          out[o] = a[i]
        }
      })
      I32(out)
    }
    ("ReduceMax", I64(a)) => {
      let out = FixedArray::make(out_n, -9223372036854775808L)
      @tensor.for_each_broadcast1(xs, ostr, fn(i, o) {
        if a[i] > out[o] {
          out[o] = a[i]
        }
      })
      I64(out)
    }
    ("ReduceMax", U8(a)) => {
      let out = FixedArray::make(out_n, b'\x00')
      @tensor.for_each_broadcast1(xs, ostr, fn(i, o) {
        if a[i] > out[o] {
          out[o] = a[i]
        }
      })
      U8(out)
    }
    ("ReduceMax", d) => {
      let get = as_double_fn(d)
      let out = FixedArray::make(out_n, @double.neg_infinity)
      @tensor.for_each_broadcast1(xs, ostr, fn(i, o) {
        let v = get(i)
        if v > out[o] || v.is_nan() {
          out[o] = v
        }
      })
      real_data(d.dtype(), out)
    }
    (_, I64(a)) => {
      let out = FixedArray::make(out_n, 0L)
      @tensor.for_each_broadcast1(xs, ostr, fn(i, o) { out[o] = out[o] + a[i] })
      if op == "ReduceMean" && count > 0 {
        for i in 0..<out_n {
          out[i] = out[i] / count.to_int64()
        }
      }
      I64(out)
    }
    (_, I32(a)) => {
      let out = FixedArray::make(out_n, 0)
      @tensor.for_each_broadcast1(xs, ostr, fn(i, o) { out[o] = out[o] + a[i] })
      if op == "ReduceMean" && count > 0 {
        for i in 0..<out_n {
          out[i] = out[i] / count
        }
      }
      I32(out)
    }
    (_, U8(a)) => {
      let out = FixedArray::make(out_n, 0)
      @tensor.for_each_broadcast1(xs, ostr, fn(i, o) {
        out[o] = out[o] + a[i].to_int()
      })
      U8(FixedArray::makei(out_n, fn(i) { (out[i] & 0xFF).to_byte() }))
    }
    (_, d) => {
      let get = as_double_fn(d)
      let out = FixedArray::make(out_n, 0.0)
      @tensor.for_each_broadcast1(xs, ostr, fn(i, o) {
        out[o] = out[o] + get(i)
      })
      if op == "ReduceMean" {
        for i in 0..<out_n {
          out[i] = out[i] / count.to_double()
        }
      }
      real_data(d.dtype(), out)
    }
  }
  tensor_new(final_shape, data)
}

///|
fn real_data(t : @tensor.DType, v : FixedArray[Double]) -> @tensor.Data {
  match t {
    F32 =>
      F32(FixedArray::makei(v.length(), fn(i) { Float::from_double(v[i]) }))
    _ => F64(v)
  }
}

///|
fn prepare_argmax(info : NodeInfo) -> Kernel raise {
  ignore(expect_type(info, 0, numeric_types))
  let axis = info.attrs.int("axis", 0L)
  let keepdims = info.attrs.flag("keepdims", true)
  let last = if info.since >= 12 {
    info.attrs.flag("select_last_index", false)
  } else {
    false
  }
  Kernel::new([I64], fn(inputs) raise {
    let x = arg(inputs, 0)
    let xs = x.shape()
    let r = xs.length()
    let a = norm_axis(axis, r)
    let len = xs[a]
    if len == 0 {
      raise OpError("ArgMax over an empty axis")
    }
    let mut outer = 1
    for i in 0..<a {
      outer *= xs[i]
    }
    let mut inner = 1
    for i in (a + 1)..<r {
      inner *= xs[i]
    }
    let get = as_double_fn(x.data())
    let ints = match x.data() {
      I64(v) => Some(v)
      _ => None
    }
    let out = FixedArray::make(outer * inner, 0L)
    for o in 0..<outer {
      for i in 0..<inner {
        let base = o * len * inner + i
        let mut best = 0
        for j in 1..<len {
          let better = match ints {
            Some(v) => {
              let p = v[base + j * inner]
              let q = v[base + best * inner]
              if last {
                p >= q
              } else {
                p > q
              }
            }
            None => {
              let p = get(base + j * inner)
              let q = get(base + best * inner)
              if last {
                p >= q
              } else {
                p > q
              }
            }
          }
          if better {
            best = j
          }
        }
        out[o * inner + i] = best.to_int64()
      }
    }
    let dims = []
    for i in 0..<r {
      if i == a {
        if keepdims {
          dims.push(1)
        }
      } else {
        dims.push(xs[i])
      }
    }
    [tensor_new(FixedArray::from_array(dims), I64(out))]
  })
}
