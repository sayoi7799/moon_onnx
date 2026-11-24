// Shape manipulation and data movement operators. They work for every
// supported dtype: most compute, for each output element, the flat index of
// its source element and then gather with `take`.

///|
/// out[i] = src[idx[i]]; idx[i] < 0 selects `fill[0]`.
fn take(
  src : @tensor.Data,
  idx : FixedArray[Int32],
  fill : @tensor.Data?,
) -> @tensor.Data {
  let n = idx.length()
  match src {
    F32(a) => {
      let f = match fill {
        Some(F32(v)) => v[0]
        _ => 0.0
      }
      F32(
        FixedArray::makei(n, fn(i) { if idx[i] < 0 { f } else { a[idx[i]] } }),
      )
    }
    F64(a) => {
      let f = match fill {
        Some(F64(v)) => v[0]
        _ => 0.0
      }
      F64(
        FixedArray::makei(n, fn(i) { if idx[i] < 0 { f } else { a[idx[i]] } }),
      )
    }
    I32(a) => {
      let f = match fill {
        Some(I32(v)) => v[0]
        _ => 0
      }
      I32(
        FixedArray::makei(n, fn(i) { if idx[i] < 0 { f } else { a[idx[i]] } }),
      )
    }
    I64(a) => {
      let f = match fill {
        Some(I64(v)) => v[0]
        _ => 0L
      }
      I64(
        FixedArray::makei(n, fn(i) { if idx[i] < 0 { f } else { a[idx[i]] } }),
      )
    }
    U8(a) => {
      let f = match fill {
        Some(U8(v)) => v[0]
        _ => b'\x00'
      }
      U8(FixedArray::makei(n, fn(i) { if idx[i] < 0 { f } else { a[idx[i]] } }))
    }
    Bool(a) => {
      let f = match fill {
        Some(Bool(v)) => v[0]
        _ => false
      }
      Bool(
        FixedArray::makei(n, fn(i) { if idx[i] < 0 { f } else { a[idx[i]] } }),
      )
    }
  }
}

///|
fn i64s_to_ints(
  v : FixedArray[Int64],
  what : String,
) -> FixedArray[Int32] raise {
  let out = FixedArray::make(v.length(), 0)
  for i in 0..<v.length() {
    let x = v[i]
    if x > 2147483647L || x < -2147483648L {
      raise OpError("\{what}: value \{x} is out of range")
    }
    out[i] = x.to_int()
  }
  out
}

///|
/// Shape-like input (int64 1-D tensor).
fn shape_input(t : @tensor.Tensor, what : String) -> FixedArray[Int64] raise {
  if t.rank() != 1 {
    raise OpError(
      "\{what} must be a 1-D tensor, got shape \{shape_str(t.shape())}",
    )
  }
  t.to_i64_array()
}

///|
fn reshaped(
  t : @tensor.Tensor,
  shape : FixedArray[Int32],
) -> @tensor.Tensor raise {
  t.reshaped(shape) catch {
    @tensor.TensorError(m) => raise OpError(m)
  }
}

// ------------------------------------------------------------ Reshape

///|
fn prepare_reshape(info : NodeInfo) -> Kernel raise {
  let t = expect_type(info, 0, all_types)
  ignore(expect_type(info, 1, [I64]))
  let allowzero = if info.since >= 14 {
    info.attrs.flag("allowzero", false)
  } else {
    false
  }
  Kernel::new([t], fn(inputs) raise {
    let x = arg(inputs, 0)
    let spec = shape_input(arg(inputs, 1), "Reshape shape")
    let xs = x.shape()
    let out = FixedArray::make(spec.length(), 0)
    let mut infer = -1
    let mut known = 1L
    for i in 0..<spec.length() {
      let d = spec[i]
      if d == -1L {
        if infer >= 0 {
          raise OpError("Reshape: more than one -1 in shape")
        }
        infer = i
      } else if d == 0L && !allowzero {
        if i >= xs.length() {
          raise OpError(
            "Reshape: 0 at position \{i} but input has rank \{xs.length()}",
          )
        }
        out[i] = xs[i]
        known = known * out[i].to_int64()
      } else if d < 0L {
        raise OpError("Reshape: invalid dimension \{d}")
      } else {
        out[i] = d.to_int()
        known = known * d
      }
    }
    if infer >= 0 {
      if allowzero && known == 0L {
        raise OpError("Reshape: cannot infer -1 together with a 0 dimension")
      }
      if known == 0L || x.size().to_int64() % known != 0L {
        raise OpError(
          "Reshape: cannot reshape \{shape_str(xs)} into a shape with \{known} x ? elements",
        )
      }
      out[infer] = (x.size().to_int64() / known).to_int()
    }
    if numel(out) != x.size() {
      raise OpError(
        "Reshape: cannot reshape \{shape_str(xs)} (\{x.size()} elements) into \{shape_str(out)}",
      )
    }
    [reshaped(x, out)]
  })
}

///|
fn prepare_flatten(info : NodeInfo) -> Kernel raise {
  let t = expect_type(info, 0, all_types)
  let axis = info.attrs.int("axis", 1L)
  if info.since < 11 && axis < 0L {
    raise OpError("Flatten-\{info.since}: negative axis is not allowed")
  }
  Kernel::new([t], fn(inputs) raise {
    let x = arg(inputs, 0)
    let r = x.rank()
    let a = if axis < 0L { axis + r.to_int64() } else { axis }
    if a < 0L || a > r.to_int64() {
      raise OpError("Flatten: axis \{axis} out of range for rank \{r}")
    }
    let a = a.to_int()
    let mut outer = 1
    for i in 0..<a {
      outer *= x.shape()[i]
    }
    let inner = if outer == 0 { 0 } else { x.size() / outer }
    let inner = if outer == 0 {
      let mut p = 1
      for i in a..<r {
        p *= x.shape()[i]
      }
      p
    } else {
      inner
    }
    [reshaped(x, [outer, inner])]
  })
}

///|
fn prepare_identity_like(info : NodeInfo) -> @tensor.DType raise {
  expect_type(info, 0, all_types)
}

///|
fn prepare_squeeze(info : NodeInfo) -> Kernel raise {
  let t = prepare_identity_like(info)
  let attr_axes = if info.since < 13 { info.attrs.ints("axes") } else { None }
  if info.since >= 13 && info.has_input(1) {
    ignore(expect_type(info, 1, [I64]))
  }
  Kernel::new([t], fn(inputs) raise {
    let x = arg(inputs, 0)
    let xs = x.shape()
    let r = xs.length()
    let axes = match attr_axes {
      Some(a) => Some(FixedArray::from_array(a))
      None =>
        match opt_arg(inputs, 1) {
          Some(a) => Some(a.to_i64_array())
          None => None
        }
    }
    let drop = FixedArray::make(r, false)
    match axes {
      None =>
        for i in 0..<r {
          drop[i] = xs[i] == 1
        }
      Some(list) =>
        for ax in list {
          let a = norm_axis(ax, r)
          if xs[a] != 1 {
            raise OpError(
              "Squeeze: dimension \{a} of \{shape_str(xs)} is not 1",
            )
          }
          drop[a] = true
        }
    }
    let dims = []
    for i in 0..<r {
      if !drop[i] {
        dims.push(xs[i])
      }
    }
    [reshaped(x, FixedArray::from_array(dims))]
  })
}

///|
fn prepare_unsqueeze(info : NodeInfo) -> Kernel raise {
  let t = prepare_identity_like(info)
  let attr_axes = if info.since < 13 { info.attrs.ints("axes") } else { None }
  if info.since >= 13 {
    ignore(expect_type(info, 1, [I64]))
  }
  Kernel::new([t], fn(inputs) raise {
    let x = arg(inputs, 0)
    let axes = match attr_axes {
      Some(a) => FixedArray::from_array(a)
      None => arg(inputs, 1).to_i64_array()
    }
    let xs = x.shape()
    let r = xs.length() + axes.length()
    let insert = FixedArray::make(r, false)
    for ax in axes {
      let a = norm_axis(ax, r)
      if insert[a] {
        raise OpError("Unsqueeze: axis \{ax} given twice")
      }
      insert[a] = true
    }
    let out = FixedArray::make(r, 1)
    let mut j = 0
    for i in 0..<r {
      if !insert[i] {
        out[i] = xs[j]
        j += 1
      }
    }
    [reshaped(x, out)]
  })
}

// ---------------------------------------------------------- Transpose

///|
fn prepare_transpose(info : NodeInfo) -> Kernel raise {
  let t = expect_type(info, 0, all_types)
  let perm_attr = info.attrs.ints("perm")
  Kernel::new([t], fn(inputs) raise {
    let x = arg(inputs, 0)
    let xs = x.shape()
    let r = xs.length()
    let perm = match perm_attr {
      Some(p) => {
        if p.length() != r {
          raise OpError(
            "Transpose: perm has \{p.length()} entries for rank \{r}",
          )
        }
        let seen = FixedArray::make(r, false)
        let out = FixedArray::make(r, 0)
        for i in 0..<r {
          let a = norm_axis(p[i], r)
          if seen[a] {
            raise OpError("Transpose: perm is not a permutation")
          }
          seen[a] = true
          out[i] = a
        }
        out
      }
      None => FixedArray::makei(r, fn(i) { r - 1 - i })
    }
    let out_shape = FixedArray::makei(r, fn(i) { xs[perm[i]] })
    let in_str = @tensor.strides(xs)
    let src_str = FixedArray::makei(r, fn(i) { in_str[perm[i]] })
    let n = x.size()
    let idx = FixedArray::make(n, 0)
    @tensor.for_each_broadcast1(out_shape, src_str, fn(o, s) { idx[o] = s })
    [tensor_new(out_shape, take(x.data(), idx, None))]
  })
}

// ------------------------------------------------------------- Concat

///|
fn prepare_concat(info : NodeInfo) -> Kernel raise {
  let t = expect_type(info, 0, all_types)
  for i in 1..<info.present.length() {
    same_type(info, i, t)
  }
  let axis = info.attrs.int("axis", 1L)
  Kernel::new([t], fn(inputs) raise {
    let parts = []
    for i in 0..<inputs.length() {
      parts.push(arg(inputs, i))
    }
    let first = parts[0]
    let r = first.rank()
    let a = norm_axis(axis, r)
    let out_shape = FixedArray::makei(r, fn(i) { first.shape()[i] })
    out_shape[a] = 0
    for p in parts {
      if p.rank() != r {
        raise OpError("Concat: inputs have different ranks")
      }
      for i in 0..<r {
        if i != a && p.shape()[i] != first.shape()[i] {
          raise OpError(
            "Concat: shapes \{shape_str(first.shape())} and \{shape_str(p.shape())} differ outside axis \{a}",
          )
        }
      }
      out_shape[a] += p.shape()[a]
    }
    let mut outer = 1
    for i in 0..<a {
      outer *= out_shape[i]
    }
    let mut inner = 1
    for i in (a + 1)..<r {
      inner *= out_shape[i]
    }
    let n = numel(out_shape)
    let idx = FixedArray::make(n, 0)
    // concatenate by gathering from a virtual buffer of all inputs back to back
    let mut base = 0
    let mut col = 0
    let row = out_shape[a] * inner
    for p in parts {
      let chunk = p.shape()[a] * inner
      for o in 0..<outer {
        for j in 0..<chunk {
          idx[o * row + col + j] = base + o * chunk + j
        }
      }
      base += p.size()
      col += chunk
    }
    let joined = concat_data(parts.map(fn(p) { p.data() }))
    [tensor_new(out_shape, take(joined, idx, None))]
  })
}

///|
fn concat_data(parts : Array[@tensor.Data]) -> @tensor.Data {
  let total = parts.fold(init=0, fn(s, p) { s + p.length() })
  let out = @tensor.Data::zeros(parts[0].dtype(), total)
  let mut off = 0
  for p in parts {
    match (out, p) {
      (F32(o), F32(a)) =>
        FixedArray::blit_to(a, o, len=a.length(), dst_offset=off)
      (F64(o), F64(a)) =>
        FixedArray::blit_to(a, o, len=a.length(), dst_offset=off)
      (I32(o), I32(a)) =>
        FixedArray::blit_to(a, o, len=a.length(), dst_offset=off)
      (I64(o), I64(a)) =>
        FixedArray::blit_to(a, o, len=a.length(), dst_offset=off)
      (U8(o), U8(a)) =>
        FixedArray::blit_to(a, o, len=a.length(), dst_offset=off)
      (Bool(o), Bool(a)) =>
        FixedArray::blit_to(a, o, len=a.length(), dst_offset=off)
      _ => ()
    }
    off += p.length()
  }
  out
}

// -------------------------------------------------------------- Slice

///|
fn prepare_slice(info : NodeInfo) -> Kernel raise {
  let t = expect_type(info, 0, all_types)
  let legacy = info.since < 10
  let attr_starts = if legacy { info.attrs.ints("starts") } else { None }
  let attr_ends = if legacy { info.attrs.ints("ends") } else { None }
  let attr_axes = if legacy { info.attrs.ints("axes") } else { None }
  if !legacy {
    let it = expect_type(info, 1, index_types)
    same_type(info, 2, it)
    for i in 3..<info.present.length() {
      if info.present[i] {
        same_type(info, i, it)
      }
    }
  }
  Kernel::new([t], fn(inputs) raise {
    let x = arg(inputs, 0)
    let xs = x.shape()
    let r = xs.length()
    let starts = match attr_starts {
      Some(s) => FixedArray::from_array(s)
      None => shape_input(arg(inputs, 1), "Slice starts")
    }
    let ends = match attr_ends {
      Some(s) => FixedArray::from_array(s)
      None => shape_input(arg(inputs, 2), "Slice ends")
    }
    let axes = match (attr_axes, legacy) {
      (Some(a), _) => FixedArray::from_array(a)
      (None, true) => FixedArray::makei(starts.length(), fn(i) { i.to_int64() })
      (None, false) =>
        match opt_arg(inputs, 3) {
          Some(a) => shape_input(a, "Slice axes")
          None => FixedArray::makei(starts.length(), fn(i) { i.to_int64() })
        }
    }
    let steps = if legacy {
      FixedArray::make(starts.length(), 1L)
    } else {
      match opt_arg(inputs, 4) {
        Some(s) => shape_input(s, "Slice steps")
        None => FixedArray::make(starts.length(), 1L)
      }
    }
    if ends.length() != starts.length() ||
      axes.length() != starts.length() ||
      steps.length() != starts.length() {
      raise OpError(
        "Slice: starts, ends, axes and steps must have the same length",
      )
    }
    let st = FixedArray::make(r, 0)
    let step = FixedArray::make(r, 1)
    let cnt = FixedArray::makei(r, fn(i) { xs[i] })
    let used = FixedArray::make(r, false)
    for i in 0..<starts.length() {
      let a = norm_axis(axes[i], r)
      if used[a] {
        raise OpError("Slice: axis \{axes[i]} given twice")
      }
      used[a] = true
      let dim = xs[a].to_int64()
      let s = steps[i]
      if s == 0L {
        raise OpError("Slice: step cannot be 0")
      }
      let clamp = fn(v : Int64, lo : Int64, hi : Int64) {
        if v < lo {
          lo
        } else if v > hi {
          hi
        } else {
          v
        }
      }
      let mut b = starts[i]
      let mut e = ends[i]
      if b < 0L {
        b = b + dim
      }
      if e < 0L {
        e = e + dim
      }
      if s > 0L {
        b = clamp(b, 0L, dim)
        e = clamp(e, 0L, dim)
      } else {
        b = clamp(b, 0L, dim - 1L)
        e = clamp(e, -1L, dim - 1L)
      }
      let count = if s > 0L {
        if e > b {
          (e - b + s - 1L) / s
        } else {
          0L
        }
      } else if b > e {
        (b - e + -s - 1L) / -s
      } else {
        0L
      }
      st[a] = b.to_int()
      step[a] = s.to_int()
      cnt[a] = count.to_int()
    }
    let in_str = @tensor.strides(xs)
    let src_str = FixedArray::makei(r, fn(i) { in_str[i] * step[i] })
    let base = {
      let mut o = 0
      for i in 0..<r {
        o += st[i] * in_str[i]
      }
      o
    }
    let n = numel(cnt)
    let idx = FixedArray::make(n, 0)
    @tensor.for_each_broadcast1(cnt, src_str, fn(o, s) { idx[o] = base + s })
    [tensor_new(cnt, take(x.data(), idx, None))]
  })
}

// ------------------------------------------------------------- Gather

///|
fn prepare_gather(info : NodeInfo) -> Kernel raise {
  let t = expect_type(info, 0, all_types)
  ignore(expect_type(info, 1, index_types))
  let axis = info.attrs.int("axis", 0L)
  Kernel::new([t], fn(inputs) raise {
    let x = arg(inputs, 0)
    let ind = arg(inputs, 1)
    let xs = x.shape()
    let r = xs.length()
    let a = norm_axis(axis, r)
    let len = xs[a]
    let iv = ind.to_i64_array()
    let mut outer = 1
    for i in 0..<a {
      outer *= xs[i]
    }
    let mut inner = 1
    for i in (a + 1)..<r {
      inner *= xs[i]
    }
    let dims = []
    for i in 0..<a {
      dims.push(xs[i])
    }
    for d in ind.shape() {
      dims.push(d)
    }
    for i in (a + 1)..<r {
      dims.push(xs[i])
    }
    let out_shape = FixedArray::from_array(dims)
    let ni = iv.length()
    let idx = FixedArray::make(outer * ni * inner, 0)
    for k in 0..<ni {
      let v = iv[k]
      let j = if v < 0L { v + len.to_int64() } else { v }
      if j < 0L || j >= len.to_int64() {
        raise OpError(
          "Gather: index \{v} is out of range for axis of size \{len}",
        )
      }
      let j = j.to_int()
      for o in 0..<outer {
        for i in 0..<inner {
          idx[(o * ni + k) * inner + i] = (o * len + j) * inner + i
        }
      }
    }
    [tensor_new(out_shape, take(x.data(), idx, None))]
  })
}

// -------------------------------------------------------------- Shape

///|
fn prepare_shape(info : NodeInfo) -> Kernel raise {
  ignore(expect_type(info, 0, all_types))
  let start = if info.since >= 15 { info.attrs.int("start", 0L) } else { 0L }
  let end_ = if info.since >= 15 { info.attrs.int_opt("end") } else { None }
  Kernel::new([I64], fn(inputs) raise {
    let x = arg(inputs, 0)
    let r = x.rank().to_int64()
    let clamp = fn(v : Int64) {
      let v = if v < 0L { v + r } else { v }
      if v < 0L {
        0L
      } else if v > r {
        r
      } else {
        v
      }
    }
    let s = clamp(start).to_int()
    let e = clamp(end_.unwrap_or(r)).to_int()
    let n = if e > s { e - s } else { 0 }
    let v = FixedArray::makei(n, fn(i) { x.shape()[s + i].to_int64() })
    [tensor_new([n], I64(v))]
  })
}

// ---------------------------------------------------------------- Pad

///|
fn prepare_pad(info : NodeInfo) -> Kernel raise {
  let t = expect_type(info, 0, all_types)
  let mode = info.attrs.string("mode", "constant")
  let modes = if info.since >= 19 {
    ["constant", "reflect", "edge", "wrap"]
  } else {
    ["constant", "reflect", "edge"]
  }
  if !modes.contains(mode) {
    raise OpError("Pad-\{info.since}: mode '\{mode}' is not supported")
  }
  let legacy = info.since < 11
  let attr_pads = if legacy { info.attrs.ints("pads") } else { None }
  let attr_value = if legacy { info.attrs.float("value", 0.0) } else { 0.0 }
  if !legacy {
    ignore(expect_type(info, 1, [I64]))
    if info.has_input(2) {
      same_type(info, 2, t)
    }
    if info.has_input(3) {
      ignore(expect_type(info, 3, index_types))
    }
  }
  Kernel::new([t], fn(inputs) raise {
    let x = arg(inputs, 0)
    let xs = x.shape()
    let r = xs.length()
    let raw = match attr_pads {
      Some(p) => FixedArray::from_array(p)
      None => shape_input(arg(inputs, 1), "Pad pads")
    }
    let axes = match opt_arg(inputs, 3) {
      Some(a) if !legacy => {
        let v = a.to_i64_array()
        FixedArray::makei(v.length(), fn(i) { v[i] })
      }
      _ => FixedArray::makei(r, fn(i) { i.to_int64() })
    }
    let na = axes.length()
    if raw.length() != 2 * na {
      raise OpError("Pad: pads has \{raw.length()} values, expected \{2 * na}")
    }
    let before = FixedArray::make(r, 0)
    let after = FixedArray::make(r, 0)
    for i in 0..<na {
      let a = norm_axis(axes[i], r)
      before[a] = raw[i].to_int()
      after[a] = raw[i + na].to_int()
    }
    let out_shape = FixedArray::makei(r, fn(i) { xs[i] + before[i] + after[i] })
    for i in 0..<r {
      if out_shape[i] < 0 {
        raise OpError("Pad: negative output dimension \{out_shape[i]}")
      }
      if mode != "constant" && xs[i] == 0 && out_shape[i] > 0 {
        raise OpError(
          "Pad: mode '\{mode}' needs a non-empty input along axis \{i}",
        )
      }
      if mode == "reflect" &&
        (before[i] >= xs[i] || after[i] >= xs[i]) &&
        (before[i] > 0 || after[i] > 0) &&
        xs[i] > 0 {
        raise OpError("Pad: reflect padding must be smaller than the dimension")
      }
    }
    let fill : @tensor.Data = match (opt_arg(inputs, 2), x.data()) {
      (Some(v), _) if !legacy => {
        if v.size() != 1 {
          raise OpError("Pad: constant_value must be a scalar")
        }
        v.data()
      }
      (_, F32(_)) => F32([attr_value])
      (_, F64(_)) => F64([attr_value.to_double()])
      (_, d) => @tensor.Data::zeros(d.dtype(), 1)
    }
    let in_str = @tensor.strides(xs)
    let n = numel(out_shape)
    let idx = FixedArray::make(n, 0)
    let pos = FixedArray::make(r, 0)
    for o in 0..<n {
      let mut rem = o
      for d = r - 1; d >= 0; d = d - 1 {
        pos[d] = rem % out_shape[d]
        rem = rem / out_shape[d]
      }
      let mut off = 0
      let mut outside = false
      for d in 0..<r {
        let len = xs[d]
        let mut p = pos[d] - before[d]
        if p < 0 || p >= len {
          match mode {
            "constant" => outside = true
            "edge" => p = if p < 0 { 0 } else { len - 1 }
            "wrap" => p = (p % len + len) % len
            _ => {
              // reflect without repeating the edge
              let period = 2 * (len - 1)
              if period == 0 {
                p = 0
              } else {
                p = (p % period + period) % period
                if p >= len {
                  p = period - p
                }
              }
            }
          }
        }
        off += p * in_str[d]
      }
      idx[o] = if outside { -1 } else { off }
    }
    [tensor_new(out_shape, take(x.data(), idx, Some(fill)))]
  })
}

// ------------------------------------------- Constant, ConstantOfShape

///|
fn prepare_constant(info : NodeInfo) -> Kernel raise {
  let names = info.attrs.names()
  if names.length() != 1 {
    raise OpError(
      "Constant needs exactly one value attribute, got \{names.length()}",
    )
  }
  let value : @tensor.Tensor = match names[0] {
    "value" =>
      match info.attrs.tensor("value") {
        Some(t) => t
        None => raise OpError("Constant: missing value")
      }
    "value_float" =>
      @tensor.Tensor::scalar_f32(info.attrs.float("value_float", 0.0))
    "value_int" => @tensor.Tensor::scalar_i64(info.attrs.int("value_int", 0L))
    "value_floats" => {
      let v = info.attrs.floats("value_floats").unwrap_or([])
      tensor_new([v.length()], F32(FixedArray::from_array(v)))
    }
    "value_ints" => {
      let v = info.attrs.ints("value_ints").unwrap_or([])
      tensor_new([v.length()], I64(FixedArray::from_array(v)))
    }
    other =>
      raise OpError(
        "Constant with attribute '\{other}' (strings / sparse) is not supported",
      )
  }
  Kernel::new([value.dtype()], fn(_) { [value] })
}

///|
fn prepare_constant_of_shape(info : NodeInfo) -> Kernel raise {
  ignore(expect_type(info, 0, [I64]))
  let value = match info.attrs.tensor("value") {
    Some(v) => {
      if v.size() != 1 {
        raise OpError("ConstantOfShape: value must have exactly one element")
      }
      v
    }
    None => @tensor.Tensor::scalar_f32(0.0)
  }
  Kernel::new([value.dtype()], fn(inputs) raise {
    let s = shape_input(arg(inputs, 0), "ConstantOfShape input")
    let shape = FixedArray::makei(s.length(), fn(i) { s[i].to_int() })
    for i in 0..<s.length() {
      if s[i] < 0L || s[i] > 2147483647L {
        raise OpError("ConstantOfShape: invalid dimension \{s[i]}")
      }
    }
    let n = numel(shape)
    [tensor_new(shape, take(value.data(), FixedArray::make(n, 0), None))]
  })
}

// ------------------------------------------------------------- Expand

///|
fn prepare_expand(info : NodeInfo) -> Kernel raise {
  let t = expect_type(info, 0, all_types)
  ignore(expect_type(info, 1, [I64]))
  Kernel::new([t], fn(inputs) raise {
    let x = arg(inputs, 0)
    let s = shape_input(arg(inputs, 1), "Expand shape")
    let target = i64s_to_ints(s, "Expand shape")
    let out_shape = @tensor.broadcast_shapes(x.shape(), target) catch {
      @tensor.TensorError(m) => raise OpError("Expand: \{m}")
    }
    let n = numel(out_shape)
    let idx = FixedArray::make(n, 0)
    let sx = @tensor.broadcast_strides(x.shape(), out_shape)
    @tensor.for_each_broadcast1(out_shape, sx, fn(o, i) { idx[o] = i })
    [tensor_new(out_shape, take(x.data(), idx, None))]
  })
}
