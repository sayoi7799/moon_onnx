// Convolution, pooling and batch normalization (float32, N-d spatial).

///|
/// Spatial window parameters shared by Conv and the pooling operators.
priv struct Window {
  kernel : FixedArray[Int32]
  strides : FixedArray[Int32]
  dilations : FixedArray[Int32]
  /// begin pads followed by end pads, as in the ONNX `pads` attribute
  pads : FixedArray[Int32]
  auto_pad : String
  ceil_mode : Bool
}

///|
fn ints_attr(
  info : NodeInfo,
  name : String,
  len : Int32,
  default : Int32,
) -> FixedArray[Int32]? raise {
  match info.attrs.ints(name) {
    None => if len >= 0 { Some(FixedArray::make(len, default)) } else { None }
    Some(v) => {
      if len >= 0 && v.length() != len {
        raise OpError(
          "attribute '\{name}' must have \{len} values, got \{v.length()}",
        )
      }
      Some(FixedArray::makei(v.length(), fn(i) { v[i].to_int() }))
    }
  }
}

///|
/// Parses window attributes. `spatial` is the number of spatial dims if
/// known at load time (from kernel_shape), else -1.
fn parse_window(info : NodeInfo, kernel : FixedArray[Int32]) -> Window raise {
  let k = kernel.length()
  for d in kernel {
    if d <= 0 {
      raise OpError(
        "kernel_shape values must be positive, got \{shape_str(kernel)}",
      )
    }
  }
  let strides = ints_attr(info, "strides", k, 1).unwrap()
  let dilations = if info.attrs.has("dilations") ||
    info.op_type == "Conv" ||
    info.since >= 10 {
    ints_attr(info, "dilations", k, 1).unwrap()
  } else {
    FixedArray::make(k, 1)
  }
  let pads = ints_attr(info, "pads", 2 * k, 0).unwrap()
  for s in strides {
    if s <= 0 {
      raise OpError("strides must be positive, got \{shape_str(strides)}")
    }
  }
  for d in dilations {
    if d <= 0 {
      raise OpError("dilations must be positive, got \{shape_str(dilations)}")
    }
  }
  for p in pads {
    if p < 0 {
      raise OpError("negative pads are not supported: \{shape_str(pads)}")
    }
  }
  let auto_pad = info.attrs.string("auto_pad", "NOTSET")
  if !["NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"].contains(auto_pad) {
    raise OpError("unknown auto_pad '\{auto_pad}'")
  }
  if auto_pad != "NOTSET" && info.attrs.has("pads") {
    let mut nonzero = false
    for p in pads {
      if p != 0 {
        nonzero = true
      }
    }
    if nonzero {
      raise OpError("pads cannot be used together with auto_pad=\{auto_pad}")
    }
  }
  let ceil_mode = if info.since >= 10 && info.op_type != "Conv" {
    info.attrs.flag("ceil_mode", false)
  } else {
    false
  }
  { kernel, strides, dilations, pads, auto_pad, ceil_mode, }
}

///|
/// Resolves (pads_begin, pads_end, output size) for each spatial dim.
fn resolve_window(
  w : Window,
  input : FixedArray[Int32],
) -> (FixedArray[Int32], FixedArray[Int32], FixedArray[Int32]) raise {
  let k = w.kernel.length()
  let pb = FixedArray::make(k, 0)
  let pe = FixedArray::make(k, 0)
  let out = FixedArray::make(k, 0)
  for i in 0..<k {
    let eff = (w.kernel[i] - 1) * w.dilations[i] + 1
    let s = w.strides[i]
    let n = input[i]
    match w.auto_pad {
      "SAME_UPPER" | "SAME_LOWER" => {
        let o = (n + s - 1) / s
        let total = (o - 1) * s + eff - n
        let total = if total < 0 { 0 } else { total }
        if w.auto_pad == "SAME_UPPER" {
          pb[i] = total / 2
        } else {
          pb[i] = total - total / 2
        }
        pe[i] = total - pb[i]
        out[i] = o
      }
      _ => {
        if w.auto_pad == "NOTSET" {
          pb[i] = w.pads[i]
          pe[i] = w.pads[i + k]
        }
        let span = n + pb[i] + pe[i] - eff
        if span < 0 {
          raise OpError(
            "input spatial size \{n} (padded \{n + pb[i] + pe[i]}) is smaller than the kernel extent \{eff}",
          )
        }
        let mut o = if w.ceil_mode {
          (span + s - 1) / s + 1
        } else {
          span / s + 1
        }
        // In ceil mode the last window must start inside the input or the
        // begin padding (otherwise it would only cover end padding).
        if w.ceil_mode && (o - 1) * s >= n + pb[i] {
          o = o - 1
        }
        out[i] = o
      }
    }
  }
  (pb, pe, out)
}

///|
fn spatial_of(shape : FixedArray[Int32]) -> FixedArray[Int32] {
  FixedArray::makei(shape.length() - 2, fn(i) { shape[i + 2] })
}

// ---------------------------------------------------------------- Conv

///|
fn prepare_conv(info : NodeInfo) -> Kernel raise {
  ignore(expect_type(info, 0, [F32]))
  same_type(info, 1, F32)
  if info.has_input(2) {
    same_type(info, 2, F32)
  }
  let group = info.attrs.int("group", 1L)
  if group < 1L {
    raise OpError("group must be >= 1, got \{group}")
  }
  let group = group.to_int()
  let kernel_attr = ints_attr(info, "kernel_shape", -1, 0)
  // When the weights are an initializer, validate their shape at load time.
  if info.constant(1) is Some(w) {
    if w.rank() < 3 {
      raise OpError(
        "Conv weights must have rank >= 3, got \{shape_str(w.shape())}",
      )
    }
    if w.shape()[0] % group != 0 {
      raise OpError(
        "Conv: \{w.shape()[0]} output channels not divisible by group \{group}",
      )
    }
    if kernel_attr is Some(ks) &&
      !@tensor.shapes_equal(ks, spatial_of(w.shape())) {
      raise OpError(
        "kernel_shape \{shape_str(ks)} does not match weights \{shape_str(w.shape())}",
      )
    }
  }
  // Parse everything once with a placeholder kernel to validate attributes.
  let k_guess = match (kernel_attr, info.constant(1)) {
    (Some(ks), _) => ks
    (None, Some(w)) => spatial_of(w.shape())
    _ =>
      FixedArray::make(
        info.attrs.ints("strides").map(fn(a) { a.length() }).unwrap_or(2),
        1,
      )
  }
  ignore(parse_window(info, k_guess))
  Kernel::new([F32], fn(inputs) raise {
    let x = arg(inputs, 0)
    let w = arg(inputs, 1)
    let b = opt_arg(inputs, 2)
    [conv_f32(info, x, w, b, group, kernel_attr)]
  })
}

///|
fn conv_f32(
  info : NodeInfo,
  x : @tensor.Tensor,
  w : @tensor.Tensor,
  b : @tensor.Tensor?,
  group : Int32,
  kernel_attr : FixedArray[Int32]?,
) -> @tensor.Tensor raise {
  if x.rank() < 3 || w.rank() != x.rank() {
    raise OpError(
      "Conv: input \{shape_str(x.shape())} and weights \{shape_str(w.shape())} must have the same rank >= 3",
    )
  }
  let xs = x.shape()
  let ws = w.shape()
  let n = xs[0]
  let c = xs[1]
  let m = ws[0]
  let cg = ws[1]
  if cg * group != c {
    raise OpError(
      "Conv: input has \{c} channels but weights expect \{cg} x group \{group}",
    )
  }
  if m % group != 0 {
    raise OpError("Conv: \{m} output channels not divisible by group \{group}")
  }
  let kernel = spatial_of(ws)
  if kernel_attr is Some(ks) && !@tensor.shapes_equal(ks, kernel) {
    raise OpError(
      "kernel_shape \{shape_str(ks)} does not match weights \{shape_str(ws)}",
    )
  }
  let win = parse_window(info, kernel)
  let in_sp = spatial_of(xs)
  let (pb, _, out_sp) = resolve_window(win, in_sp)
  let bias = match b {
    Some(bt) => {
      if bt.rank() != 1 || bt.shape()[0] != m {
        raise OpError(
          "Conv: bias must have shape [\{m}], got \{shape_str(bt.shape())}",
        )
      }
      Some(bt.to_f32_array())
    }
    None => None
  }
  let mg = m / group
  let out_hw = numel(out_sp)
  let in_hw = numel(in_sp)
  let ksz = numel(kernel)
  let xv = x.to_f32_array()
  let wv = w.to_f32_array()
  let y = FixedArray::make(n * m * out_hw, (0.0 : Float))
  let k = kernel.length()
  let is_pointwise = {
    let mut p = true
    for i in 0..<k {
      if kernel[i] != 1 ||
        win.strides[i] != 1 ||
        pb[i] != 0 ||
        out_sp[i] != in_sp[i] {
        p = false
      }
    }
    p
  }
  let is_depthwise = cg == 1 && mg == 1 && k == 2
  if is_depthwise {
    conv2d_depthwise(
      xv,
      wv,
      y,
      n,
      c,
      in_sp,
      out_sp,
      kernel,
      win.strides,
      win.dilations,
      pb,
    )
  } else {
    let col = if is_pointwise {
      FixedArray::make(0, (0.0 : Float))
    } else {
      FixedArray::make(cg * ksz * out_hw, (0.0 : Float))
    }
    for img in 0..<n {
      for g in 0..<group {
        let x_off = (img * c + g * cg) * in_hw
        let y_off = (img * m + g * mg) * out_hw
        let w_off = g * mg * cg * ksz
        if is_pointwise {
          // Y_g[mg, hw] = W_g[mg, cg] * X_g[cg, hw]
          sgemm_acc(mg, out_hw, cg, wv, w_off, xv, x_off, y, y_off)
        } else {
          if k == 2 {
            im2col_2d(
              xv,
              x_off,
              cg,
              in_sp,
              out_sp,
              kernel,
              win.strides,
              win.dilations,
              pb,
              col,
            )
          } else {
            im2col_nd(
              xv,
              x_off,
              cg,
              in_sp,
              out_sp,
              kernel,
              win.strides,
              win.dilations,
              pb,
              col,
            )
          }
          sgemm_acc(mg, out_hw, cg * ksz, wv, w_off, col, 0, y, y_off)
        }
      }
    }
  }
  if bias is Some(bv) {
    for img in 0..<n {
      for oc in 0..<m {
        let off = (img * m + oc) * out_hw
        let bb = bv[oc]
        for i in 0..<out_hw {
          y[off + i] = y[off + i] + bb
        }
      }
    }
  }
  let out_shape = FixedArray::make(xs.length(), 0)
  out_shape[0] = n
  out_shape[1] = m
  for i in 0..<k {
    out_shape[i + 2] = out_sp[i]
  }
  tensor_new(out_shape, F32(y))
}

///|
/// col[(ci*kh + ki)*kw + kj, oh*ow_n + ow] = x[ci, oh*s0 - p0 + ki*d0, ow*s1 - p1 + kj*d1]
fn im2col_2d(
  x : FixedArray[Float],
  x_off : Int32,
  channels : Int32,
  in_sp : FixedArray[Int32],
  out_sp : FixedArray[Int32],
  kernel : FixedArray[Int32],
  strides : FixedArray[Int32],
  dil : FixedArray[Int32],
  pb : FixedArray[Int32],
  col : FixedArray[Float],
) -> Unit {
  let ih = in_sp[0]
  let iw = in_sp[1]
  let oh = out_sp[0]
  let ow = out_sp[1]
  let kh = kernel[0]
  let kw = kernel[1]
  let mut row = 0
  for ci in 0..<channels {
    let cbase = x_off + ci * ih * iw
    for ki in 0..<kh {
      for kj in 0..<kw {
        let rbase = row * oh * ow
        for y in 0..<oh {
          let iy = y * strides[0] - pb[0] + ki * dil[0]
          let obase = rbase + y * ow
          if iy < 0 || iy >= ih {
            for xo in 0..<ow {
              col[obase + xo] = 0.0
            }
          } else {
            let ibase = cbase + iy * iw
            for xo in 0..<ow {
              let ix = xo * strides[1] - pb[1] + kj * dil[1]
              col[obase + xo] = if ix < 0 || ix >= iw {
                0.0
              } else {
                x[ibase + ix]
              }
            }
          }
        }
        row += 1
      }
    }
  }
}

///|
/// Generic N-d im2col (used for 1-D and 3-D convolutions).
fn im2col_nd(
  x : FixedArray[Float],
  x_off : Int32,
  channels : Int32,
  in_sp : FixedArray[Int32],
  out_sp : FixedArray[Int32],
  kernel : FixedArray[Int32],
  strides : FixedArray[Int32],
  dil : FixedArray[Int32],
  pb : FixedArray[Int32],
  col : FixedArray[Float],
) -> Unit {
  let k = kernel.length()
  let out_n = out_sp.fold(init=1, fn(a, d) { a * d })
  let k_n = kernel.fold(init=1, fn(a, d) { a * d })
  let in_n = in_sp.fold(init=1, fn(a, d) { a * d })
  let in_str = @tensor.strides(in_sp)
  let kpos = FixedArray::make(k, 0)
  let opos = FixedArray::make(k, 0)
  for ci in 0..<channels {
    for kk in 0..<k_n {
      // decompose kernel index
      let mut r = kk
      for d = k - 1; d >= 0; d = d - 1 {
        kpos[d] = r % kernel[d]
        r = r / kernel[d]
      }
      let row = (ci * k_n + kk) * out_n
      for o in 0..<out_n {
        let mut r2 = o
        for d = k - 1; d >= 0; d = d - 1 {
          opos[d] = r2 % out_sp[d]
          r2 = r2 / out_sp[d]
        }
        let mut off = 0
        let mut inside = true
        for d in 0..<k {
          let p = opos[d] * strides[d] - pb[d] + kpos[d] * dil[d]
          if p < 0 || p >= in_sp[d] {
            inside = false
          }
          off += p * in_str[d]
        }
        col[row + o] = if inside { x[x_off + ci * in_n + off] } else { 0.0 }
      }
    }
  }
}

///|
/// Depthwise 2-D convolution (one input channel per group, one output
/// channel per group): a direct loop, avoiding im2col.
fn conv2d_depthwise(
  x : FixedArray[Float],
  w : FixedArray[Float],
  y : FixedArray[Float],
  n : Int32,
  c : Int32,
  in_sp : FixedArray[Int32],
  out_sp : FixedArray[Int32],
  kernel : FixedArray[Int32],
  strides : FixedArray[Int32],
  dil : FixedArray[Int32],
  pb : FixedArray[Int32],
) -> Unit {
  let ih = in_sp[0]
  let iw = in_sp[1]
  let oh = out_sp[0]
  let ow = out_sp[1]
  let kh = kernel[0]
  let kw = kernel[1]
  for img in 0..<n {
    for ch in 0..<c {
      let xb = (img * c + ch) * ih * iw
      let yb = (img * c + ch) * oh * ow
      let wb = ch * kh * kw
      for oy in 0..<oh {
        for ox in 0..<ow {
          let mut acc : Float = 0.0
          for ki in 0..<kh {
            let iy = oy * strides[0] - pb[0] + ki * dil[0]
            if iy >= 0 && iy < ih {
              for kj in 0..<kw {
                let ix = ox * strides[1] - pb[1] + kj * dil[1]
                if ix >= 0 && ix < iw {
                  acc = acc + w[wb + ki * kw + kj] * x[xb + iy * iw + ix]
                }
              }
            }
          }
          y[yb + oy * ow + ox] = acc
        }
      }
    }
  }
}

// ------------------------------------------------------------- Pooling

///|
fn pool_kernel(info : NodeInfo) -> FixedArray[Int32] raise {
  ints_attr(info, "kernel_shape", -1, 0).unwrap()
}

///|
fn prepare_maxpool(info : NodeInfo) -> Kernel raise {
  let allowed : Array[@tensor.DType] = if info.since >= 12 {
    [F32, U8]
  } else {
    [F32]
  }
  let t = expect_type(info, 0, allowed)
  let kernel = pool_kernel(info)
  let win = parse_window(info, kernel)
  let storage_order = info.attrs.int("storage_order", 0L)
  if storage_order != 0L && storage_order != 1L {
    raise OpError("storage_order must be 0 or 1, got \{storage_order}")
  }
  Kernel::new([t, I64], fn(inputs) raise {
    let x = arg(inputs, 0)
    pool(x, win, "max", false, storage_order == 1L)
  })
}

///|
fn prepare_averagepool(info : NodeInfo) -> Kernel raise {
  ignore(expect_type(info, 0, [F32]))
  let kernel = pool_kernel(info)
  let win = parse_window(info, kernel)
  let include_pad = if info.since >= 7 {
    info.attrs.flag("count_include_pad", false)
  } else {
    false
  }
  Kernel::new([F32], fn(inputs) raise {
    let x = arg(inputs, 0)
    pool(x, win, "avg", include_pad, false)
  })
}

///|
/// Generic N-d pooling over [N, C, spatial...].
/// MaxPool also produces Indices: flat offsets into the input (row-major,
/// or column-major over spatial dims when storage_order = 1).
fn pool(
  x : @tensor.Tensor,
  win : Window,
  mode : String,
  include_pad : Bool,
  col_major : Bool,
) -> Array[@tensor.Tensor] raise {
  let xs = x.shape()
  let k = win.kernel.length()
  if xs.length() != k + 2 {
    raise OpError(
      "pooling input \{shape_str(xs)} must have rank \{k + 2} for kernel_shape of length \{k}",
    )
  }
  let in_sp = spatial_of(xs)
  let (pb, pe, out_sp) = resolve_window(win, in_sp)
  let n = xs[0]
  let c = xs[1]
  let in_hw = numel(in_sp)
  let out_hw = numel(out_sp)
  let k_n = numel(win.kernel)
  let in_str = @tensor.strides(in_sp)
  // column-major spatial strides for storage_order = 1
  let cm_str = FixedArray::make(k, 1)
  for d in 1..<k {
    cm_str[d] = cm_str[d - 1] * in_sp[d - 1]
  }
  // kernel offsets
  let koff = FixedArray::make(k_n * k, 0)
  for kk in 0..<k_n {
    let mut r = kk
    for d = k - 1; d >= 0; d = d - 1 {
      koff[kk * k + d] = r % win.kernel[d] * win.dilations[d]
      r = r / win.kernel[d]
    }
  }
  let total = n * c * out_hw
  let indices = FixedArray::make(if mode == "max" { total } else { 0 }, 0L)
  let start = FixedArray::make(k, 0)
  let opos = FixedArray::make(k, 0)
  let is_u8 = x.dtype() == U8
  let xf : FixedArray[Float] = if is_u8 { [] } else { x.to_f32_array() }
  let xu : FixedArray[Byte] = match x.data() {
    U8(a) => a
    _ => []
  }
  let yf = FixedArray::make(if is_u8 { 0 } else { total }, (0.0 : Float))
  let yu = FixedArray::make(if is_u8 { total } else { 0 }, b'\x00')
  for plane in 0..<(n * c) {
    let xbase = plane * in_hw
    for o in 0..<out_hw {
      let mut r = o
      for d = k - 1; d >= 0; d = d - 1 {
        opos[d] = r % out_sp[d]
        r = r / out_sp[d]
        start[d] = opos[d] * win.strides[d] - pb[d]
      }
      let out_i = plane * out_hw + o
      if mode == "max" {
        let mut best : Double = @double.neg_infinity
        let mut best_idx = -1
        let mut best_cm = -1
        for kk in 0..<k_n {
          let mut off = 0
          let mut cm = 0
          let mut inside = true
          for d in 0..<k {
            let p = start[d] + koff[kk * k + d]
            if p < 0 || p >= in_sp[d] {
              inside = false
            }
            off += p * in_str[d]
            cm += p * cm_str[d]
          }
          if inside {
            let v = if is_u8 {
              xu[xbase + off].to_int().to_double()
            } else {
              xf[xbase + off].to_double()
            }
            if best_idx < 0 || v > best || (v.is_nan() && !best.is_nan()) {
              best = v
              best_idx = off
              best_cm = cm
            }
          }
        }
        if is_u8 {
          yu[out_i] = if best_idx < 0 {
            b'\x00'
          } else {
            best.to_int().to_byte()
          }
        } else {
          yf[out_i] = if best_idx < 0 {
            @float.neg_infinity
          } else {
            Float::from_double(best)
          }
        }
        let flat = if col_major { best_cm } else { best_idx }
        indices[out_i] = (plane * in_hw + flat).to_int64()
      } else {
        let mut sum = 0.0
        let mut count = 0
        for kk in 0..<k_n {
          let mut off = 0
          let mut inside = true
          let mut in_padded = true
          for d in 0..<k {
            let p = start[d] + koff[kk * k + d]
            if p < 0 || p >= in_sp[d] {
              inside = false
            }
            if p < -pb[d] || p >= in_sp[d] + pe[d] {
              in_padded = false
            }
            off += p * in_str[d]
          }
          if inside {
            sum = sum + xf[xbase + off].to_double()
            count += 1
          } else if include_pad && in_padded {
            count += 1
          }
        }
        yf[out_i] = if count == 0 {
          @float.not_a_number
        } else {
          Float::from_double(sum / count.to_double())
        }
      }
    }
  }
  let out_shape = FixedArray::make(k + 2, 0)
  out_shape[0] = n
  out_shape[1] = c
  for d in 0..<k {
    out_shape[d + 2] = out_sp[d]
  }
  let y = if is_u8 {
    tensor_new(out_shape, U8(yu))
  } else {
    tensor_new(out_shape, F32(yf))
  }
  if mode == "max" {
    [y, tensor_new(out_shape, I64(indices))]
  } else {
    [y]
  }
}

///|
fn prepare_global_average_pool(info : NodeInfo) -> Kernel raise {
  ignore(expect_type(info, 0, [F32]))
  Kernel::new([F32], fn(inputs) raise {
    let x = arg(inputs, 0)
    let xs = x.shape()
    if xs.length() < 3 {
      raise OpError(
        "GlobalAveragePool input must have rank >= 3, got \{shape_str(xs)}",
      )
    }
    let planes = xs[0] * xs[1]
    let hw = numel(spatial_of(xs))
    let xv = x.to_f32_array()
    let y = FixedArray::make(planes, (0.0 : Float))
    for p in 0..<planes {
      let mut s = 0.0
      for i in 0..<hw {
        s = s + xv[p * hw + i].to_double()
      }
      y[p] = Float::from_double(s / hw.to_double())
    }
    let out_shape = FixedArray::make(xs.length(), 1)
    out_shape[0] = xs[0]
    out_shape[1] = xs[1]
    [tensor_new(out_shape, F32(y))]
  })
}

// --------------------------------------------------------- BatchNorm

///|
/// Inference-mode batch normalization:
/// y = (x - mean) / sqrt(var + epsilon) * scale + B, per channel (dim 1).
fn prepare_batchnorm(info : NodeInfo) -> Kernel raise {
  ignore(expect_type(info, 0, [F32]))
  for i in 1..<5 {
    same_type(info, i, F32)
  }
  if info.since < 7 && !info.attrs.flag("is_test", false) {
    raise OpError(
      "BatchNormalization-\{info.since} with is_test=0 (training) is not supported",
    )
  }
  if info.since < 9 && !info.attrs.flag("spatial", true) {
    raise OpError("BatchNormalization with spatial=0 is not supported")
  }
  if info.since >= 14 && info.attrs.flag("training_mode", false) {
    raise OpError("BatchNormalization with training_mode=1 is not supported")
  }
  if info.num_outputs > 1 {
    raise OpError(
      "BatchNormalization: the optional outputs (running statistics) exist only in training mode, which is not supported",
    )
  }
  let eps = info.attrs.float("epsilon", 1.0e-5).to_double()
  Kernel::new([F32], fn(inputs) raise {
    let x = arg(inputs, 0)
    let xs = x.shape()
    if xs.length() < 2 {
      raise OpError(
        "BatchNormalization input must have rank >= 2, got \{shape_str(xs)}",
      )
    }
    let c = xs[1]
    let params = FixedArray::makei(4, fn(i) { inputs[i + 1] })
    let vals = []
    for i in 0..<4 {
      guard params[i] is Some(p) else {
        raise OpError("missing input \{i + 1}")
      }
      if p.size() != c {
        raise OpError(
          "BatchNormalization: input \{i + 1} has \{p.size()} values, expected \{c} channels",
        )
      }
      vals.push(p.to_f32_array())
    }
    let scale = vals[0]
    let bias = vals[1]
    let mean = vals[2]
    let var_ = vals[3]
    let n = xs[0]
    let inner = x.size() / (if n * c == 0 { 1 } else { n * c })
    let xv = x.to_f32_array()
    let y = FixedArray::make(xv.length(), (0.0 : Float))
    for ch in 0..<c {
      let a = scale[ch].to_double() / (var_[ch].to_double() + eps).sqrt()
      let b = bias[ch].to_double() - mean[ch].to_double() * a
      let af = Float::from_double(a)
      let bf = Float::from_double(b)
      for img in 0..<n {
        let off = (img * c + ch) * inner
        for i in 0..<inner {
          y[off + i] = xv[off + i] * af + bf
        }
      }
    }
    [tensor_new(xs, F32(y))]
  })
}
