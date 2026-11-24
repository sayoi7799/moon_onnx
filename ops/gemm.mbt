// float32 matrix multiplication and the Gemm / MatMul operators.

///|
/// Row-major C[m x n] += A[m x k] * B[k x n] where each matrix starts at the
/// given offset and rows are contiguous. Loop order i-k-j keeps the inner
/// loop streaming over contiguous rows of B and C; the k dimension is
/// blocked so a panel of B stays in cache.
pub fn sgemm_acc(
  m : Int32,
  n : Int32,
  k : Int32,
  a : FixedArray[Float],
  a_off : Int32,
  b : FixedArray[Float],
  b_off : Int32,
  c : FixedArray[Float],
  c_off : Int32,
) -> Unit {
  let kb = 128
  let mut k0 = 0
  while k0 < k {
    let k1 = if k0 + kb < k { k0 + kb } else { k }
    for i in 0..<m {
      let crow = c_off + i * n
      let arow = a_off + i * k
      for p in k0..<k1 {
        let av = a[arow + p]
        let brow = b_off + p * n
        for j in 0..<n {
          c[crow + j] = c[crow + j] + av * b[brow + j]
        }
      }
    }
    k0 = k1
  }
}

///|
/// Transposes a rows x cols row-major matrix.
fn transpose2d(
  src : FixedArray[Float],
  off : Int32,
  rows : Int32,
  cols : Int32,
) -> FixedArray[Float] {
  let out = FixedArray::make(rows * cols, (0.0 : Float))
  for r in 0..<rows {
    for c in 0..<cols {
      out[c * rows + r] = src[off + r * cols + c]
    }
  }
  out
}

///|
/// Gemm: Y = alpha * A' * B' + beta * C, with A' = transA ? A^T : A, etc.
/// C (optional from Gemm-11) is broadcast to (M, N).
fn prepare_gemm(info : NodeInfo) -> Kernel raise {
  ignore(expect_type(info, 0, [F32]))
  same_type(info, 1, F32)
  if info.has_input(2) {
    same_type(info, 2, F32)
  }
  let alpha = info.attrs.float("alpha", 1.0)
  let beta = info.attrs.float("beta", 1.0)
  let trans_a = info.attrs.flag("transA", false)
  let trans_b = info.attrs.flag("transB", false)
  // Gemm-6: C is broadcast only when broadcast=1
  let strict_c = info.since < 7 && !info.attrs.flag("broadcast", false)
  Kernel::new([F32], fn(inputs) raise {
    let a = arg(inputs, 0)
    let b = arg(inputs, 1)
    if a.rank() != 2 || b.rank() != 2 {
      raise OpError(
        "Gemm: A and B must be 2-D, got \{shape_str(a.shape())} and \{shape_str(b.shape())}",
      )
    }
    let (m, ka) = if trans_a {
      (a.shape()[1], a.shape()[0])
    } else {
      (a.shape()[0], a.shape()[1])
    }
    let (kb, n) = if trans_b {
      (b.shape()[1], b.shape()[0])
    } else {
      (b.shape()[0], b.shape()[1])
    }
    if ka != kb {
      raise OpError(
        "Gemm: inner dimensions differ: A' is \{m}x\{ka}, B' is \{kb}x\{n}",
      )
    }
    let av = a.to_f32_array()
    let bv = b.to_f32_array()
    let a2 = if trans_a {
      transpose2d(av, 0, a.shape()[0], a.shape()[1])
    } else {
      av
    }
    let b2 = if trans_b {
      transpose2d(bv, 0, b.shape()[0], b.shape()[1])
    } else {
      bv
    }
    let y = FixedArray::make(m * n, (0.0 : Float))
    sgemm_acc(m, n, ka, a2, 0, b2, 0, y, 0)
    if alpha != 1.0 {
      for i in 0..<y.length() {
        y[i] = alpha * y[i]
      }
    }
    match opt_arg(inputs, 2) {
      Some(c) if beta != 0.0 => {
        let out_shape : FixedArray[Int32] = [m, n]
        if strict_c && !@tensor.shapes_equal(c.shape(), out_shape) {
          raise OpError(
            "Gemm-6 with broadcast=0 needs C of shape [\{m},\{n}], got \{shape_str(c.shape())}",
          )
        }
        let bc = @tensor.broadcast_shapes(c.shape(), out_shape) catch {
          @tensor.TensorError(msg) => raise OpError("Gemm: C: \{msg}")
        }
        if !@tensor.shapes_equal(bc, out_shape) {
          raise OpError(
            "Gemm: C of shape \{shape_str(c.shape())} cannot be broadcast to [\{m},\{n}]",
          )
        }
        let cv = c.to_f32_array()
        let sc = @tensor.broadcast_strides(c.shape(), out_shape)
        @tensor.for_each_broadcast1(out_shape, sc, fn(o, i) {
          y[o] = y[o] + beta * cv[i]
        })
      }
      _ => ()
    }
    [tensor_new([m, n], F32(y))]
  })
}

///|
/// MatMul with NumPy semantics: 1-D operands are promoted (and the added
/// dimension removed afterwards); leading batch dimensions broadcast.
fn prepare_matmul(info : NodeInfo) -> Kernel raise {
  ignore(expect_type(info, 0, [F32]))
  same_type(info, 1, F32)
  Kernel::new([F32], fn(inputs) raise {
    let a = arg(inputs, 0)
    let b = arg(inputs, 1)
    [matmul_f32(a, b)]
  })
}

///|
fn matmul_f32(a : @tensor.Tensor, b : @tensor.Tensor) -> @tensor.Tensor raise {
  if a.rank() == 0 || b.rank() == 0 {
    raise OpError("MatMul: scalar operands are not allowed")
  }
  let a_shape : FixedArray[Int32] = if a.rank() == 1 {
    [1, a.shape()[0]]
  } else {
    a.shape()
  }
  let b_shape : FixedArray[Int32] = if b.rank() == 1 {
    [b.shape()[0], 1]
  } else {
    b.shape()
  }
  let ra = a_shape.length()
  let rb = b_shape.length()
  let m = a_shape[ra - 2]
  let k = a_shape[ra - 1]
  let k2 = b_shape[rb - 2]
  let n = b_shape[rb - 1]
  if k != k2 {
    raise OpError(
      "MatMul: inner dimensions differ: \{shape_str(a.shape())} x \{shape_str(b.shape())}",
    )
  }
  let a_batch = FixedArray::makei(ra - 2, fn(i) { a_shape[i] })
  let b_batch = FixedArray::makei(rb - 2, fn(i) { b_shape[i] })
  let batch = @tensor.broadcast_shapes(a_batch, b_batch) catch {
    @tensor.TensorError(msg) =>
      raise OpError("MatMul: batch dimensions: \{msg}")
  }
  let nb = numel(batch)
  let av = a.to_f32_array()
  let bv = b.to_f32_array()
  let out = FixedArray::make(nb * m * n, (0.0 : Float))
  let sa = @tensor.broadcast_strides(a_batch, batch)
  let sb = @tensor.broadcast_strides(b_batch, batch)
  @tensor.for_each_broadcast2(batch, sa, sb, fn(o, ia, ib) {
    sgemm_acc(m, n, k, av, ia * m * k, bv, ib * k * n, out, o * m * n)
  })
  // output shape: batch ++ [m] (unless a was 1-D) ++ [n] (unless b was 1-D)
  let dims = []
  for d in batch {
    dims.push(d)
  }
  if a.rank() != 1 {
    dims.push(m)
  }
  if b.rank() != 1 {
    dims.push(n)
  }
  tensor_new(FixedArray::from_array(dims), F32(out))
}
