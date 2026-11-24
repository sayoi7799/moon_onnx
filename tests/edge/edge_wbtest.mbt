// Edge cases: every failure must be an OnnxError that says where it
// happened (node / input / byte offset) - never a crash or a wrong result.

///|
fn model_bytes(name : String) -> Bytes {
  @base64.decode(edge_models[name]) catch {
    _ => abort("bad fixture")
  }
}

///|
/// Loads a fixture; returns the load error message, or "" if it loaded.
fn load_error(name : String) -> String {
  try {
    ignore(@runtime.Session::from_bytes(model_bytes(name)))
    ""
  } catch {
    e => Show::to_string(e)
  }
}

///|
fn expect_load_error(name : String, parts : Array[String]) -> Unit raise {
  let msg = load_error(name)
  assert_true(msg != "", msg="\{name}: loaded without error")
  for p in parts {
    assert_true(msg.contains(p), msg="\{name}: '\{msg}' should mention '\{p}'")
  }
}

///|
fn run_error(
  name : String,
  feeds : Array[(String, @tensor.Tensor)],
) -> String raise {
  let s = @runtime.Session::from_bytes(model_bytes(name))
  try {
    ignore(s.run(feeds))
    ""
  } catch {
    e => Show::to_string(e)
  }
}

///|
fn mnist() -> Bytes raise {
  @fileio.read_file("testdata/models/mnist-12.onnx")
}

///|
test "truncated model files are rejected with a byte offset, at every length" {
  let full = mnist()
  // every prefix length up to 2 KB, then a stride through the rest
  let mut cut = 0
  let mut rejected = 0
  while cut < full.length() {
    let r = try {
      ignore(@runtime.Session::from_bytes_view(full[0:cut]))
      None
    } catch {
      e => Some(Show::to_string(e))
    }
    match r {
      Some(msg) => {
        assert_true(
          msg.has_prefix("decode error: protobuf decode error at byte") ||
          msg.has_prefix("invalid model:"),
          msg="cut \{cut}: \{msg}",
        )
        rejected += 1
      }
      // a cut exactly at a field boundary can still be a valid (smaller) model
      None => ()
    }
    cut += if cut < 2048 { 1 } else { 97 }
  }
  assert_true(rejected > 2000)
}

///|
test "corrupted bytes do not crash the loader" {
  let full = mnist()
  // flip bytes at many positions; each variant must load or fail cleanly
  let mut errors = 0
  for pos = 0; pos < full.length(); pos = pos + 37 {
    let b = FixedArray::makei(full.length(), fn(i) { full[i] })
    b[pos] = (b[pos].to_int() ^ 0xFF).to_byte()
    let bytes = Bytes::from_array(b[:])
    let ok = try {
      let s = @runtime.Session::from_bytes(bytes)
      // if it loads, running it must also either work or fail cleanly
      let input = @tensor.Tensor::zeros(F32, [1, 1, 28, 28])
      ignore(s.run1(input))
      true
    } catch {
      _ => false
    }
    if !ok {
      errors += 1
    }
  }
  assert_true(errors > 0)
  // garbage that is not protobuf at all
  let junk = Bytes::from_array([b'\xFF', b'\xFF', b'\xFF', b'\x07'])
  let msg = try {
    ignore(@runtime.Session::from_bytes(junk))
    ""
  } catch {
    e => Show::to_string(e)
  }
  assert_true(msg.has_prefix("decode error"), msg~)
}

///|
test "unsupported operators, domains, attributes, dtypes and versions" {
  expect_load_error("unsupported_op", ["unsupported", "'erf_node'", "Erf"])
  expect_load_error("custom_domain", ["'fused'", "domain 'com.example'"])
  expect_load_error("unknown_attribute", ["'relu_bad'", "attribute 'flavour'"])
  expect_load_error("float16_input", ["graph input 'h'", "FLOAT16"])
  expect_load_error("opset_too_new", ["opset 30", "newest supported opset 25"])
  expect_load_error("legacy_version", ["'old_relu'", "Relu-1"])
  expect_load_error("bad_auto_pad", ["'pool'", "auto_pad 'WEIRD'"])
  expect_load_error("bn_training", ["'bn'", "training_mode=1"])
  expect_load_error("sequence_input", ["'s'", "sequence"])
}

///|
test "structurally invalid graphs" {
  expect_load_error("undefined_input", ["'lost'", "'nowhere'"])
  expect_load_error("cycle", ["cycle"])
  expect_load_error("duplicate_output", ["'second'", "'y' is already defined"])
  expect_load_error("output_not_produced", ["'z'", "never produced"])
}

///|
test "input mismatches name the input and dimension" {
  let s = @runtime.Session::from_bytes(mnist())
  let name = s.inputs()[0].name()
  let bad_shape = @tensor.Tensor::zeros(F32, [1, 1, 27, 28])
  let msg = try {
    ignore(s.run([(name, bad_shape)]))
    ""
  } catch {
    e => Show::to_string(e)
  }
  assert_true(msg.contains("'\{name}' dimension 2 must be 28, got 27"), msg~)
  let bad_rank = @tensor.Tensor::zeros(F32, [28, 28])
  let msg = try {
    ignore(s.run([(name, bad_rank)]))
    ""
  } catch {
    e => Show::to_string(e)
  }
  assert_true(msg.contains("must have rank 4"), msg~)
  let bad_type = @tensor.Tensor::zeros(F64, [1, 1, 28, 28])
  let msg = try {
    ignore(s.run([(name, bad_type)]))
    ""
  } catch {
    e => Show::to_string(e)
  }
  assert_true(msg.contains("must be float32, got float64"), msg~)
  let msg = try {
    ignore(s.run([]))
    ""
  } catch {
    e => Show::to_string(e)
  }
  assert_true(msg.contains("missing input '\{name}'"), msg~)
  let ok = @tensor.Tensor::zeros(F32, [1, 1, 28, 28])
  let msg = try {
    ignore(s.run([(name, ok), ("extra", ok)]))
    ""
  } catch {
    e => Show::to_string(e)
  }
  assert_true(msg.contains("'extra' is not an input"), msg~)
  // symbolic dimension 'N' must agree across inputs
  let p = @tensor.Tensor::zeros(F32, [2, 3])
  let q = @tensor.Tensor::zeros(F32, [4, 3])
  let msg = run_error("symbolic_pair", [("p", p), ("q", q)])
  assert_true(msg.contains("'N'"), msg~)
  assert_eq(run_error("symbolic_pair", [("p", p), ("q", p)]), "")
}

///|
test "runtime failures name the node" {
  let x = @tensor.Tensor::zeros(F32, [2, 3])
  let msg = run_error("reshape_runtime", [
    ("x", x),
    ("shape", @tensor.Tensor::i64([2], [4L, 4L])),
  ])
  assert_true(msg.has_prefix("runtime error: node #0 (Reshape-14)"), msg~)
  let msg = run_error("gather_out_of_range", [
    ("x", x),
    ("idx", @tensor.Tensor::i64([1], [5L])),
  ])
  assert_true(
    msg.contains("node #0 (Gather-13)") && msg.contains("index 5"),
    msg~,
  )
  let a = @tensor.Tensor::i32([2], [1, 2])
  let b = @tensor.Tensor::i32([2], [1, 0])
  let msg = run_error("int_div_zero", [("a", a), ("b", b)])
  assert_true(msg.contains("division by zero"), msg~)
}

///|
test "empty tensors flow through the graph like onnxruntime" {
  let s = @runtime.Session::from_bytes(model_bytes("empty_tensors"))
  let outs = s.run([("e", @tensor.Tensor::zeros(F32, [0, 3]))])
  let sum = outs[0].1
  let t = outs[1].1
  let (sum_shape, sum_vals, t_shape) = empty_expected
  assert_eq(sum.shape(), FixedArray::from_array(sum_shape))
  assert_eq(t.shape(), FixedArray::from_array(t_shape))
  let vals = sum.to_f32_array()
  for i, v in sum_vals {
    assert_eq(vals[i].to_double(), v)
  }
}

///|
test "error messages quoted in the README" {
  inspect(
    load_error("unsupported_op"),
    content="unsupported: node #0 'erf_node' (Erf): operator 'Erf' is not supported",
  )
  inspect(
    load_error("bn_training"),
    content="unsupported: node #0 'bn' (BatchNormalization-15): BatchNormalization with training_mode=1 is not supported",
  )
  let s = @runtime.Session::from_bytes(mnist())
  let msg = try {
    ignore(s.run([("Input3", @tensor.Tensor::zeros(F32, [1, 1, 27, 28]))]))
    ""
  } catch {
    e => Show::to_string(e)
  }
  inspect(
    msg,
    content="invalid input: input 'Input3' dimension 2 must be 28, got 27 (shape [1,1,27,28])",
  )
}
