// Runner for the official ONNX backend node tests (data in data_*.mbt).

///|
priv struct NodeCase {
  name : String
  family : String
  /// "pass" or "reject:<reason>"
  expect : String
  model : String
  inputs : Array[String]
  outputs : Array[String]
}

///|
/// Outcome of one test.
priv enum Outcome {
  Passed
  /// Rejected with an error, as expected for an out-of-scope test.
  Rejected(String)
  /// Anything else: wrong values, unexpected error or unexpected success.
  Failed(String)
}

///|
fn b64(s : String) -> Bytes {
  @base64.decode(s, ignore_whitespace=true) catch {
    _ => abort("bad base64 in test data")
  }
}

///|
/// Float comparison used by onnx.backend.test.runner:
/// |actual - expected| <= atol + rtol * |expected|, rtol = 1e-3, atol = 1e-7.
/// NaN must match NaN; infinities must match exactly.
fn close(actual : Double, expected : Double) -> Bool {
  if expected.is_nan() {
    return actual.is_nan()
  }
  if actual == expected {
    return true
  }
  if actual.is_nan() || actual.is_inf() || expected.is_inf() {
    return false
  }
  (actual - expected).abs() <= 1.0e-7 + 1.0e-3 * expected.abs()
}

///|
/// Returns None when `actual` matches `expected`, else a description.
fn compare(actual : @tensor.Tensor, expected : @tensor.Tensor) -> String? {
  if actual.dtype() != expected.dtype() {
    return Some("dtype \{actual.dtype()} != expected \{expected.dtype()}")
  }
  if !@tensor.shapes_equal(actual.shape(), expected.shape()) {
    return Some(
      "shape \{@tensor.shape_to_string(actual.shape())} != expected \{@tensor.shape_to_string(expected.shape())}",
    )
  }
  let n = actual.size()
  let mut bad = -1
  match (actual.data(), expected.data()) {
    (F32(a), F32(e)) =>
      for i in 0..<n {
        if !close(a[i].to_double(), e[i].to_double()) {
          bad = i
          break
        }
      }
    (F64(a), F64(e)) =>
      for i in 0..<n {
        if !close(a[i], e[i]) {
          bad = i
          break
        }
      }
    (I32(a), I32(e)) =>
      for i in 0..<n {
        if a[i] != e[i] {
          bad = i
          break
        }
      }
    (I64(a), I64(e)) =>
      for i in 0..<n {
        if a[i] != e[i] {
          bad = i
          break
        }
      }
    (U8(a), U8(e)) =>
      for i in 0..<n {
        if a[i] != e[i] {
          bad = i
          break
        }
      }
    (Bool(a), Bool(e)) =>
      for i in 0..<n {
        if a[i] != e[i] {
          bad = i
          break
        }
      }
    _ => return Some("dtype mismatch")
  }
  if bad >= 0 {
    Some(
      "element \{bad}: got \{element(actual, bad)}, expected \{element(expected, bad)}",
    )
  } else {
    None
  }
}

///|
fn element(t : @tensor.Tensor, i : Int32) -> String {
  match t.data() {
    F32(a) => a[i].to_string()
    F64(a) => a[i].to_string()
    I32(a) => a[i].to_string()
    I64(a) => a[i].to_string()
    U8(a) => a[i].to_int().to_string()
    Bool(a) => a[i].to_string()
  }
}

///|
fn run_case(c : NodeCase) -> Outcome {
  let expect_pass = c.expect == "pass"
  let result : Result[Array[(String, @tensor.Tensor)], String] = try {
    let session = @runtime.Session::from_bytes(b64(c.model))
    let specs = session.inputs()
    let feeds = []
    // inputs are matched positionally with the graph inputs (as the official runner does)
    for i, spec in specs {
      if i >= c.inputs.length() {
        raise @graph.InvalidInput("test provides \{c.inputs.length()} inputs")
      }
      feeds.push((spec.name(), @tensor.from_proto_bytes(b64(c.inputs[i])[:])))
    }
    Ok(session.run(feeds))
  } catch {
    e => Err(Show::to_string(e))
  }
  match result {
    Err(msg) => if expect_pass { Failed(msg) } else { Rejected(msg) }
    Ok(outs) => {
      if !expect_pass {
        return Failed("expected rejection (\{c.expect}) but the model ran")
      }
      if outs.length() != c.outputs.length() {
        return Failed(
          "\{outs.length()} outputs, expected \{c.outputs.length()}",
        )
      }
      for i, o in outs {
        let expected = @tensor.from_proto_bytes(b64(c.outputs[i])[:]) catch {
          e => return Failed("bad expected output: \{e}")
        }
        if compare(o.1, expected) is Some(msg) {
          return Failed("output \{i} '\{o.0}': \{msg}")
        }
      }
      Passed
    }
  }
}

///|
/// Per-operator statistics.
priv struct Stats {
  family : String
  mut total : Int32
  mut in_scope : Int32
  mut passed : Int32
  mut rejected : Int32
}

///|
/// Runs every case; returns per-family stats (sorted) and failure messages.
fn run_suites(suites : Array[Array[NodeCase]]) -> (Array[Stats], Array[String]) {
  let by : Map[String, Stats] = Map([])
  let failures = []
  for suite in suites {
    for c in suite {
      let s = match by.get(c.family) {
        Some(s) => s
        None => {
          let s = {
            family: c.family,
            total: 0,
            in_scope: 0,
            passed: 0,
            rejected: 0,
          }
          by[c.family] = s
          s
        }
      }
      s.total += 1
      if c.expect == "pass" {
        s.in_scope += 1
      }
      match run_case(c) {
        Passed => s.passed += 1
        // a rejection must be an explicit "not supported" error
        Rejected(msg) =>
          if msg.contains("not supported") || msg.contains("unsupported") {
            s.rejected += 1
          } else {
            failures.push(
              "\{c.name}: rejected for an unexpected reason: \{msg}",
            )
          }
        Failed(msg) => failures.push("\{c.name}: \{msg}")
      }
    }
  }
  let stats = []
  for _, s in by {
    stats.push(s)
  }
  stats.sort_by(fn(a, b) { a.family.compare(b.family) })
  (stats, failures)
}

///|
fn table(title : String, stats : Array[Stats]) -> String {
  let lines = [
    "| \{title} | official tests | in MVP scope | passed | correctly rejected |",
    "|---|---:|---:|---:|---:|",
  ]
  let mut t = 0
  let mut i = 0
  let mut p = 0
  let mut r = 0
  for s in stats {
    lines.push(
      "| \{s.family} | \{s.total} | \{s.in_scope} | \{s.passed} | \{s.rejected} |",
    )
    t += s.total
    i += s.in_scope
    p += s.passed
    r += s.rejected
  }
  lines.push("| **total** | **\{t}** | **\{i}** | **\{p}** | **\{r}** |")
  lines.join("\n")
}
