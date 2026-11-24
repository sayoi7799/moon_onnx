// Whole-model tests: outputs must match onnxruntime within
// |ours - ort| <= 1e-4 + 1e-3 * |ort| and the top-5 classes must be
// identical (same classes, same order). See scripts/gen_model_tests.py.
//
// mnist-12 is committed under testdata/; the other models are downloaded by
// scripts/fetch_models.py into .cache/models. Missing models are reported
// and skipped so a fresh clone still passes `moon test`; CI fetches them.

///|
let rtol : Double = 1.0e-3

///|
let atol : Double = 1.0e-4

///|
fn hex8(u : UInt32) -> String {
  let digits = "0123456789abcdef"
  let buf = StringBuilder()
  for i = 7; i >= 0; i = i - 1 {
    buf.write_char(
      digits.get_char(((u >> (i * 4)) & 0xF).reinterpret_as_int()).unwrap(),
    )
  }
  buf.to_string()
}

///|
fn f32_hash(a : FixedArray[Float]) -> String {
  let mut h : UInt32 = 0x811C9DC5
  for v in a {
    let bits = v.reinterpret_as_uint()
    for k in 0..<4 {
      h = h ^ ((bits >> (8 * k)) & 0xFF)
      h = h * 0x01000193
    }
  }
  hex8(h)
}

///|
fn model_file(model : String) -> String {
  if model == "mnist-12" {
    "testdata/models/mnist-12.onnx"
  } else {
    ".cache/models/\{model}.onnx"
  }
}

///|
priv struct ModelReport {
  model : String
  cases : Int32
  max_abs : Double
  max_rel : Double
  top1_same : Int32
  top5_same : Int32
}

///|
fn run_model(model : String) -> ModelReport? raise {
  let path = model_file(model)
  if !@fileio.file_exists(path) {
    println("SKIPPED \{model}: \{path} not found (run scripts/fetch_models.py)")
    return None
  }
  let session = @runtime.Session::from_bytes(@fileio.read_file(path))
  let mut n = 0
  let mut max_abs = 0.0
  let mut max_rel = 0.0
  let mut top1 = 0
  let mut top5 = 0
  for c in model_cases {
    let (m, case_, kind, input_file, expected_file, input_hash, expected_top5) = c
    if m != model {
      continue
    }
    let raw = @tensor.from_proto_bytes(@fileio.read_file(input_file)[:])
    let input = if kind == "image" {
      @vision.imagenet_normalize(raw)
    } else {
      raw
    }
    // the input must be bit-identical to what onnxruntime received
    assert_eq(
      f32_hash(input.to_f32_array()),
      input_hash,
      msg="\{model}/\{case_}: input differs",
    )
    let out = session.run1(input)
    let expected = @tensor.from_proto_bytes(@fileio.read_file(expected_file)[:])
    assert_true(
      @tensor.shapes_equal(out.shape(), expected.shape()),
      msg="\{model}/\{case_}: shape \{@tensor.shape_to_string(out.shape())}",
    )
    let a = out.to_f32_array()
    let e = expected.to_f32_array()
    for i in 0..<a.length() {
      let d = (a[i].to_double() - e[i].to_double()).abs()
      let lim = atol + rtol * e[i].to_double().abs()
      if !(d <= lim) {
        fail(
          "\{model}/\{case_}: element \{i} = \{a[i]}, onnxruntime \{e[i]} (diff \{d})",
        )
      }
      if d > max_abs {
        max_abs = d
      }
      let r = d / (e[i].to_double().abs() + 1.0e-12)
      if r > max_rel && e[i].to_double().abs() > 1.0e-3 {
        max_rel = r
      }
    }
    let ours = @vision.top_k(a, 5)
    assert_eq(
      @vision.top_k(e, 5),
      expected_top5,
      msg="\{model}/\{case_}: manifest",
    )
    assert_eq(
      ours,
      expected_top5,
      msg="\{model}/\{case_}: top-5 differs from onnxruntime",
    )
    n += 1
    top1 += 1
    top5 += 1
  }
  assert_true(n >= 5, msg="\{model}: only \{n} cases")
  Some({ model, cases: n, max_abs, max_rel, top1_same: top1, top5_same: top5, })
}

///|
fn report_line(r : ModelReport) -> String {
  "| \{r.model} | \{r.cases} | \{r.top1_same}/\{r.cases} | \{r.top5_same}/\{r.cases} | \{r.max_abs} | \{r.max_rel} |"
}

///|
test "mnist-12 matches onnxruntime" {
  if run_model("mnist-12") is Some(r) {
    println("MODEL-ROW " + report_line(r))
  }
}

///|
test "squeezenet1.1-7 matches onnxruntime" {
  if run_model("squeezenet1.1-7") is Some(r) {
    println("MODEL-ROW " + report_line(r))
  }
}

///|
test "mobilenetv2-12 matches onnxruntime" {
  if run_model("mobilenetv2-12") is Some(r) {
    println("MODEL-ROW " + report_line(r))
  }
}

///|
test "resnet18-v1-7 matches onnxruntime" {
  if run_model("resnet18-v1-7") is Some(r) {
    println("MODEL-ROW " + report_line(r))
  }
}
