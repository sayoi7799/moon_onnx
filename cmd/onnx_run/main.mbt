// onnx_run: classify an image (or a preprocessed tensor) with an ONNX model.
//
//   moon run cmd/onnx_run --target native --release -- MODEL INPUT [options]
//
// INPUT is a binary PPM (.ppm) or BMP (.bmp) image, or a serialized
// TensorProto (.pb) that is fed to the model unchanged. Images are prepared
// from the model's input shape: [N,3,H,W] -> resize short side to 256*H/224,
// centre crop HxW, ImageNet normalization; [N,1,H,W] -> grayscale 0..255.
//
// Options:
//   --labels FILE   one class name per line
//   --repeat N      run N times and report the average and best time
//   --top K         number of classes to print (default 5)
//   --invert        for grayscale models: invert (black digit on white paper)

///|
fn usage() -> Unit {
  println(
    "usage: onnx_run MODEL.onnx INPUT.{ppm,bmp,pb} [--labels FILE] [--repeat N] [--top K] [--invert]",
  )
}

///|
fn now_ms() -> Double {
  @env.now().to_double()
}

///|
fn prepare_input(
  spec : @graph.ValueSpec,
  path : String,
  invert : Bool,
) -> @tensor.Tensor raise {
  let bytes = @fileio.read_file(path)
  if path.has_suffix(".pb") {
    return @tensor.from_proto_bytes(bytes[:])
  }
  let img = @vision.decode_image(bytes)
  let dims = match spec.shape() {
    Some(d) => d
    None =>
      fail(
        "input '\{spec.name()}' has no declared shape; pass a .pb tensor instead",
      )
  }
  let fixed = fn(i : Int32, default : Int32) {
    if i < dims.length() && dims[i] is Fixed(n) {
      n
    } else {
      default
    }
  }
  if dims.length() != 4 {
    fail(
      "input '\{spec.name()}' has rank \{dims.length()}; images need [N, C, H, W]",
    )
  }
  let c = fixed(1, 3)
  let h = fixed(2, 224)
  let w = fixed(3, h)
  if c == 3 {
    let short = h * 256 / 224
    let crop = img.resize_center_crop(short, h)
    ignore(w)
    @vision.imagenet_normalize(crop.to_chw())
  } else if c == 1 {
    img.resize(w, h).to_gray(invert)
  } else {
    fail("cannot build an image input with \{c} channels")
  }
}

///|
fn main {
  let args = @env.args()
  let positional = []
  let mut labels_path = ""
  let mut repeat = 1
  let mut top = 5
  let mut invert = false
  // args[0] is the program; on js it is node followed by the script path
  let mut i = 1
  if args.length() > 1 &&
    (
      args[1].has_suffix(".js") ||
      args[1].has_suffix(".cjs") ||
      args[1].has_suffix(".mjs")
    ) {
    i = 2
  }
  while i < args.length() {
    let a = args[i]
    match a {
      "--labels" if i + 1 < args.length() => {
        labels_path = args[i + 1]
        i += 1
      }
      "--repeat" if i + 1 < args.length() => {
        repeat = @string.parse_int(args[i + 1]) catch { _ => 1 }
        i += 1
      }
      "--top" if i + 1 < args.length() => {
        top = @string.parse_int(args[i + 1]) catch { _ => 5 }
        i += 1
      }
      "--invert" => invert = true
      "-h" | "--help" => {
        usage()
        return
      }
      _ => positional.push(a)
    }
    i += 1
  }
  if positional.length() != 2 {
    usage()
    return
  }
  run(positional[0], positional[1], labels_path, repeat, top, invert) catch {
    e => println("error: \{e}")
  }
}

///|
fn run(
  model_path : String,
  input_path : String,
  labels_path : String,
  repeat : Int32,
  top : Int32,
  invert : Bool,
) -> Unit raise {
  let t0 = now_ms()
  let bytes = @fileio.read_file(model_path)
  let session = @runtime.Session::from_bytes(bytes)
  let t1 = now_ms()
  println(
    "model    \{model_path} (opset \{session.opset()}, \{session.node_count()} nodes, producer '\{session.producer()}')",
  )
  println("operators \{session.operators().join(" ")}")
  for s in session.inputs() {
    println("input    \{s.to_string()}")
  }
  for s in session.outputs() {
    println("output   \{s.to_string()}")
  }
  let specs = session.inputs()
  if specs.length() != 1 {
    fail("this tool supports models with exactly one input")
  }
  let input = prepare_input(specs[0], input_path, invert)
  println("tensor   \{input.to_string_brief()} from \{input_path}")
  let labels = if labels_path == "" {
    []
  } else {
    let text = @utf8.decode(@fileio.read_file(labels_path)[:])
    text.split("\n").map(fn(l) { l.trim_end(chars="\r").to_owned() }).to_array()
  }
  let mut out = input
  let mut total = 0.0
  let mut best = @double.infinity
  for _ in 0..<repeat {
    let s = now_ms()
    out = session.run1(input)
    let d = now_ms() - s
    total = total + d
    if d < best {
      best = d
    }
  }
  let scores = out.to_f32_array()
  let probs = @vision.softmax(scores)
  println("")
  println("top-\{top}:")
  for rank, k in @vision.top_k(scores, top) {
    let name = if k < labels.length() { " " + labels[k] } else { "" }
    let p = (probs[k] * 10000.0).round() / 100.0
    println("  #\{rank + 1}  class \{k}\{name}  score \{scores[k]}  p=\{p}%")
  }
  println("")
  println("load     \{t1 - t0} ms")
  if repeat > 1 {
    println(
      "infer    \{total / repeat.to_double()} ms average over \{repeat} runs, best \{best} ms",
    )
  } else {
    println("infer    \{total} ms")
  }
}
