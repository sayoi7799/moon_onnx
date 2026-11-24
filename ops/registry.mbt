// Operator registry.
//
// Loading a node goes through `prepare`: it resolves the operator version
// from the model's opset, checks arity and attributes against the ONNX
// schema table, validates input dtypes, and returns a `Kernel` - a closure
// with all attributes already parsed - plus the output dtypes. Everything
// that can be rejected without running the model is rejected here.

///|
pub(all) suberror OpError {
  OpError(String)
}

///|
pub impl Show for OpError with fn output(self, logger) {
  let OpError(msg) = self
  logger.write_string(msg)
}

///|
/// What the loader knows about a node before running it.
pub struct NodeInfo {
  op_type : String
  /// The resolved since_version of the operator.
  since : Int32
  attrs : Attrs
  /// One entry per declared input; false for omitted optional inputs ("").
  present : Array[Bool]
  /// dtype of each present input (None for omitted inputs).
  input_types : Array[@tensor.DType?]
  /// Values of inputs that are initializers or Constant outputs, if known.
  constants : Array[@tensor.Tensor?]
  num_outputs : Int32
}

///|
pub fn NodeInfo::new(
  op_type : String,
  since : Int32,
  attrs : Attrs,
  present : Array[Bool],
  input_types : Array[@tensor.DType?],
  constants : Array[@tensor.Tensor?],
  num_outputs : Int32,
) -> NodeInfo {
  { op_type, since, attrs, present, input_types, constants, num_outputs, }
}

///|
pub fn NodeInfo::has_input(self : NodeInfo, i : Int32) -> Bool {
  i < self.present.length() && self.present[i]
}

///|
pub fn NodeInfo::input_type(self : NodeInfo, i : Int32) -> @tensor.DType raise {
  guard i < self.input_types.length() && self.input_types[i] is Some(t) else {
    raise OpError("input \{i} has unknown element type")
  }
  t
}

///|
pub fn NodeInfo::constant(self : NodeInfo, i : Int32) -> @tensor.Tensor? {
  if i < self.constants.length() {
    self.constants[i]
  } else {
    None
  }
}

///|
/// A prepared operator: output dtypes plus the compute function.
/// `run` receives one entry per declared input (None for omitted optional
/// inputs) and returns one tensor per declared output.
pub struct Kernel {
  output_types : Array[@tensor.DType]
  run : (Array[@tensor.Tensor?]) -> Array[@tensor.Tensor] raise Error
}

///|
pub fn Kernel::new(
  output_types : Array[@tensor.DType],
  run : (Array[@tensor.Tensor?]) -> Array[@tensor.Tensor] raise Error,
) -> Kernel {
  { output_types, run, }
}

///|
priv struct Schema {
  since : Int32
  attrs : Array[String]
  required : Array[String]
  min_inputs : Int32
  max_inputs : Int32
  min_outputs : Int32
  max_outputs : Int32
}

///|
let schemas : Map[String, Array[Schema]] = {
  let m : Map[String, Array[Schema]] = Map([])
  for row in schema_table {
    let (
      op,
      since,
      attrs,
      required,
      min_inputs,
      max_inputs,
      min_outputs,
      max_outputs,
    ) = row
    let s = {
      since,
      attrs,
      required,
      min_inputs,
      max_inputs,
      min_outputs,
      max_outputs,
    }
    match m.get(op) {
      Some(list) => list.push(s)
      None => m[op] = [s]
    }
  }
  m
}

///|
/// Oldest since_version implemented for each operator. Older versions
/// carry legacy semantics (consumed_inputs, broadcast flags, ...) that the
/// MVP does not implement and are rejected.
let min_supported_since : Map[String, Int32] = {
  "Conv": 1,
  "MaxPool": 1,
  "AveragePool": 1,
  "GlobalAveragePool": 1,
  "BatchNormalization": 6,
  "Relu": 6,
  "LeakyRelu": 6,
  "Clip": 6,
  "Sigmoid": 6,
  "Tanh": 6,
  "Softmax": 1,
  "LogSoftmax": 1,
  "Gemm": 6,
  "MatMul": 1,
  "Add": 6,
  "Sub": 6,
  "Mul": 6,
  "Div": 6,
  "Pow": 7,
  "Sqrt": 6,
  "Exp": 6,
  "Log": 6,
  "Neg": 6,
  "Abs": 6,
  "Min": 6,
  "Max": 6,
  "Reshape": 5,
  "Flatten": 1,
  "Transpose": 1,
  "Concat": 4,
  "Squeeze": 1,
  "Unsqueeze": 1,
  "Slice": 1,
  "Gather": 1,
  "Shape": 1,
  "Cast": 6,
  "Pad": 2,
  "Identity": 1,
  "Constant": 1,
  "ConstantOfShape": 9,
  "Expand": 8,
  "ReduceMean": 1,
  "ReduceSum": 1,
  "ReduceMax": 1,
  "ArgMax": 1,
  "Dropout": 7,
}

///|
/// Names of all supported operators (default ONNX domain).
pub fn supported_ops() -> Array[String] {
  let out = []
  for k, _ in min_supported_since {
    out.push(k)
  }
  out.sort()
  out
}

///|
/// Resolves the since_version used by a model whose default-domain opset is
/// `opset`: the largest since_version <= opset.
pub fn resolve_version(op_type : String, opset : Int32) -> Int32 raise {
  guard schemas.get(op_type) is Some(list) else {
    raise OpError("operator '\{op_type}' is not supported")
  }
  if opset > max_supported_opset {
    raise OpError(
      "opset \{opset} is newer than the newest supported opset \{max_supported_opset}",
    )
  }
  let mut best = -1
  for s in list {
    if s.since <= opset && s.since > best {
      best = s.since
    }
  }
  if best < 0 {
    raise OpError("operator '\{op_type}' does not exist in opset \{opset}")
  }
  let min = min_supported_since.get(op_type).unwrap_or(1)
  if best < min {
    raise OpError(
      "\{op_type}-\{best} (opset \{opset}) is not supported; versions from \{op_type}-\{min} are",
    )
  }
  best
}

///|
fn find_schema(op_type : String, since : Int32) -> Schema raise {
  guard schemas.get(op_type) is Some(list) else {
    raise OpError("operator '\{op_type}' is not supported")
  }
  for s in list {
    if s.since == since {
      return s
    }
  }
  raise OpError("\{op_type} has no version \{since}")
}

///|
fn check_schema(info : NodeInfo) -> Unit raise {
  let s = find_schema(info.op_type, info.since)
  let n_in = info.present.length()
  if n_in < s.min_inputs || n_in > s.max_inputs {
    let want = if s.max_inputs >= 1048576 {
      "at least \{s.min_inputs}"
    } else if s.min_inputs == s.max_inputs {
      "\{s.min_inputs}"
    } else {
      "\{s.min_inputs} to \{s.max_inputs}"
    }
    raise OpError("expects \{want} inputs, got \{n_in}")
  }
  for i in 0..<s.min_inputs {
    if i < n_in && !info.present[i] && s.max_inputs < 1048576 {
      raise OpError("required input \{i} is missing")
    }
  }
  if info.num_outputs < s.min_outputs || info.num_outputs > s.max_outputs {
    raise OpError(
      "expects \{s.min_outputs} to \{s.max_outputs} outputs, got \{info.num_outputs}",
    )
  }
  for name in info.attrs.names() {
    if !s.attrs.contains(name) {
      raise OpError(
        "attribute '\{name}' is not defined for \{info.op_type}-\{info.since}",
      )
    }
  }
  for name in s.required {
    if !info.attrs.has(name) {
      raise OpError("required attribute '\{name}' is missing")
    }
  }
}

///|
/// Prepares a node for execution. Raises OpError if anything about the node
/// (version, attributes, input types) is unsupported.
pub fn prepare(info : NodeInfo) -> Kernel raise {
  check_schema(info)
  let k = match info.op_type {
    "Conv" => prepare_conv(info)
    "MaxPool" => prepare_maxpool(info)
    "AveragePool" => prepare_averagepool(info)
    "GlobalAveragePool" => prepare_global_average_pool(info)
    "BatchNormalization" => prepare_batchnorm(info)
    "Relu"
    | "LeakyRelu"
    | "Sigmoid"
    | "Tanh"
    | "Sqrt"
    | "Exp"
    | "Log"
    | "Neg"
    | "Abs" => prepare_unary(info)
    "Clip" => prepare_clip(info)
    "Softmax" | "LogSoftmax" => prepare_softmax(info)
    "Gemm" => prepare_gemm(info)
    "MatMul" => prepare_matmul(info)
    "Add" | "Sub" | "Mul" | "Div" => prepare_arith(info)
    "Pow" => prepare_pow(info)
    "Min" | "Max" => prepare_minmax(info)
    "Reshape" => prepare_reshape(info)
    "Flatten" => prepare_flatten(info)
    "Transpose" => prepare_transpose(info)
    "Concat" => prepare_concat(info)
    "Squeeze" => prepare_squeeze(info)
    "Unsqueeze" => prepare_unsqueeze(info)
    "Slice" => prepare_slice(info)
    "Gather" => prepare_gather(info)
    "Shape" => prepare_shape(info)
    "Cast" => prepare_cast(info)
    "Pad" => prepare_pad(info)
    "Identity" => prepare_identity(info)
    "Dropout" => prepare_dropout(info)
    "Constant" => prepare_constant(info)
    "ConstantOfShape" => prepare_constant_of_shape(info)
    "Expand" => prepare_expand(info)
    "ReduceMean" | "ReduceSum" | "ReduceMax" => prepare_reduce(info)
    "ArgMax" => prepare_argmax(info)
    op => raise OpError("operator '\{op}' is not supported")
  }
  if k.output_types.length() < info.num_outputs {
    raise OpError(
      "\{info.op_type}: \{info.num_outputs} outputs requested, only \{k.output_types.length()} supported",
    )
  }
  k
}

///|
/// Checks that input `i` has one of the allowed dtypes.
fn expect_type(
  info : NodeInfo,
  i : Int32,
  allowed : Array[@tensor.DType],
) -> @tensor.DType raise {
  let t = info.input_type(i)
  if !allowed.contains(t) {
    let names = allowed.map(fn(d) { d.name() }).join(", ")
    raise OpError(
      "\{info.op_type}-\{info.since}: input \{i} has type \{t}; supported: \{names}",
    )
  }
  t
}

///|
fn same_type(info : NodeInfo, i : Int32, t : @tensor.DType) -> Unit raise {
  let u = info.input_type(i)
  if u != t {
    raise OpError(
      "\{info.op_type}: input \{i} has type \{u}, expected \{t} like input 0",
    )
  }
}

///|
let float_types : Array[@tensor.DType] = [F32, F64]

///|
let numeric_types : Array[@tensor.DType] = [F32, F64, I32, I64, U8]

///|
let signed_types : Array[@tensor.DType] = [F32, F64, I32, I64]

///|
let all_types : Array[@tensor.DType] = [F32, F64, I32, I64, U8, Bool]

///|
let index_types : Array[@tensor.DType] = [I32, I64]

///|
/// Fetches a required runtime input.
fn arg(inputs : Array[@tensor.Tensor?], i : Int32) -> @tensor.Tensor raise {
  guard i < inputs.length() && inputs[i] is Some(t) else {
    raise OpError("missing input \{i}")
  }
  t
}

///|
fn opt_arg(inputs : Array[@tensor.Tensor?], i : Int32) -> @tensor.Tensor? {
  if i < inputs.length() {
    inputs[i]
  } else {
    None
  }
}

///|
/// Wraps tensor-level errors raised inside kernels.
fn tensor_new(
  shape : FixedArray[Int32],
  data : @tensor.Data,
) -> @tensor.Tensor raise {
  @tensor.Tensor::new(shape, data) catch {
    @tensor.TensorError(m) => raise OpError(m)
  }
}

///|
fn numel(shape : FixedArray[Int32]) -> Int32 raise {
  @tensor.numel(shape) catch {
    @tensor.TensorError(m) => raise OpError(m)
  }
}

///|
fn norm_axis(axis : Int64, rank : Int32) -> Int32 raise {
  @tensor.normalize_axis(axis, rank) catch {
    @tensor.TensorError(m) => raise OpError(m)
  }
}

///|
fn shape_str(s : FixedArray[Int32]) -> String {
  @tensor.shape_to_string(s)
}
