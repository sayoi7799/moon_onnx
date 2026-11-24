// Loading and executing an ONNX graph.
//
// load():
//   1. decode the protobuf;
//   2. convert initializers and check graph inputs/outputs;
//   3. topologically sort the nodes (Kahn's algorithm), rejecting undefined
//      inputs, duplicate outputs and cycles;
//   4. prepare every node in order: resolve its operator version, validate
//      attributes and input dtypes, propagate output dtypes. Constant nodes
//      are evaluated immediately so later nodes can validate against them;
//   5. compute, for each intermediate value, the last step that reads it.
// run():
//   bind inputs by name (checking dtype and shape), execute the steps in
//   order and drop every intermediate tensor right after its last use.

///|
pub(all) suberror OnnxError {
  /// The file is not a valid ONNX protobuf.
  DecodeFailed(String)
  /// The model uses an operator, attribute or type that is not supported.
  Unsupported(String)
  /// The model is structurally invalid (undefined value, cycle, ...).
  InvalidModel(String)
  /// The inputs passed to run() do not match the model.
  InvalidInput(String)
  /// A node failed while running.
  RuntimeFailure(String)
}

///|
pub impl Show for OnnxError with fn output(self, logger) {
  let (kind, msg) = match self {
    DecodeFailed(m) => ("decode error", m)
    Unsupported(m) => ("unsupported", m)
    InvalidModel(m) => ("invalid model", m)
    InvalidInput(m) => ("invalid input", m)
    RuntimeFailure(m) => ("runtime error", m)
  }
  logger.write_string("\{kind}: \{msg}")
}

///|
pub fn OnnxError::message(self : OnnxError) -> String {
  match self {
    DecodeFailed(m)
    | Unsupported(m)
    | InvalidModel(m)
    | InvalidInput(m)
    | RuntimeFailure(m) => m
  }
}

///|
/// A dimension of a declared input/output shape.
#derive(Eq, Debug)
pub(all) enum Dim {
  Fixed(Int32)
  Symbolic(String)
  Unknown
}

///|
pub impl Show for Dim with fn output(self, logger) {
  match self {
    Fixed(n) => logger.write_string(n.to_string())
    Symbolic(s) => logger.write_string(s)
    Unknown => logger.write_string("?")
  }
}

///|
/// Name, element type and (optional) declared shape of a graph input/output.
pub struct ValueSpec {
  name : String
  dtype : @tensor.DType
  shape : Array[Dim]?
}

///|
pub fn ValueSpec::name(self : ValueSpec) -> String {
  self.name
}

///|
pub fn ValueSpec::dtype(self : ValueSpec) -> @tensor.DType {
  self.dtype
}

///|
pub fn ValueSpec::shape(self : ValueSpec) -> Array[Dim]? {
  self.shape
}

///|
pub fn ValueSpec::to_string(self : ValueSpec) -> String {
  let s = match self.shape {
    None => "[?]"
    Some(d) => "[" + d.map(fn(x) { Show::to_string(x) }).join(",") + "]"
  }
  "\{self.name}: \{self.dtype}\{s}"
}

///|
struct Step {
  index : Int32
  label : String
  inputs : Array[Int32]
  outputs : Array[Int32]
  kernel : @ops.Kernel
}

///|
pub struct Model {
  opset : Int32
  ir_version : Int64
  producer : String
  inputs : Array[ValueSpec]
  input_slots : Array[Int32]
  outputs : Array[ValueSpec]
  output_slots : Array[Int32]
  /// Initializers and folded Constant outputs, by slot.
  constants : Array[(Int32, @tensor.Tensor)]
  num_slots : Int32
  steps : Array[Step]
  /// Slots to drop after each step.
  release : Array[Array[Int32]]
  node_count : Int32
}

///|
pub fn Model::inputs(self : Model) -> Array[ValueSpec] {
  self.inputs
}

///|
pub fn Model::outputs(self : Model) -> Array[ValueSpec] {
  self.outputs
}

///|
pub fn Model::opset(self : Model) -> Int32 {
  self.opset
}

///|
pub fn Model::ir_version(self : Model) -> Int64 {
  self.ir_version
}

///|
pub fn Model::producer(self : Model) -> String {
  self.producer
}

///|
pub fn Model::node_count(self : Model) -> Int32 {
  self.node_count
}

///|
/// Operator types used by the graph with their resolved versions, e.g. "Conv-11".
pub fn Model::operators(self : Model) -> Array[String] {
  let seen : Map[String, Bool] = Map([])
  let out = []
  for s in self.steps {
    if !seen.contains(s.label) {
      seen[s.label] = true
      out.push(s.label)
    }
  }
  out.sort()
  out
}

///|
fn node_label(i : Int32, n : @proto.NodeProto) -> String {
  if n.name == "" {
    "node #\{i} (\{n.op_type})"
  } else {
    "node #\{i} '\{n.name}' (\{n.op_type})"
  }
}

///|
fn convert_type(
  name : String,
  what : String,
  t : @proto.TypeProto,
) -> (@tensor.DType, Array[Dim]?) raise OnnxError {
  match t {
    TensorType(elem_type~, shape~) => {
      guard @tensor.DType::from_onnx_code(elem_type) is Some(dt) else {
        raise Unsupported(
          "\{what} '\{name}' has element type \{@proto.data_type_name(elem_type)}, which is not supported",
        )
      }
      let dims = shape.map(fn(ds) {
        ds.map(fn(d) {
          match d {
            DimValue(v) =>
              if v >= 0L && v <= 2147483647L {
                Fixed(v.to_int())
              } else {
                Unknown
              }
            DimParam(p) => Symbolic(p)
            DimUnknown => Unknown
          }
        })
      })
      (dt, dims)
    }
    SequenceType =>
      raise Unsupported(
        "\{what} '\{name}' is a sequence; sequences are not supported",
      )
    MapType =>
      raise Unsupported("\{what} '\{name}' is a map; maps are not supported")
    OptionalType =>
      raise Unsupported(
        "\{what} '\{name}' is optional; optional types are not supported",
      )
    SparseTensorType =>
      raise Unsupported(
        "\{what} '\{name}' is a sparse tensor; sparse tensors are not supported",
      )
    OpaqueType =>
      raise Unsupported(
        "\{what} '\{name}' has an opaque type, which is not supported",
      )
    NoType => raise InvalidModel("\{what} '\{name}' has no type")
  }
}

///|
/// Parses and validates a serialized ONNX model.
pub fn load(bytes : BytesView) -> Model raise OnnxError {
  let m = @proto.decode_model(bytes) catch {
    @proto.DecodeError(..) as e => raise DecodeFailed(Show::to_string(e))
  }
  guard m.graph is Some(g) else { raise InvalidModel("the model has no graph") }
  // default-domain opset
  let mut opset = -1
  for o in m.opset_imports {
    if o.domain == "" || o.domain == "ai.onnx" {
      opset = o.version.to_int()
    }
  }
  if opset < 0 {
    if m.opset_imports.length() == 0 && m.ir_version <= 2L {
      opset = 1 // IR v1/v2 models without opset_import use opset 1
    } else {
      raise InvalidModel("the model does not import the default ONNX opset")
    }
  }
  if g.sparse_initializer_count > 0 {
    raise Unsupported(
      "the graph has sparse initializers, which are not supported",
    )
  }
  // value slots
  let slots : Map[String, Int32] = Map([])
  let types : Array[@tensor.DType?] = []
  let new_slot = fn(name : String, t : @tensor.DType?) {
    let id = types.length()
    slots[name] = id
    types.push(t)
    id
  }
  let constants : Array[(Int32, @tensor.Tensor)] = []
  let const_by_slot : Map[Int32, @tensor.Tensor] = Map([])
  for init in g.initializers {
    if init.name == "" {
      raise InvalidModel("an initializer has no name")
    }
    if slots.contains(init.name) {
      raise InvalidModel("initializer '\{init.name}' is defined twice")
    }
    let t = @tensor.from_proto(init) catch {
      @tensor.TensorError(msg) => raise Unsupported("initializer: \{msg}")
    }
    let id = new_slot(init.name, Some(t.dtype()))
    constants.push((id, t))
    const_by_slot[id] = t
  }
  let inputs = []
  let input_slots = []
  for vi in g.inputs {
    if slots.contains(vi.name) {
      // IR < 4 lists initializers as inputs too; they stay constants.
      if const_by_slot.contains(slots[vi.name]) {
        continue
      }
      raise InvalidModel("graph input '\{vi.name}' is declared twice")
    }
    let (dt, shape) = convert_type(vi.name, "graph input", vi.value_type)
    let id = new_slot(vi.name, Some(dt))
    inputs.push({ name: vi.name, dtype: dt, shape, })
    input_slots.push(id)
  }
  // producers for topological sort
  let nodes = g.nodes
  let producer : Map[String, Int32] = Map([])
  for i, n in nodes {
    for o in n.outputs {
      if o == "" {
        continue
      }
      if slots.contains(o) || producer.contains(o) {
        raise InvalidModel(
          "\{node_label(i, n)}: output '\{o}' is already defined",
        )
      }
      producer[o] = i
    }
  }
  // Kahn's algorithm over node dependencies
  let indegree = FixedArray::make(nodes.length(), 0)
  let consumers : Array[Array[Int32]] = Array::makei(nodes.length(), fn(_) {
    []
  })
  for i, n in nodes {
    for inp in n.inputs {
      if inp == "" {
        continue
      }
      match producer.get(inp) {
        Some(p) => {
          indegree[i] += 1
          consumers[p].push(i)
        }
        None =>
          if !slots.contains(inp) {
            raise InvalidModel(
              "\{node_label(i, n)}: input '\{inp}' is not a graph input, initializer or node output",
            )
          }
      }
    }
  }
  let order = []
  let ready = []
  for i in 0..<nodes.length() {
    if indegree[i] == 0 {
      ready.push(i)
    }
  }
  // keep file order among ready nodes: pop the smallest index
  while ready.length() > 0 {
    let mut best = 0
    for j in 1..<ready.length() {
      if ready[j] < ready[best] {
        best = j
      }
    }
    let i = ready.remove(best)
    order.push(i)
    for c in consumers[i] {
      indegree[c] -= 1
      if indegree[c] == 0 {
        ready.push(c)
      }
    }
  }
  if order.length() != nodes.length() {
    let stuck = []
    for i in 0..<nodes.length() {
      if indegree[i] > 0 && stuck.length() < 3 {
        stuck.push(node_label(i, nodes[i]))
      }
    }
    raise InvalidModel("the graph has a cycle involving \{stuck.join(", ")}")
  }
  // prepare nodes
  let steps = []
  for i in order {
    let n = nodes[i]
    let label = node_label(i, n)
    if n.domain != "" && n.domain != "ai.onnx" {
      raise Unsupported(
        "\{label}: operator domain '\{n.domain}' is not supported (only the default ONNX domain)",
      )
    }
    let since = @ops.resolve_version(n.op_type, opset) catch {
      e => raise Unsupported("\{label}: \{e}")
    }
    let attrs = @ops.Attrs::from_proto(n.attributes) catch {
      e => raise Unsupported("\{label}: \{e}")
    }
    let present = n.inputs.map(fn(s) { s != "" })
    let in_ids = n.inputs.map(fn(s) { if s == "" { -1 } else { slots[s] } })
    let in_types = in_ids.map(fn(id) { if id < 0 { None } else { types[id] } })
    let consts = in_ids.map(fn(id) {
      if id < 0 {
        None
      } else {
        const_by_slot.get(id)
      }
    })
    let info = @ops.NodeInfo::new(
      n.op_type,
      since,
      attrs,
      present,
      in_types,
      consts,
      n.outputs.length(),
    )
    let kernel = @ops.prepare(info) catch {
      e => raise Unsupported("\{label} (\{n.op_type}-\{since}): \{e}")
    }
    let out_ids = []
    for k, o in n.outputs {
      if o == "" {
        out_ids.push(-1)
      } else {
        out_ids.push(new_slot(o, Some(kernel.output_types[k])))
      }
    }
    if n.op_type == "Constant" {
      let vals = (kernel.run)([]) catch {
        e => raise InvalidModel("\{label}: \{e}")
      }
      if out_ids[0] >= 0 {
        constants.push((out_ids[0], vals[0]))
        const_by_slot[out_ids[0]] = vals[0]
      }
      continue
    }
    steps.push({
      index: i,
      label: "\{n.op_type}-\{since}",
      inputs: in_ids,
      outputs: out_ids,
      kernel,
    })
  }
  // graph outputs
  let outputs = []
  let output_slots = []
  for vi in g.outputs {
    guard slots.get(vi.name) is Some(id) else {
      raise InvalidModel("graph output '\{vi.name}' is never produced")
    }
    let (dt, shape) = match vi.value_type {
      NoType => (types[id].unwrap(), None)
      t => convert_type(vi.name, "graph output", t)
    }
    if types[id] is Some(actual) && actual != dt {
      raise InvalidModel(
        "graph output '\{vi.name}' is declared as \{dt} but computed as \{actual}",
      )
    }
    outputs.push({ name: vi.name, dtype: dt, shape, })
    output_slots.push(id)
  }
  // liveness: release each intermediate after its last reader
  let num_slots = types.length()
  let last_use = FixedArray::make(num_slots, -1)
  for si, s in steps {
    for id in s.inputs {
      if id >= 0 {
        last_use[id] = si
      }
    }
  }
  let keep = FixedArray::make(num_slots, false)
  for id in output_slots {
    keep[id] = true
  }
  for c in constants {
    keep[c.0] = true
  }
  let release : Array[Array[Int32]] = Array::makei(steps.length(), fn(_) { [] })
  for id in 0..<num_slots {
    if !keep[id] && last_use[id] >= 0 {
      release[last_use[id]].push(id)
    }
  }
  // outputs of a step that nobody reads can be dropped right away
  for si, s in steps {
    for id in s.outputs {
      if id >= 0 && !keep[id] && last_use[id] < 0 {
        release[si].push(id)
      }
    }
  }
  {
    opset,
    ir_version: m.ir_version,
    producer: if m.producer_version == "" {
      m.producer_name
    } else {
      "\{m.producer_name} \{m.producer_version}"
    },
    inputs,
    input_slots,
    outputs,
    output_slots,
    constants,
    num_slots,
    steps,
    release,
    node_count: nodes.length(),
  }
}

///|
fn check_input(
  spec : ValueSpec,
  t : @tensor.Tensor,
  bound : Map[String, Int32],
) -> Unit raise OnnxError {
  if t.dtype() != spec.dtype {
    raise InvalidInput(
      "input '\{spec.name}' must be \{spec.dtype}, got \{t.dtype()}",
    )
  }
  guard spec.shape is Some(dims) else { return }
  let shape = t.shape()
  if dims.length() != shape.length() {
    raise InvalidInput(
      "input '\{spec.name}' must have rank \{dims.length()} (\{spec.to_string()}), got shape \{@tensor.shape_to_string(shape)}",
    )
  }
  for i, d in dims {
    match d {
      Fixed(n) =>
        if shape[i] != n {
          raise InvalidInput(
            "input '\{spec.name}' dimension \{i} must be \{n}, got \{shape[i]} (shape \{@tensor.shape_to_string(shape)})",
          )
        }
      Symbolic(p) =>
        match bound.get(p) {
          Some(v) =>
            if v != shape[i] {
              raise InvalidInput(
                "input '\{spec.name}' dimension \{i} ('\{p}') is \{shape[i]}, but '\{p}' is \{v} in another input",
              )
            }
          None => bound[p] = shape[i]
        }
      Unknown => ()
    }
  }
}

///|
/// Runs the model. `feeds` must contain every graph input exactly once.
/// Returns the graph outputs in declaration order.
pub fn Model::run(
  self : Model,
  feeds : Array[(String, @tensor.Tensor)],
) -> Array[(String, @tensor.Tensor)] raise OnnxError {
  let values : FixedArray[@tensor.Tensor?] = FixedArray::make(
    self.num_slots,
    None,
  )
  for c in self.constants {
    values[c.0] = Some(c.1)
  }
  let given : Map[String, @tensor.Tensor] = Map([])
  for f in feeds {
    if given.contains(f.0) {
      raise InvalidInput("input '\{f.0}' is given twice")
    }
    given[f.0] = f.1
  }
  let bound : Map[String, Int32] = Map([])
  for i, spec in self.inputs {
    guard given.get(spec.name) is Some(t) else {
      raise InvalidInput("missing input '\{spec.name}' (\{spec.to_string()})")
    }
    check_input(spec, t, bound)
    values[self.input_slots[i]] = Some(t)
    given.remove(spec.name)
  }
  for name, _ in given {
    let names = self.inputs.map(fn(s) { s.name }).join(", ")
    raise InvalidInput(
      "'\{name}' is not an input of this model (inputs: \{names})",
    )
  }
  for si, step in self.steps {
    let args = step.inputs.map(fn(id) { if id < 0 { None } else { values[id] } })
    let outs = (step.kernel.run)(args) catch {
      e => raise RuntimeFailure("node #\{step.index} (\{step.label}): \{e}")
    }
    for k, id in step.outputs {
      if id >= 0 {
        guard k < outs.length() else {
          raise RuntimeFailure(
            "node #\{step.index} (\{step.label}): output \{k} missing",
          )
        }
        if outs[k].dtype() != step.kernel.output_types[k] {
          raise RuntimeFailure(
            "node #\{step.index} (\{step.label}): output \{k} has type \{outs[k].dtype()}, expected \{step.kernel.output_types[k]}",
          )
        }
        values[id] = Some(outs[k])
      }
    }
    for id in self.release[si] {
      values[id] = None
    }
  }
  let result = []
  for i, spec in self.outputs {
    guard values[self.output_slots[i]] is Some(t) else {
      raise RuntimeFailure("output '\{spec.name}' was not computed")
    }
    result.push((spec.name, t))
  }
  result
}
