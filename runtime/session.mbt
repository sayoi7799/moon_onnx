// Public API: load a model from bytes, inspect its inputs/outputs, run it.
//
//   let session = @runtime.Session::from_bytes(model_bytes)
//   for spec in session.inputs() { println(spec.to_string()) }
//   let outputs = session.run([("input", tensor)])

///|
pub struct Session {
  model : @graph.Model
}

///|
/// Loads and validates a model. Unsupported operators, attributes or data
/// types are reported here, naming the offending node.
pub fn Session::from_bytes(bytes : Bytes) -> Session raise @graph.OnnxError {
  { model: @graph.load(bytes[:]), }
}

///|
pub fn Session::from_bytes_view(
  bytes : BytesView,
) -> Session raise @graph.OnnxError {
  { model: @graph.load(bytes), }
}

///|
/// Graph inputs that must be fed to `run` (initializers excluded).
pub fn Session::inputs(self : Session) -> Array[@graph.ValueSpec] {
  self.model.inputs()
}

///|
pub fn Session::outputs(self : Session) -> Array[@graph.ValueSpec] {
  self.model.outputs()
}

///|
pub fn Session::opset(self : Session) -> Int32 {
  self.model.opset()
}

///|
pub fn Session::producer(self : Session) -> String {
  self.model.producer()
}

///|
pub fn Session::node_count(self : Session) -> Int32 {
  self.model.node_count()
}

///|
/// Operator types (with resolved versions) used by the model.
pub fn Session::operators(self : Session) -> Array[String] {
  self.model.operators()
}

///|
/// Runs inference. Inputs are matched by name; outputs are returned in the
/// order the graph declares them.
pub fn Session::run(
  self : Session,
  feeds : Array[(String, @tensor.Tensor)],
) -> Array[(String, @tensor.Tensor)] raise @graph.OnnxError {
  self.model.run(feeds)
}

///|
/// Convenience for single-input models: feeds `input` to the only input and
/// returns the first output.
pub fn Session::run1(
  self : Session,
  input : @tensor.Tensor,
) -> @tensor.Tensor raise @graph.OnnxError {
  let ins = self.model.inputs()
  if ins.length() != 1 {
    raise @graph.InvalidInput(
      "run1 needs a model with exactly one input, this one has \{ins.length()}",
    )
  }
  let outs = self.model.run([(ins[0].name(), input)])
  outs[0].1
}
