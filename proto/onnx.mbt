// ONNX protobuf messages (onnx.proto, IR as of onnx v1.22.0).
//
// Only the fields needed for inference are kept. Fields we do not support
// (sparse tensors, functions, training info, ...) are recorded as counts so
// the loader can report them instead of silently ignoring them.
// Unknown field numbers are skipped, as protobuf requires.

///|
/// TensorProto.DataType values.
pub const DT_UNDEFINED : Int32 = 0

///|
pub const DT_FLOAT : Int32 = 1

///|
pub const DT_UINT8 : Int32 = 2

///|
pub const DT_INT8 : Int32 = 3

///|
pub const DT_UINT16 : Int32 = 4

///|
pub const DT_INT16 : Int32 = 5

///|
pub const DT_INT32 : Int32 = 6

///|
pub const DT_INT64 : Int32 = 7

///|
pub const DT_STRING : Int32 = 8

///|
pub const DT_BOOL : Int32 = 9

///|
pub const DT_FLOAT16 : Int32 = 10

///|
pub const DT_DOUBLE : Int32 = 11

///|
pub const DT_UINT32 : Int32 = 12

///|
pub const DT_UINT64 : Int32 = 13

///|
/// Name of a TensorProto.DataType value, as used in the ONNX spec.
pub fn data_type_name(dt : Int32) -> String {
  match dt {
    0 => "UNDEFINED"
    1 => "FLOAT"
    2 => "UINT8"
    3 => "INT8"
    4 => "UINT16"
    5 => "INT16"
    6 => "INT32"
    7 => "INT64"
    8 => "STRING"
    9 => "BOOL"
    10 => "FLOAT16"
    11 => "DOUBLE"
    12 => "UINT32"
    13 => "UINT64"
    14 => "COMPLEX64"
    15 => "COMPLEX128"
    16 => "BFLOAT16"
    17 => "FLOAT8E4M3FN"
    18 => "FLOAT8E4M3FNUZ"
    19 => "FLOAT8E5M2"
    20 => "FLOAT8E5M2FNUZ"
    21 => "UINT4"
    22 => "INT4"
    23 => "FLOAT4E2M1"
    24 => "FLOAT8E8M0"
    25 => "UINT2"
    26 => "INT2"
    _ => "DataType(\{dt})"
  }
}

///|
/// AttributeProto.AttributeType values.
pub const ATTR_UNDEFINED : Int32 = 0

///|
pub const ATTR_FLOAT : Int32 = 1

///|
pub const ATTR_INT : Int32 = 2

///|
pub const ATTR_STRING : Int32 = 3

///|
pub const ATTR_TENSOR : Int32 = 4

///|
pub const ATTR_GRAPH : Int32 = 5

///|
pub const ATTR_FLOATS : Int32 = 6

///|
pub const ATTR_INTS : Int32 = 7

///|
pub const ATTR_STRINGS : Int32 = 8

///|
pub const ATTR_TENSORS : Int32 = 9

///|
pub const ATTR_GRAPHS : Int32 = 10

///|
pub const ATTR_SPARSE_TENSOR : Int32 = 11

///|
pub const ATTR_SPARSE_TENSORS : Int32 = 12

///|
pub const ATTR_TYPE_PROTO : Int32 = 13

///|
pub const ATTR_TYPE_PROTOS : Int32 = 14

///|
pub fn attr_type_name(t : Int32) -> String {
  match t {
    0 => "UNDEFINED"
    1 => "FLOAT"
    2 => "INT"
    3 => "STRING"
    4 => "TENSOR"
    5 => "GRAPH"
    6 => "FLOATS"
    7 => "INTS"
    8 => "STRINGS"
    9 => "TENSORS"
    10 => "GRAPHS"
    11 => "SPARSE_TENSOR"
    12 => "SPARSE_TENSORS"
    13 => "TYPE_PROTO"
    14 => "TYPE_PROTOS"
    _ => "AttributeType(\{t})"
  }
}

///|
#derive(Debug, Eq)
pub(all) struct OperatorSetId {
  domain : String
  version : Int64
}

///|
#derive(Debug, Eq)
pub(all) enum Dim {
  DimValue(Int64)
  DimParam(String)
  DimUnknown
}

///|
/// TypeProto. Only tensor types carry details; the other kinds are
/// recorded so they can be rejected with a clear message.
#derive(Debug, Eq)
pub(all) enum TypeProto {
  TensorType(elem_type~ : Int32, shape~ : Array[Dim]?)
  SequenceType
  MapType
  OptionalType
  SparseTensorType
  OpaqueType
  NoType
}

///|
#derive(Debug)
pub(all) struct ValueInfo {
  name : String
  value_type : TypeProto
}

///|
pub(all) struct TensorProto {
  name : String
  dims : Array[Int64]
  data_type : Int32
  raw_data : BytesView?
  float_data : Array[Float]
  int32_data : Array[Int32]
  int64_data : Array[Int64]
  double_data : Array[Double]
  uint64_data : Array[UInt64]
  string_data_count : Int32
  /// 0 = DEFAULT, 1 = EXTERNAL
  data_location : Int32
  external_data_count : Int32
  has_segment : Bool
}

///|
pub(all) struct AttributeProto {
  name : String
  ref_attr_name : String
  /// AttributeType; inferred from the populated field when the file omits it.
  attr_type : Int32
  f : Float?
  i : Int64?
  s : Bytes?
  t : TensorProto?
  g : GraphProto?
  floats : Array[Float]
  ints : Array[Int64]
  strings : Array[Bytes]
  tensors : Array[TensorProto]
  graphs : Array[GraphProto]
  sparse_tensor_count : Int32
  type_proto_count : Int32
}

///|
pub(all) struct NodeProto {
  inputs : Array[String]
  outputs : Array[String]
  name : String
  op_type : String
  domain : String
  attributes : Array[AttributeProto]
}

///|
pub(all) struct GraphProto {
  name : String
  nodes : Array[NodeProto]
  initializers : Array[TensorProto]
  sparse_initializer_count : Int32
  inputs : Array[ValueInfo]
  outputs : Array[ValueInfo]
  value_infos : Array[ValueInfo]
}

///|
pub(all) struct ModelProto {
  ir_version : Int64
  opset_imports : Array[OperatorSetId]
  producer_name : String
  producer_version : String
  domain : String
  model_version : Int64
  graph : GraphProto?
  function_count : Int32
  training_info_count : Int32
}

///|
/// Graphs nest through attributes (If/Loop bodies); cap the depth so a
/// malicious file cannot overflow the stack.
let max_depth : Int32 = 64

///|
/// Decodes a serialized ModelProto.
pub fn decode_model(bytes : BytesView) -> ModelProto raise DecodeError {
  let r = Reader::new(bytes, 0, "model")
  let mut ir_version = 0L
  let opset_imports = []
  let mut producer_name = ""
  let mut producer_version = ""
  let mut domain = ""
  let mut model_version = 0L
  let mut graph = None
  let mut function_count = 0
  let mut training_info_count = 0
  while !r.at_end() {
    let (field, wire) = r.read_tag()
    guard wire_ok(model_field_kind(field), wire) else {
      r.skip(field, wire)
      continue
    }
    match field {
      1 => ir_version = r.read_int64(field, wire)
      2 => producer_name = r.read_string(field, wire)
      3 => producer_version = r.read_string(field, wire)
      4 => domain = r.read_string(field, wire)
      5 => model_version = r.read_int64(field, wire)
      7 => {
        let sub = r.sub(field, wire, "model.graph")
        graph = Some(decode_graph(sub, 0))
      }
      8 => {
        let path = "model.opset_import[\{opset_imports.length()}]"
        opset_imports.push(decode_opset_id(r.sub(field, wire, path)))
      }
      20 => {
        r.skip(field, wire)
        training_info_count += 1
      }
      25 => {
        r.skip(field, wire)
        function_count += 1
      }
      _ => r.skip(field, wire)
    }
  }
  {
    ir_version,
    opset_imports,
    producer_name,
    producer_version,
    domain,
    model_version,
    graph,
    function_count,
    training_info_count,
  }
}

///|
/// Decodes a serialized TensorProto (the format of ONNX test-data `.pb` files).
pub fn decode_tensor(bytes : BytesView) -> TensorProto raise DecodeError {
  decode_tensor_reader(Reader::new(bytes, 0, "tensor"))
}

///|
fn decode_opset_id(r : Reader) -> OperatorSetId raise DecodeError {
  let mut domain = ""
  let mut version = 0L
  while !r.at_end() {
    let (field, wire) = r.read_tag()
    guard wire_ok(opset_field_kind(field), wire) else {
      r.skip(field, wire)
      continue
    }
    match field {
      1 => domain = r.read_string(field, wire)
      2 => version = r.read_int64(field, wire)
      _ => r.skip(field, wire)
    }
  }
  { domain, version, }
}

///|
fn decode_graph(r : Reader, depth : Int32) -> GraphProto raise DecodeError {
  if depth > max_depth {
    raise r.fail("graphs nested deeper than \{max_depth} levels")
  }
  let mut name = ""
  let nodes = []
  let initializers = []
  let mut sparse_initializer_count = 0
  let inputs = []
  let outputs = []
  let value_infos = []
  let p = r.path
  while !r.at_end() {
    let (field, wire) = r.read_tag()
    guard wire_ok(graph_field_kind(field), wire) else {
      r.skip(field, wire)
      continue
    }
    match field {
      1 => {
        let sub = r.sub(field, wire, "\{p}.node[\{nodes.length()}]")
        nodes.push(decode_node(sub, depth))
      }
      2 => name = r.read_string(field, wire)
      5 => {
        let sub = r.sub(
          field,
          wire,
          "\{p}.initializer[\{initializers.length()}]",
        )
        initializers.push(decode_tensor_reader(sub))
      }
      11 => {
        let sub = r.sub(field, wire, "\{p}.input[\{inputs.length()}]")
        inputs.push(decode_value_info(sub))
      }
      12 => {
        let sub = r.sub(field, wire, "\{p}.output[\{outputs.length()}]")
        outputs.push(decode_value_info(sub))
      }
      13 => {
        let sub = r.sub(field, wire, "\{p}.value_info[\{value_infos.length()}]")
        value_infos.push(decode_value_info(sub))
      }
      15 => {
        r.skip(field, wire)
        sparse_initializer_count += 1
      }
      _ => r.skip(field, wire)
    }
  }
  {
    name,
    nodes,
    initializers,
    sparse_initializer_count,
    inputs,
    outputs,
    value_infos,
  }
}

///|
fn decode_node(r : Reader, depth : Int32) -> NodeProto raise DecodeError {
  let inputs = []
  let outputs = []
  let mut name = ""
  let mut op_type = ""
  let mut domain = ""
  let attributes = []
  while !r.at_end() {
    let (field, wire) = r.read_tag()
    guard wire_ok(node_field_kind(field), wire) else {
      r.skip(field, wire)
      continue
    }
    match field {
      1 => inputs.push(r.read_string(field, wire))
      2 => outputs.push(r.read_string(field, wire))
      3 => name = r.read_string(field, wire)
      4 => op_type = r.read_string(field, wire)
      5 => {
        let sub = r.sub(
          field,
          wire,
          "\{r.path}.attribute[\{attributes.length()}]",
        )
        attributes.push(decode_attribute(sub, depth))
      }
      7 => domain = r.read_string(field, wire)
      _ => r.skip(field, wire)
    }
  }
  { inputs, outputs, name, op_type, domain, attributes, }
}

///|
fn decode_attribute(
  r : Reader,
  depth : Int32,
) -> AttributeProto raise DecodeError {
  let mut name = ""
  let mut ref_attr_name = ""
  let mut attr_type = 0
  let mut f : Float? = None
  let mut i : Int64? = None
  let mut s : Bytes? = None
  let mut t = None
  let mut g = None
  let floats = []
  let ints = []
  let strings = []
  let tensors = []
  let graphs = []
  let mut sparse_tensor_count = 0
  let mut type_proto_count = 0
  let p = r.path
  while !r.at_end() {
    let (field, wire) = r.read_tag()
    guard wire_ok(attr_field_kind(field), wire) else {
      r.skip(field, wire)
      continue
    }
    match field {
      1 => name = r.read_string(field, wire)
      2 => {
        r.expect_wire(field, wire, 5)
        f = Some(Float::reinterpret_from_uint(r.read_fixed32()))
      }
      3 => i = Some(r.read_int64(field, wire))
      4 => {
        r.expect_wire(field, wire, 2)
        s = Some(r.read_bytes().0.to_owned())
      }
      5 => t = Some(decode_tensor_reader(r.sub(field, wire, "\{p}.t")))
      6 => g = Some(decode_graph(r.sub(field, wire, "\{p}.g"), depth + 1))
      7 =>
        r.read_repeated_fixed32(field, wire, fn(u) {
          floats.push(Float::reinterpret_from_uint(u))
        })
      8 =>
        r.read_repeated_varint(field, wire, fn(u) {
          ints.push(u.reinterpret_as_int64())
        })
      9 => {
        r.expect_wire(field, wire, 2)
        strings.push(r.read_bytes().0.to_owned())
      }
      10 => {
        let sub = r.sub(field, wire, "\{p}.tensors[\{tensors.length()}]")
        tensors.push(decode_tensor_reader(sub))
      }
      11 => {
        let sub = r.sub(field, wire, "\{p}.graphs[\{graphs.length()}]")
        graphs.push(decode_graph(sub, depth + 1))
      }
      20 => attr_type = r.read_int32(field, wire)
      21 => ref_attr_name = r.read_string(field, wire)
      22 | 23 => {
        r.skip(field, wire)
        sparse_tensor_count += 1
      }
      14 | 15 => {
        r.skip(field, wire)
        type_proto_count += 1
      }
      _ => r.skip(field, wire)
    }
  }
  // Very old files (IR < 0.0.2) omit `type`; infer it from the populated field.
  if attr_type == ATTR_UNDEFINED {
    attr_type = if f is Some(_) {
      ATTR_FLOAT
    } else if i is Some(_) {
      ATTR_INT
    } else if s is Some(_) {
      ATTR_STRING
    } else if t is Some(_) {
      ATTR_TENSOR
    } else if g is Some(_) {
      ATTR_GRAPH
    } else if floats.length() > 0 {
      ATTR_FLOATS
    } else if ints.length() > 0 {
      ATTR_INTS
    } else if strings.length() > 0 {
      ATTR_STRINGS
    } else if tensors.length() > 0 {
      ATTR_TENSORS
    } else if graphs.length() > 0 {
      ATTR_GRAPHS
    } else {
      ATTR_UNDEFINED
    }
  }
  {
    name,
    ref_attr_name,
    attr_type,
    f,
    i,
    s,
    t,
    g,
    floats,
    ints,
    strings,
    tensors,
    graphs,
    sparse_tensor_count,
    type_proto_count,
  }
}

///|
fn decode_tensor_reader(r : Reader) -> TensorProto raise DecodeError {
  let mut name = ""
  let dims = []
  let mut data_type = 0
  let mut raw_data = None
  let float_data = []
  let int32_data = []
  let int64_data = []
  let double_data = []
  let uint64_data = []
  let mut string_data_count = 0
  let mut data_location = 0
  let mut external_data_count = 0
  let mut has_segment = false
  while !r.at_end() {
    let (field, wire) = r.read_tag()
    guard wire_ok(tensor_field_kind(field), wire) else {
      r.skip(field, wire)
      continue
    }
    match field {
      1 =>
        r.read_repeated_varint(field, wire, fn(u) {
          dims.push(u.reinterpret_as_int64())
        })
      2 => data_type = r.read_int32(field, wire)
      3 => {
        r.skip(field, wire)
        has_segment = true
      }
      4 =>
        r.read_repeated_fixed32(field, wire, fn(u) {
          float_data.push(Float::reinterpret_from_uint(u))
        })
      5 =>
        r.read_repeated_varint(field, wire, fn(u) {
          int32_data.push(u.to_int())
        })
      6 => {
        r.expect_wire(field, wire, 2)
        r.skip(field, wire)
        string_data_count += 1
      }
      7 =>
        r.read_repeated_varint(field, wire, fn(u) {
          int64_data.push(u.reinterpret_as_int64())
        })
      8 => name = r.read_string(field, wire)
      9 => {
        r.expect_wire(field, wire, 2)
        raw_data = Some(r.read_bytes().0)
      }
      10 =>
        r.read_repeated_fixed64(field, wire, fn(u) {
          double_data.push(u.reinterpret_as_double())
        })
      11 => r.read_repeated_varint(field, wire, fn(u) { uint64_data.push(u) })
      13 => {
        r.skip(field, wire)
        external_data_count += 1
      }
      14 => data_location = r.read_int32(field, wire)
      _ => r.skip(field, wire)
    }
  }
  {
    name,
    dims,
    data_type,
    raw_data,
    float_data,
    int32_data,
    int64_data,
    double_data,
    uint64_data,
    string_data_count,
    data_location,
    external_data_count,
    has_segment,
  }
}

///|
fn decode_value_info(r : Reader) -> ValueInfo raise DecodeError {
  let mut name = ""
  let mut value_type = NoType
  while !r.at_end() {
    let (field, wire) = r.read_tag()
    guard wire_ok(value_info_field_kind(field), wire) else {
      r.skip(field, wire)
      continue
    }
    match field {
      1 => name = r.read_string(field, wire)
      2 => value_type = decode_type(r.sub(field, wire, "\{r.path}.type"))
      _ => r.skip(field, wire)
    }
  }
  { name, value_type, }
}

///|
fn decode_type(r : Reader) -> TypeProto raise DecodeError {
  let mut result = NoType
  while !r.at_end() {
    let (field, wire) = r.read_tag()
    guard wire_ok(type_field_kind(field), wire) else {
      r.skip(field, wire)
      continue
    }
    match field {
      1 =>
        result = decode_tensor_type(r.sub(field, wire, "\{r.path}.tensor_type"))
      4 => {
        r.skip(field, wire)
        result = SequenceType
      }
      5 => {
        r.skip(field, wire)
        result = MapType
      }
      7 => {
        r.skip(field, wire)
        result = OpaqueType
      }
      8 => {
        r.skip(field, wire)
        result = SparseTensorType
      }
      9 => {
        r.skip(field, wire)
        result = OptionalType
      }
      _ => r.skip(field, wire)
    }
  }
  result
}

///|
fn decode_tensor_type(r : Reader) -> TypeProto raise DecodeError {
  let mut elem_type = 0
  let mut shape = None
  while !r.at_end() {
    let (field, wire) = r.read_tag()
    guard wire_ok(tensor_type_field_kind(field), wire) else {
      r.skip(field, wire)
      continue
    }
    match field {
      1 => elem_type = r.read_int32(field, wire)
      2 => shape = Some(decode_shape(r.sub(field, wire, "\{r.path}.shape")))
      _ => r.skip(field, wire)
    }
  }
  TensorType(elem_type~, shape~)
}

///|
fn decode_shape(r : Reader) -> Array[Dim] raise DecodeError {
  let dims = []
  while !r.at_end() {
    let (field, wire) = r.read_tag()
    guard wire_ok(shape_field_kind(field), wire) else {
      r.skip(field, wire)
      continue
    }
    match field {
      1 => {
        let sub = r.sub(field, wire, "\{r.path}.dim[\{dims.length()}]")
        dims.push(decode_dim(sub))
      }
      _ => r.skip(field, wire)
    }
  }
  dims
}

///|
fn decode_dim(r : Reader) -> Dim raise DecodeError {
  let mut dim = DimUnknown
  while !r.at_end() {
    let (field, wire) = r.read_tag()
    guard wire_ok(dim_field_kind(field), wire) else {
      r.skip(field, wire)
      continue
    }
    match field {
      1 => dim = DimValue(r.read_int64(field, wire))
      2 => dim = DimParam(r.read_string(field, wire))
      _ => r.skip(field, wire)
    }
  }
  dim
}
