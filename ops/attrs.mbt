// Node attributes, converted from AttributeProto at load time.

///|
pub(all) enum AttrValue {
  AFloat(Float)
  AInt(Int64)
  AString(Bytes)
  ATensor(@tensor.Tensor)
  AFloats(Array[Float])
  AInts(Array[Int64])
  AStrings(Array[Bytes])
  /// Graph, sparse-tensor, type-proto and tensor-list attributes are kept
  /// only by kind so that ops which do not accept them can say so.
  AUnsupported(String)
}

///|
pub fn AttrValue::kind(self : AttrValue) -> String {
  match self {
    AFloat(_) => "FLOAT"
    AInt(_) => "INT"
    AString(_) => "STRING"
    ATensor(_) => "TENSOR"
    AFloats(_) => "FLOATS"
    AInts(_) => "INTS"
    AStrings(_) => "STRINGS"
    AUnsupported(k) => k
  }
}

///|
pub struct Attrs {
  values : Map[String, AttrValue]
}

///|
pub fn Attrs::empty() -> Attrs {
  { values: Map([]), }
}

///|
pub fn Attrs::from_array(pairs : Array[(String, AttrValue)]) -> Attrs {
  let values = Map([])
  for p in pairs {
    values[p.0] = p.1
  }
  { values, }
}

///|
/// Converts protobuf attributes. A TENSOR attribute is decoded into a tensor
/// here, so unsupported tensor contents are reported at load time.
pub fn Attrs::from_proto(attrs : Array[@proto.AttributeProto]) -> Attrs raise {
  let values = Map([])
  for a in attrs {
    if a.ref_attr_name != "" {
      raise OpError(
        "attribute '\{a.name}' refers to function attribute '\{a.ref_attr_name}'; functions are not supported",
      )
    }
    if values.contains(a.name) {
      raise OpError("attribute '\{a.name}' is given twice")
    }
    let v = match a.attr_type {
      @proto.ATTR_FLOAT => AFloat(a.f.unwrap_or(0.0))
      @proto.ATTR_INT => AInt(a.i.unwrap_or(0L))
      @proto.ATTR_STRING => AString(a.s.unwrap_or(b""))
      @proto.ATTR_TENSOR =>
        match a.t {
          Some(t) =>
            ATensor(
              @tensor.from_proto(t) catch {
                @tensor.TensorError(m) =>
                  raise OpError("attribute '\{a.name}': \{m}")
              },
            )
          None =>
            raise OpError("attribute '\{a.name}' has type TENSOR but no tensor")
        }
      @proto.ATTR_FLOATS => AFloats(a.floats)
      @proto.ATTR_INTS => AInts(a.ints)
      @proto.ATTR_STRINGS => AStrings(a.strings)
      t => AUnsupported(@proto.attr_type_name(t))
    }
    values[a.name] = v
  }
  { values, }
}

///|
pub fn Attrs::names(self : Attrs) -> Array[String] {
  let out = []
  for k, _ in self.values {
    out.push(k)
  }
  out
}

///|
pub fn Attrs::has(self : Attrs, name : String) -> Bool {
  self.values.contains(name)
}

///|
fn type_error(name : String, want : String, got : AttrValue) -> OpError {
  OpError("attribute '\{name}' must be \{want}, got \{got.kind()}")
}

///|
pub fn Attrs::int(self : Attrs, name : String, default : Int64) -> Int64 raise {
  match self.values.get(name) {
    None => default
    Some(AInt(v)) => v
    Some(other) => raise type_error(name, "INT", other)
  }
}

///|
pub fn Attrs::int_opt(self : Attrs, name : String) -> Int64? raise {
  match self.values.get(name) {
    None => None
    Some(AInt(v)) => Some(v)
    Some(other) => raise type_error(name, "INT", other)
  }
}

///|
pub fn Attrs::float(
  self : Attrs,
  name : String,
  default : Float,
) -> Float raise {
  match self.values.get(name) {
    None => default
    Some(AFloat(v)) => v
    Some(other) => raise type_error(name, "FLOAT", other)
  }
}

///|
pub fn Attrs::float_opt(self : Attrs, name : String) -> Float? raise {
  match self.values.get(name) {
    None => None
    Some(AFloat(v)) => Some(v)
    Some(other) => raise type_error(name, "FLOAT", other)
  }
}

///|
pub fn Attrs::ints(self : Attrs, name : String) -> Array[Int64]? raise {
  match self.values.get(name) {
    None => None
    Some(AInts(v)) => Some(v)
    Some(other) => raise type_error(name, "INTS", other)
  }
}

///|
pub fn Attrs::floats(self : Attrs, name : String) -> Array[Float]? raise {
  match self.values.get(name) {
    None => None
    Some(AFloats(v)) => Some(v)
    Some(other) => raise type_error(name, "FLOATS", other)
  }
}

///|
pub fn Attrs::string(
  self : Attrs,
  name : String,
  default : String,
) -> String raise {
  match self.values.get(name) {
    None => default
    Some(AString(b)) =>
      @utf8.decode(b[:]) catch {
        _ => raise OpError("attribute '\{name}' is not valid UTF-8")
      }
    Some(other) => raise type_error(name, "STRING", other)
  }
}

///|
pub fn Attrs::tensor(self : Attrs, name : String) -> @tensor.Tensor? raise {
  match self.values.get(name) {
    None => None
    Some(ATensor(t)) => Some(t)
    Some(other) => raise type_error(name, "TENSOR", other)
  }
}

///|
/// Reads an INT attribute that must be 0 or 1.
pub fn Attrs::flag(self : Attrs, name : String, default : Bool) -> Bool raise {
  match self.int(name, if default { 1L } else { 0L }) {
    0L => false
    1L => true
    v => raise OpError("attribute '\{name}' must be 0 or 1, got \{v}")
  }
}
