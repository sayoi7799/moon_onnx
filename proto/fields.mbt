// Expected wire types of the onnx.proto fields we decode.
//
// protobuf parsers treat a known field number arriving with an unexpected
// wire type as an unknown field and skip it; so do we (this matches the
// reference Python implementation). Field numbers not listed are unknown.

///|
/// Field kinds: a scalar kind is the wire type itself (0, 1, 2 or 5);
/// repeated numeric fields accept either the element wire type or the
/// packed (length-delimited) encoding.
const REPEATED_VARINT : Int32 = 10

///|
const REPEATED_FIXED64 : Int32 = 11

///|
const REPEATED_FIXED32 : Int32 = 15

///|
fn wire_ok(kind : Int32?, wire : Int32) -> Bool {
  match kind {
    None => false
    Some(REPEATED_VARINT) => wire == 0 || wire == 2
    Some(REPEATED_FIXED64) => wire == 1 || wire == 2
    Some(REPEATED_FIXED32) => wire == 5 || wire == 2
    Some(k) => wire == k
  }
}

///|
fn model_field_kind(field : Int32) -> Int32? {
  match field {
    1 | 5 => Some(0)
    2 | 3 | 4 | 6 | 7 | 8 | 14 | 20 | 25 | 26 => Some(2)
    _ => None
  }
}

///|
fn opset_field_kind(field : Int32) -> Int32? {
  match field {
    1 => Some(2)
    2 => Some(0)
    _ => None
  }
}

///|
fn graph_field_kind(field : Int32) -> Int32? {
  match field {
    1 | 2 | 5 | 10 | 11 | 12 | 13 | 14 | 15 | 16 => Some(2)
    _ => None
  }
}

///|
fn node_field_kind(field : Int32) -> Int32? {
  match field {
    1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 => Some(2)
    _ => None
  }
}

///|
fn attr_field_kind(field : Int32) -> Int32? {
  match field {
    2 => Some(5)
    3 | 20 => Some(0)
    7 => Some(REPEATED_FIXED32)
    8 => Some(REPEATED_VARINT)
    1 | 4 | 5 | 6 | 9 | 10 | 11 | 13 | 14 | 15 | 21 | 22 | 23 => Some(2)
    _ => None
  }
}

///|
fn tensor_field_kind(field : Int32) -> Int32? {
  match field {
    1 | 5 | 7 | 11 => Some(REPEATED_VARINT)
    4 => Some(REPEATED_FIXED32)
    10 => Some(REPEATED_FIXED64)
    2 | 14 => Some(0)
    3 | 6 | 8 | 9 | 12 | 13 | 16 => Some(2)
    _ => None
  }
}

///|
fn value_info_field_kind(field : Int32) -> Int32? {
  match field {
    1 | 2 | 3 | 4 => Some(2)
    _ => None
  }
}

///|
fn type_field_kind(field : Int32) -> Int32? {
  match field {
    1 | 4 | 5 | 6 | 7 | 8 | 9 => Some(2)
    _ => None
  }
}

///|
fn tensor_type_field_kind(field : Int32) -> Int32? {
  match field {
    1 => Some(0)
    2 => Some(2)
    _ => None
  }
}

///|
fn shape_field_kind(field : Int32) -> Int32? {
  match field {
    1 => Some(2)
    _ => None
  }
}

///|
fn dim_field_kind(field : Int32) -> Int32? {
  match field {
    1 => Some(0)
    2 | 3 => Some(2)
    _ => None
  }
}
