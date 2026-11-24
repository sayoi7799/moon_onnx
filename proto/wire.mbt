// Protobuf wire-format reader.
//
// Only the subset of the protobuf encoding used by onnx.proto is needed:
// varint (wire type 0), 64-bit (1), length-delimited (2) and 32-bit (5).
// Groups (wire types 3 and 4) are deprecated and rejected.

///|
/// Error raised for malformed or unsupported protobuf input.
/// `path` names the message field being decoded (e.g. `graph.node[3].attribute[0]`),
/// `offset` is the absolute byte offset in the input.
pub suberror DecodeError {
  DecodeError(path~ : String, offset~ : Int32, msg~ : String)
}

///|
pub impl Show for DecodeError with fn output(self, logger) {
  let DecodeError(path~, offset~, msg~) = self
  logger.write_string(
    "protobuf decode error at byte \{offset} (\{path}): \{msg}",
  )
}

///|
/// A cursor over a length-delimited message.
/// `base` is the absolute offset of `buf[0]` in the original input,
/// so error messages can point at the right byte.
priv struct Reader {
  buf : BytesView
  base : Int32
  mut pos : Int32
  path : String
}

///|
fn Reader::new(buf : BytesView, base : Int32, path : String) -> Reader {
  { buf, base, pos: 0, path, }
}

///|
fn Reader::fail(self : Reader, msg : String) -> DecodeError {
  DecodeError(path=self.path, offset=self.base + self.pos, msg~)
}

///|
fn Reader::at_end(self : Reader) -> Bool {
  self.pos >= self.buf.length()
}

///|
fn Reader::read_varint(self : Reader) -> UInt64 raise DecodeError {
  let mut result : UInt64 = 0
  let mut shift = 0
  let start = self.pos
  while true {
    guard self.pos < self.buf.length() else {
      self.pos = start
      raise self.fail("truncated varint")
    }
    let b = self.buf[self.pos]
    self.pos += 1
    if shift == 63 && b.to_int() > 1 {
      self.pos = start
      raise self.fail("varint overflows 64 bits")
    }
    result = result | ((b.to_uint64() & 0x7F) << shift)
    if (b.to_int() & 0x80) == 0 {
      break
    }
    shift += 7
    if shift > 63 {
      self.pos = start
      raise self.fail("varint longer than 10 bytes")
    }
  }
  result
}

///|
fn Reader::read_fixed32(self : Reader) -> UInt32 raise DecodeError {
  guard self.pos + 4 <= self.buf.length() else {
    raise self.fail("truncated fixed32")
  }
  let p = self.pos
  let b = self.buf
  self.pos += 4
  b[p].to_uint() |
  (b[p + 1].to_uint() << 8) |
  (b[p + 2].to_uint() << 16) |
  (b[p + 3].to_uint() << 24)
}

///|
fn Reader::read_fixed64(self : Reader) -> UInt64 raise DecodeError {
  let lo = self.read_fixed32().to_uint64()
  let hi = self.read_fixed32().to_uint64()
  lo | (hi << 32)
}

///|
/// Reads a length prefix and returns the payload together with its absolute offset.
fn Reader::read_bytes(self : Reader) -> (BytesView, Int32) raise DecodeError {
  let start = self.pos
  let len = self.read_varint()
  let remaining = self.buf.length() - self.pos
  if len > remaining.to_uint64() {
    self.pos = start
    raise self.fail(
      "length-delimited field of \{len} bytes exceeds remaining \{remaining} bytes",
    )
  }
  let n = len.to_int()
  let view = self.buf[self.pos:self.pos + n]
  let abs = self.base + self.pos
  self.pos += n
  (view, abs)
}

///|
/// Reads a field tag; returns (field_number, wire_type).
fn Reader::read_tag(self : Reader) -> (Int32, Int32) raise DecodeError {
  let start = self.pos
  let tag = self.read_varint()
  let field = (tag >> 3).to_int()
  let wire = (tag & 7).to_int()
  if field <= 0 || tag >> 3 > 0x1FFFFFFFUL {
    self.pos = start
    raise self.fail("invalid field number \{tag >> 3}")
  }
  (field, wire)
}

///|
fn Reader::skip(
  self : Reader,
  field : Int32,
  wire : Int32,
) -> Unit raise DecodeError {
  match wire {
    0 => ignore(self.read_varint())
    1 => ignore(self.read_fixed64())
    2 => ignore(self.read_bytes())
    5 => ignore(self.read_fixed32())
    // Deprecated groups: no onnx.proto field uses them, but a conforming
    // parser must skip an unknown group up to its matching END_GROUP tag.
    3 => self.skip_group(field, 0)
    4 => raise self.fail("field \{field}: unexpected END_GROUP")
    _ => raise self.fail("field \{field}: invalid wire type \{wire}")
  }
}

///|
fn Reader::skip_group(
  self : Reader,
  field : Int32,
  depth : Int32,
) -> Unit raise DecodeError {
  if depth > max_depth {
    raise self.fail("groups nested deeper than \{max_depth} levels")
  }
  while true {
    guard !self.at_end() else {
      raise self.fail("field \{field}: group is not terminated")
    }
    let (f, w) = self.read_tag()
    match w {
      4 => {
        guard f == field else {
          raise self.fail("field \{f}: END_GROUP does not match group \{field}")
        }
        return
      }
      3 => self.skip_group(f, depth + 1)
      _ => self.skip(f, w)
    }
  }
}

///|
fn Reader::expect_wire(
  self : Reader,
  field : Int32,
  wire : Int32,
  want : Int32,
) -> Unit raise DecodeError {
  if wire != want {
    raise self.fail("field \{field}: wire type \{wire}, expected \{want}")
  }
}

///|
fn Reader::sub(
  self : Reader,
  field : Int32,
  wire : Int32,
  path : String,
) -> Reader raise DecodeError {
  self.expect_wire(field, wire, 2)
  let (view, abs) = self.read_bytes()
  Reader::new(view, abs, path)
}

///|
fn Reader::read_string(
  self : Reader,
  field : Int32,
  wire : Int32,
) -> String raise DecodeError {
  self.expect_wire(field, wire, 2)
  let start = self.pos
  let (view, _) = self.read_bytes()
  @utf8.decode(view) catch {
    _ => {
      self.pos = start
      raise self.fail("field \{field}: invalid UTF-8 string")
    }
  }
}

///|
fn Reader::read_int64(
  self : Reader,
  field : Int32,
  wire : Int32,
) -> Int64 raise DecodeError {
  self.expect_wire(field, wire, 0)
  self.read_varint().reinterpret_as_int64()
}

///|
/// int32 values are sign-extended to 64 bits on the wire; keep the low 32 bits.
fn Reader::read_int32(
  self : Reader,
  field : Int32,
  wire : Int32,
) -> Int32 raise DecodeError {
  self.expect_wire(field, wire, 0)
  self.read_varint().to_int()
}

///|
/// Repeated scalar fields may be packed (wire type 2) or not (one element per tag).
/// Both encodings must be accepted.
fn Reader::read_repeated_varint(
  self : Reader,
  field : Int32,
  wire : Int32,
  push : (UInt64) -> Unit,
) -> Unit raise DecodeError {
  match wire {
    0 => push(self.read_varint())
    2 => {
      let (view, abs) = self.read_bytes()
      let r = Reader::new(view, abs, self.path)
      while !r.at_end() {
        push(r.read_varint())
      }
    }
    _ =>
      raise self.fail("field \{field}: wire type \{wire} for repeated varint")
  }
}

///|
fn Reader::read_repeated_fixed32(
  self : Reader,
  field : Int32,
  wire : Int32,
  push : (UInt32) -> Unit,
) -> Unit raise DecodeError {
  match wire {
    5 => push(self.read_fixed32())
    2 => {
      let (view, abs) = self.read_bytes()
      if view.length() % 4 != 0 {
        raise DecodeError(
          path=self.path,
          offset=abs,
          msg="field \{field}: packed fixed32 length \{view.length()} is not a multiple of 4",
        )
      }
      let r = Reader::new(view, abs, self.path)
      while !r.at_end() {
        push(r.read_fixed32())
      }
    }
    _ =>
      raise self.fail("field \{field}: wire type \{wire} for repeated fixed32")
  }
}

///|
fn Reader::read_repeated_fixed64(
  self : Reader,
  field : Int32,
  wire : Int32,
  push : (UInt64) -> Unit,
) -> Unit raise DecodeError {
  match wire {
    1 => push(self.read_fixed64())
    2 => {
      let (view, abs) = self.read_bytes()
      if view.length() % 8 != 0 {
        raise DecodeError(
          path=self.path,
          offset=abs,
          msg="field \{field}: packed fixed64 length \{view.length()} is not a multiple of 8",
        )
      }
      let r = Reader::new(view, abs, self.path)
      while !r.at_end() {
        push(r.read_fixed64())
      }
    }
    _ =>
      raise self.fail("field \{field}: wire type \{wire} for repeated fixed64")
  }
}
