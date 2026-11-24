///|
extern "js" fn js_read_file(path : String) -> Bytes =
  #|(p) => {
  #|  const fs = globalThis.process.getBuiltinModule("fs");
  #|  const b = fs.readFileSync(p);
  #|  return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  #|}

///|
extern "js" fn js_file_exists(path : String) -> Bool =
  #|(p) => globalThis.process.getBuiltinModule("fs").existsSync(p)

///|
/// Reads a whole file. Raises if it cannot be read.
pub fn read_file(path : String) -> Bytes raise {
  if !js_file_exists(path) {
    fail("file not found: \{path}")
  }
  js_read_file(path)
}

///|
pub fn file_exists(path : String) -> Bool {
  js_file_exists(path)
}
