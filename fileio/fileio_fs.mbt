///|
/// Reads a whole file. Raises if it cannot be read.
pub fn read_file(path : String) -> Bytes raise {
  @fs.read_file_to_bytes(path)
}

///|
pub fn file_exists(path : String) -> Bool {
  @fs.path_exists(path)
}
