///|
test "read a file on every backend" {
  // moon.mod is always present in the module root
  let b = @fileio.read_file("moon.mod")
  assert_true(b.length() > 10)
  assert_true(@fileio.file_exists("moon.mod"))
  assert_true(!@fileio.file_exists("does/not/exist"))
}
