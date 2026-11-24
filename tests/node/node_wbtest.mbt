///|
test "official ONNX node tests (onnx v1.22.0)" {
  let (stats, failures) = run_suites(node_suites)
  println("NODE-TABLE-BEGIN")
  println(table("operator", stats))
  println("NODE-TABLE-END")
  if failures.length() > 0 {
    fail("\{failures.length()} node tests failed:\n" + failures.join("\n"))
  }
}

///|
test "official ONNX pytorch-converted tests (onnx v1.22.0)" {
  let (stats, failures) = run_suites(pytorch_suites)
  println("PYTORCH-TABLE-BEGIN")
  println(table("module test op", stats))
  println("PYTORCH-TABLE-END")
  if failures.length() > 0 {
    fail(
      "\{failures.length()} pytorch-converted tests failed:\n" +
      failures.join("\n"),
    )
  }
}
