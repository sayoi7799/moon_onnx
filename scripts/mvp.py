"""The MVP operator and dtype scope (shared by the generators)."""

MVP_OPS = set(
    (
        "Conv MaxPool AveragePool GlobalAveragePool BatchNormalization "
        "Relu LeakyRelu Clip Sigmoid Tanh Softmax LogSoftmax Gemm MatMul "
        "Add Sub Mul Div Pow Sqrt Exp Log Neg Abs Min Max "
        "Reshape Flatten Transpose Concat Squeeze Unsqueeze Slice Gather Shape Cast Pad "
        "Identity Constant ConstantOfShape Expand "
        "ReduceMean ReduceSum ReduceMax ArgMax Dropout"
    ).split()
)

# TensorProto.DataType codes: FLOAT, UINT8, INT32, INT64, BOOL, DOUBLE
MVP_DTYPES = {1, 2, 6, 7, 9, 11}


def test_family(test: str, single_op: dict) -> str | None:
    """Operator a node test belongs to.

    `single_op` maps test name -> op_type for single-node tests (None for
    multi-node graphs). A function-expanded test ("..._expanded[_verN]")
    belongs to the operator of the test it expands, whatever ops its body
    uses (e.g. test_castlike_*_expanded is a CastLike test even though its
    body is a single Cast node).
    """
    if "_expanded" in test:
        base = test.split("_expanded")[0]
        return test_family(base, single_op) if base in single_op else None
    return single_op.get(test)
