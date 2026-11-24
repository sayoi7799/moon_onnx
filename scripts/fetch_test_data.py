"""Download the official ONNX backend test data used by the tests.

Source: github.com/onnx/onnx at tag v1.22.0 (Apache-2.0), the last release
that commits the generated node tests to the repository:
  onnx/backend/test/data/node/<test>/...           (single-operator tests)
  onnx/backend/test/data/pytorch-converted/<test>/... (PyTorch module tests)

Every downloaded file is verified against its git blob SHA-1 from the
repository tree. Files go to .cache/node and .cache/pytorch-converted.

Only node tests whose operator is in the MVP set are downloaded in full; for
the others only model.onnx is fetched (needed to know the operator).

Run: .cache/venv/Scripts/python scripts/fetch_test_data.py
"""

import concurrent.futures as cf
import hashlib
import json
import os
import sys
import urllib.request

import onnx

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, ONNX_COMMIT  # noqa: E402
from mvp import MVP_OPS, test_family  # noqa: E402

API = "https://api.github.com/repos/onnx/onnx"
RAW = f"https://raw.githubusercontent.com/onnx/onnx/{ONNX_COMMIT}/onnx/backend/test/data/"


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "moon_onnx-test-fetch"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def subtree(name):
    listing = get_json(f"{API}/contents/onnx/backend/test/data?ref={ONNX_COMMIT}")
    sha = next(x["sha"] for x in listing if x["name"] == name)
    tree = get_json(f"{API}/git/trees/{sha}?recursive=1")
    assert not tree["truncated"]
    return [p for p in tree["tree"] if p["type"] == "blob"]


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch(kind, entry):
    dest = os.path.join(CACHE, kind, entry["path"])
    if os.path.exists(dest):
        with open(dest, "rb") as f:
            if blob_sha(f.read()) == entry["sha"]:
                return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(RAW + kind + "/" + entry["path"], headers={"User-Agent": "moon_onnx-test-fetch"})
    with urllib.request.urlopen(req) as r:
        data = r.read()
    if blob_sha(data) != entry["sha"]:
        raise RuntimeError(f"hash mismatch for {kind}/{entry['path']}")
    with open(dest, "wb") as f:
        f.write(data)


def node_op(test_dir):
    """Operator a node test exercises: the single node's op_type, or for a
    multi-node (function-expanded) test, the op of the test it expands."""
    m = onnx.load(os.path.join(test_dir, "model.onnx"))
    ops = {n.op_type for n in m.graph.node}
    if len(m.graph.node) == 1:
        return next(iter(ops))
    return None


def main():
    pool = cf.ThreadPoolExecutor(16)
    node = subtree("node")
    models = [e for e in node if e["path"].endswith("/model.onnx")]
    list(pool.map(lambda e: fetch("node", e), models))
    ops = {}
    for e in models:
        test = e["path"].split("/")[0]
        ops[test] = node_op(os.path.join(CACHE, "node", test))
    family = {t: test_family(t, ops) for t in ops}
    wanted = {t for t, op in family.items() if op in MVP_OPS}
    rest = [e for e in node if e["path"].split("/")[0] in wanted and not e["path"].endswith("/model.onnx")]
    list(pool.map(lambda e: fetch("node", e), rest))
    with open(os.path.join(CACHE, "node_ops.json"), "w") as f:
        json.dump({t: family[t] for t in sorted(wanted)}, f, indent=0)
    print(f"node: {len(models)} tests listed, {len(wanted)} use MVP operators")
    pt = subtree("pytorch-converted")
    list(pool.map(lambda e: fetch("pytorch-converted", e), pt))
    print(f"pytorch-converted: {len(pt)} files")


if __name__ == "__main__":
    main()
