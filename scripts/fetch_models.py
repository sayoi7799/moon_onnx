"""Download the ONNX Model Zoo models used by the model-level tests.

Source: https://github.com/onnx/models (commit c5cae0f, files in Git LFS).
Each file is checked against the sha256 recorded in the LFS pointer.
  mnist-12            MIT          (CNTK, opset 12)
  squeezenet1.1-7     Apache-2.0   (MXNet, opset 7)
  mobilenetv2-12      Apache-2.0   (PyTorch 1.8, opset 12)
  resnet18-v1-7       Apache-2.0   (MXNet, opset 8)
The .tar.gz archives contain the Model Zoo's sample inputs/outputs
(test_data_set_*), which the tests use as inputs.

Files go to .cache/models/. Run: .cache/venv/Scripts/python scripts/fetch_models.py
"""

import os
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, sha256_file  # noqa: E402

ZOO_COMMIT = "c5cae0f1942c8d7727372c7b1f6b485aa39e4421"
MEDIA = f"https://media.githubusercontent.com/media/onnx/models/{ZOO_COMMIT}/validated/vision/classification/"

MODELS = {
    "mnist-12": ("mnist/model/", "5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd",
                 "a53a59dcaca8804a0f6dfda9a3cf2e082979589391dbde73640b60684f1d24e9"),
    "squeezenet1.1-7": ("squeezenet/model/", "1eeff551a67ae8d565ca33b572fc4b66e3ef357b0eb2863bb9ff47a918cc4088",
                        "95dc7fbafbc7372aabe75555db9e89f05e0715ab7472cf7f8fb751ff54543c3e"),
    "mobilenetv2-12": ("mobilenet/model/", "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5",
                       "5f83b422708a708b592a5fd56d84710038511789df3ae510b84123930c37762d"),
    "resnet18-v1-7": ("resnet/model/", "4e8f8653e7a2222b3904cc3fe8e304cd8b339ce1d05fd24688162f86fb6df52c",
                      "2163225f89c858f120b5376758b752e57139601e967f6c6f614a3d77e9d66505"),
}


def fetch(url, dest, sha):
    if os.path.exists(dest) and sha256_file(dest) == sha:
        return
    print("downloading", url)
    req = urllib.request.Request(url, headers={"User-Agent": "moon_onnx-model-fetch"})
    with urllib.request.urlopen(req) as r, open(dest + ".part", "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    got = sha256_file(dest + ".part")
    if got != sha:
        os.remove(dest + ".part")
        raise RuntimeError(f"sha256 mismatch for {dest}: {got}")
    os.replace(dest + ".part", dest)


def main():
    out = os.path.join(CACHE, "models")
    os.makedirs(out, exist_ok=True)
    for name, (path, model_sha, tar_sha) in MODELS.items():
        fetch(MEDIA + path + name + ".onnx", os.path.join(out, name + ".onnx"), model_sha)
        tar = os.path.join(out, name + ".tar.gz")
        fetch(MEDIA + path + name + ".tar.gz", tar, tar_sha)
        dest = os.path.join(out, name + "-data")
        if not os.path.isdir(dest):
            with tarfile.open(tar) as t:
                members = [m for m in t.getmembers() if m.isfile() and m.name.endswith(".pb")]
                for m in members:
                    m.name = "/".join(m.name.split("/")[-2:])  # test_data_set_N/x.pb
                t.extractall(dest, members=members, filter="data")
        print(name, "ok:", sorted(os.listdir(dest)))


if __name__ == "__main__":
    main()
