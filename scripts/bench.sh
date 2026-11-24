#!/usr/bin/env bash
# Inference time of each model on each backend (release builds, 1 thread).
# Usage: bash scripts/bench.sh [repeats]   (models from scripts/fetch_models.py)
set -u
R=${1:-5}
for t in native wasm-gc js; do
  for m in mnist-12 squeezenet1.1-7 mobilenetv2-12 resnet18-v1-7; do
    f=.cache/models/$m.onnx; [ "$m" = mnist-12 ] && f=testdata/models/mnist-12.onnx
    img=examples/chelsea.ppm; [ "$m" = mnist-12 ] && img=examples/digit3.ppm
    line=$(moon run cmd/onnx_run --target $t --release -- "$f" "$img" --repeat "$R" 2>/dev/null | grep -E "^infer|#1 ")
    echo "$t | $m | $(echo "$line" | tr '\n' ' ')"
  done
done
