// Image preprocessing helpers for the examples and model tests.

///|
let imagenet_mean : FixedArray[Float] = [
  Float::from_double(0.485),
  Float::from_double(0.456),
  Float::from_double(0.406),
]

///|
let imagenet_std : FixedArray[Float] = [
  Float::from_double(0.229),
  Float::from_double(0.224),
  Float::from_double(0.225),
]

///|
/// uint8 CHW image [3, H, W] -> float32 [1, 3, H, W] with ImageNet
/// normalization: ((x / 255) - mean[c]) / std[c], each step in float32.
pub fn imagenet_normalize(img : @tensor.Tensor) -> @tensor.Tensor raise {
  let s = img.shape()
  guard s.length() == 3 && s[0] == 3 else {
    fail("expected a [3, H, W] image, got \{@tensor.shape_to_string(s)}")
  }
  guard img.data() is U8(px) else { fail("expected a uint8 image") }
  let hw = s[1] * s[2]
  let out = FixedArray::make(3 * hw, (0.0 : Float))
  let scale : Float = 255.0
  for c in 0..<3 {
    for i in 0..<hw {
      let x = Float::from_int(px[c * hw + i].to_int()) / scale
      out[c * hw + i] = (x - imagenet_mean[c]) / imagenet_std[c]
    }
  }
  @tensor.Tensor::new([1, 3, s[1], s[2]], F32(out))
}

///|
/// Indices of the k largest values (ties keep the lower index first).
pub fn top_k(scores : FixedArray[Float], k : Int32) -> Array[Int32] {
  let idx = Array::makei(scores.length(), fn(i) { i })
  idx.sort_by(fn(a, b) {
    if scores[a] > scores[b] {
      -1
    } else if scores[a] < scores[b] {
      1
    } else {
      a.compare(b)
    }
  })
  idx[0:if k < idx.length() { k } else { idx.length() }].to_owned()
}

///|
/// Numerically stable softmax over a vector (for printing probabilities).
pub fn softmax(scores : FixedArray[Float]) -> FixedArray[Double] {
  let mut m = @double.neg_infinity
  for v in scores {
    if v.to_double() > m {
      m = v.to_double()
    }
  }
  let e = FixedArray::makei(scores.length(), fn(i) {
    @math.exp(scores[i].to_double() - m)
  })
  let s = e.fold(init=0.0, fn(a, b) { a + b })
  FixedArray::makei(e.length(), fn(i) { e[i] / s })
}
