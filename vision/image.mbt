// Minimal image I/O for the command-line example: binary PPM (P6) and
// uncompressed 24/32-bit BMP. Images are returned as uint8 HWC RGB.

///|
pub struct Image {
  width : Int32
  height : Int32
  /// row-major RGB, 3 bytes per pixel
  rgb : FixedArray[Byte]
}

///|
pub fn Image::width(self : Image) -> Int32 {
  self.width
}

///|
pub fn Image::height(self : Image) -> Int32 {
  self.height
}

///|
/// Decodes PPM (P6, maxval 255) or BMP (BI_RGB, 24 or 32 bits per pixel).
pub fn decode_image(b : Bytes) -> Image raise {
  if b.length() >= 2 && b[0] == b'P' && b[1] == b'6' {
    decode_ppm(b)
  } else if b.length() >= 2 && b[0] == b'B' && b[1] == b'M' {
    decode_bmp(b)
  } else {
    fail("unsupported image format (expected binary PPM 'P6' or BMP)")
  }
}

///|
fn decode_ppm(b : Bytes) -> Image raise {
  let mut pos = 2
  let fields = []
  while fields.length() < 3 {
    // skip whitespace and comments
    while pos < b.length() {
      let c = b[pos]
      if c == b'#' {
        while pos < b.length() && b[pos] != b'\n' {
          pos += 1
        }
      } else if c == b' ' || c == b'\n' || c == b'\r' || c == b'\t' {
        pos += 1
      } else {
        break
      }
    }
    let mut v = 0
    let start = pos
    while pos < b.length() && b[pos] >= b'0' && b[pos] <= b'9' {
      v = v * 10 + (b[pos].to_int() - '0'.to_int())
      pos += 1
    }
    if pos == start {
      fail("malformed PPM header")
    }
    fields.push(v)
  }
  pos += 1 // single whitespace before the pixel data
  let (w, h, maxval) = (fields[0], fields[1], fields[2])
  if maxval != 255 {
    fail("only 8-bit PPM files are supported (maxval \{maxval})")
  }
  if b.length() - pos < w * h * 3 {
    fail("truncated PPM data")
  }
  {
    width: w,
    height: h,
    rgb: FixedArray::makei(w * h * 3, fn(i) { b[pos + i] }),
  }
}

///|
fn le_u32(b : Bytes, p : Int32) -> Int32 {
  b[p].to_int() |
  (b[p + 1].to_int() << 8) |
  (b[p + 2].to_int() << 16) |
  (b[p + 3].to_int() << 24)
}

///|
fn decode_bmp(b : Bytes) -> Image raise {
  if b.length() < 54 {
    fail("truncated BMP header")
  }
  let offset = le_u32(b, 10)
  let w = le_u32(b, 18)
  let raw_h = le_u32(b, 22)
  let bpp = b[28].to_int() | (b[29].to_int() << 8)
  let compression = le_u32(b, 30)
  if (compression != 0 && compression != 3) || (bpp != 24 && bpp != 32) {
    fail("only uncompressed 24/32-bit BMP files are supported")
  }
  let bottom_up = raw_h > 0
  let h = if bottom_up { raw_h } else { -raw_h }
  let bytes_pp = bpp / 8
  let stride = (w * bytes_pp + 3) / 4 * 4
  if b.length() < offset + stride * h {
    fail("truncated BMP data")
  }
  let rgb = FixedArray::make(w * h * 3, b'\x00')
  for y in 0..<h {
    let src_row = if bottom_up { h - 1 - y } else { y }
    for x in 0..<w {
      let p = offset + src_row * stride + x * bytes_pp
      let o = (y * w + x) * 3
      rgb[o] = b[p + 2]
      rgb[o + 1] = b[p + 1]
      rgb[o + 2] = b[p]
    }
  }
  { width: w, height: h, rgb, }
}

///|
/// Bilinear resize (pixel centres aligned, no antialiasing).
pub fn Image::resize(self : Image, w : Int32, h : Int32) -> Image {
  let out = FixedArray::make(w * h * 3, b'\x00')
  let sx = self.width.to_double() / w.to_double()
  let sy = self.height.to_double() / h.to_double()
  for y in 0..<h {
    let fy = (y.to_double() + 0.5) * sy - 0.5
    let y0 = clamp(fy.floor().to_int(), 0, self.height - 1)
    let y1 = clamp(y0 + 1, 0, self.height - 1)
    let wy = fy - fy.floor()
    for x in 0..<w {
      let fx = (x.to_double() + 0.5) * sx - 0.5
      let x0 = clamp(fx.floor().to_int(), 0, self.width - 1)
      let x1 = clamp(x0 + 1, 0, self.width - 1)
      let wx = fx - fx.floor()
      for c in 0..<3 {
        let p = fn(xx : Int32, yy : Int32) {
          self.rgb[(yy * self.width + xx) * 3 + c].to_int().to_double()
        }
        let v = (p(x0, y0) * (1.0 - wx) + p(x1, y0) * wx) * (1.0 - wy) +
          (p(x0, y1) * (1.0 - wx) + p(x1, y1) * wx) * wy
        out[(y * w + x) * 3 + c] = clamp((v + 0.5).floor().to_int(), 0, 255).to_byte()
      }
    }
  }
  { width: w, height: h, rgb: out, }
}

///|
fn clamp(v : Int32, lo : Int32, hi : Int32) -> Int32 {
  if v < lo {
    lo
  } else if v > hi {
    hi
  } else {
    v
  }
}

///|
/// Resize so the short side is `short`, then centre-crop `size` x `size`
/// (the usual ImageNet evaluation transform: 256 then 224).
pub fn Image::resize_center_crop(
  self : Image,
  short : Int32,
  size : Int32,
) -> Image {
  let s = short.to_double() /
    (if self.width < self.height { self.width } else { self.height }).to_double()
  let w = clamp(
    (self.width.to_double() * s + 0.5).floor().to_int(),
    short,
    1 << 20,
  )
  let h = clamp(
    (self.height.to_double() * s + 0.5).floor().to_int(),
    short,
    1 << 20,
  )
  let r = self.resize(w, h)
  let left = (w - size) / 2
  let top = (h - size) / 2
  let out = FixedArray::make(size * size * 3, b'\x00')
  for y in 0..<size {
    for x in 0..<size {
      for c in 0..<3 {
        out[(y * size + x) * 3 + c] = r.rgb[((top + y) * w + left + x) * 3 + c]
      }
    }
  }
  { width: size, height: size, rgb: out, }
}

///|
/// uint8 CHW tensor [3, H, W].
pub fn Image::to_chw(self : Image) -> @tensor.Tensor raise {
  let hw = self.width * self.height
  let out = FixedArray::make(3 * hw, b'\x00')
  for i in 0..<hw {
    for c in 0..<3 {
      out[c * hw + i] = self.rgb[i * 3 + c]
    }
  }
  @tensor.Tensor::new([3, self.height, self.width], U8(out))
}

///|
/// Grayscale float32 [1, 1, H, W] with values in 0..255 (MNIST-style input;
/// white digit on black, as in the MNIST data set).
pub fn Image::to_gray(self : Image, invert : Bool) -> @tensor.Tensor raise {
  let hw = self.width * self.height
  let out = FixedArray::make(hw, (0.0 : Float))
  for i in 0..<hw {
    let r = self.rgb[i * 3].to_int().to_double()
    let g = self.rgb[i * 3 + 1].to_int().to_double()
    let b = self.rgb[i * 3 + 2].to_int().to_double()
    let v = 0.299 * r + 0.587 * g + 0.114 * b
    out[i] = Float::from_double(if invert { 255.0 - v } else { v })
  }
  @tensor.Tensor::new([1, 1, self.height, self.width], F32(out))
}
