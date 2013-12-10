// Copyright 2013 The Servo Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link(name = "png",
       package_id = "png",
       vers = "0.1")];

use std::vec;
use std::cmp::min;
use std::io;
use std::io::File;
use std::iter::range_step_inclusive;
use std::mem::size_of;
use std::num::abs;
use std::str::from_utf8;

use zlib::InflateStream;

mod zlib;

#[deriving(Eq)]
pub enum ColorType {
    K1, K2, K4, K8, K16,
    KA8, KA16,
    Pal1, Pal2, Pal4, Pal8,
    RGB8, RGB16,
    RGBA8, RGBA16,
}

impl ColorType {
    fn is_palette(self) -> bool {
        match self {
            Pal1 | Pal2 | Pal4 | Pal8 => true,
            _ => false
        }
    }

    fn pixel_bits(self) -> uint {
        match self {
            K1 | Pal1 => 1,
            K2 | Pal2 => 2,
            K4 | Pal4 => 4,
            K8 | Pal8 => 8,
            K16 | KA8 => 16,
            RGB8 => 24,
            KA16 | RGBA8 => 32,
            RGB16 => 48,
            RGBA16 => 64
        }
    }
}

pub struct Image {
    width: u32,
    height: u32,
    color_type: ColorType,
    pixels: ~[u8]
}

pub enum ImageState<'a> {
    Partial(Option<&'a Image>),
    Complete(Image),
    Error(~str)
}

static MAGIC: [u8, ..8] = [
    0x89,
    'P' as u8,
    'N' as u8,
    'G' as u8,
    '\r' as u8, // DOS line ending (CR
    '\n' as u8, //                  LF)
    0x1A,       // DOS EOF
    '\n' as u8  // Unix line ending (LF)
];

#[packed]
struct Ihdr {
    width: u32,
    height: u32,
    bits: u8,
    color_type: u8,
    compression_method: u8,
    filter_method: u8,
    interlace_method: u8
}

impl Ihdr {
    fn get_color_type(&self) -> Result<ColorType, ~str> {
        let bits = self.bits;
        let invalid = |name| Err(format!("invalid bit depth {} for color type {} ({:s})",
                                         bits, self.color_type, name));
        Ok(match self.color_type {
            0 => match bits {
                1 => K1,
                2 => K2,
                4 => K4,
                8 => K8,
                16 => K16,
                _ => return invalid("grayscale")
            },
            2 => match bits {
                8 => RGB8,
                16 => RGB16,
                _ => return invalid("truecolor")
            },
            3 => match bits {
                1 => Pal1,
                2 => Pal2,
                4 => Pal4,
                8 => Pal8,
                _ => return invalid("palette")
            },
            4 => match bits {
                8 => KA8,
                16 => KA16,
                _ => return invalid("grayscale with alpha")
            },
            6 => match bits {
                8 => RGBA8,
                16 => RGBA16,
                _ => return invalid("truecolor with alpha")
            },
            _ => return Err(format!("invalid color type {}", self.color_type))
        })
    }

    fn to_image(&self) -> Result<PartialImage, ~str> {
        let color_type = match self.get_color_type() {
            Ok(c) => c,
            Err(m) => return Err(m)
        };

        let color_decoded = match color_type {
            K1 | K2 | K4 | K8| KA8 => KA8, // TODO(eddyb) use K8 or KA8 for non-alpha types?
            K16 | KA16 | RGB16 | RGBA16 => return Err(~"unsupported bit depth of 16"),
            _ => RGBA8
        };

        let pixel_bytes = color_decoded.pixel_bits() / 8;

        let pixel_bits_raw = color_type.pixel_bits();

        if self.compression_method != 0 {
            return Err(format!("unknown compression method {}", self.compression_method));
        }

        if self.filter_method != 0 {
            return Err(format!("unknown filter method {}", self.filter_method));
        }

        if self.interlace_method > 1 {
            return Err(format!("unknown interlace method {}", self.interlace_method));
        }

        let w = self.width as uint;
        let h = self.height as uint;

        let initial_scanline_width = if self.interlace_method == 1 {
            (w + 7) / 8
        } else {
            w
        };

        Ok(PartialImage {
            image: Image {
                width: self.width,
                height: self.height,
                color_type: color_decoded,
                pixels: vec::from_elem(w * h * pixel_bytes, 0u8)
            },
            color_type: color_type,
            interlace: self.interlace_method,
            palette: None,
            transparent_color: None,
            idat_inflate_stream: None,
            x_byte_pos: 0,
            y_byte_pos: 0,
            scanline_bytes: w * pixel_bytes,
            scanline: vec::from_elem((w * pixel_bits_raw + 7) / 8, 0u8),
            scanline_pos: None,
            pixel_prev: [0, ..4],
            pixel_bytes_raw: (pixel_bits_raw + 7) / 8,
            scanline_bytes_raw: (initial_scanline_width * pixel_bits_raw + 7) / 8,
            filter: NoFilter
        })
    }
}

enum Filter {
    NoFilter = 0,
    Sub = 1,
    Up = 2,
    Average = 3,
    Paeth = 4
}

impl Filter {
    fn from_u8(value: u8) -> Result<Filter, ~str> {
        Ok(match value { // TODO(eddyb) automate this.
            0 => NoFilter,
            1 => Sub,
            2 => Up,
            3 => Average,
            4 => Paeth,
            _ => return Err(format!("unknown filter {}", value))
        })
    }
}

struct PartialImage {
    image: Image,
    color_type: ColorType,
    interlace: u8,
    palette: Option<~[u8]>,
    transparent_color: Option<[u16, ..3]>,
    idat_inflate_stream: Option<~InflateStream>,
    x_byte_pos: uint,
    y_byte_pos: uint,
    scanline_bytes: uint,
    scanline: ~[u8], // TODO(eddyb) optionalize?
    scanline_pos: Option<uint>,
    pixel_prev: [u8, ..4],
    pixel_bytes_raw: uint, // FIXME(eddyb) don't waste space.
    scanline_bytes_raw: uint,
    filter: Filter
}

impl PartialImage {
    fn update_idat(&mut self, mut data: &[u8]) -> Result<(), ~str> {
        let mut scanline_pos = self.scanline_pos;
        let mut filter = self.filter;

        while data.len() > 0 {
            let mut i = match scanline_pos {
                Some(pos) => pos,
                None => {
                    match Filter::from_u8(data[0]) {
                        Ok(f) => { filter = f; }
                        Err(m) => return Err(m)
                    }
                    data = data.slice_from(1);
                    0
                }
            };

            let line = data.slice_to(min(self.scanline_bytes_raw - i, data.len()));

            match filter {
                NoFilter => for &x in line.iter() {
                    self.scanline[i] = x;
                    i += 1;
                },
                Sub => for &x in line.iter() {
                    let a = if i < self.pixel_bytes_raw {
                        0
                    } else {
                        self.scanline[i - self.pixel_bytes_raw]
                    };
                    self.scanline[i] = a + x;
                    i += 1;
                },
                Up => for &x in line.iter() {
                    let b = self.scanline[i];
                    self.scanline[i] = b + x;
                    i += 1;
                },
                Average => for &x in line.iter() {
                    let a = if i < self.pixel_bytes_raw {
                        0
                    } else {
                        self.scanline[i - self.pixel_bytes_raw] as u16
                    };
                    let b = self.scanline[i] as u16;
                    self.scanline[i] = ((a + b) / 2) as u8 + x;
                    i += 1;
                },
                Paeth => {
                    let mut px_pos = i % self.pixel_bytes_raw;
                    for &x in line.iter() {
                        let a = if i < self.pixel_bytes_raw {
                            0
                        } else {
                            self.scanline[i - self.pixel_bytes_raw] as i16
                        };
                        let b = self.scanline[i] as i16;
                        let c = self.pixel_prev[px_pos] as i16;
                        self.pixel_prev[px_pos] = self.scanline[i];
                        let p = a + b - c;
                        let pa = abs(p - a);
                        let pb = abs(p - b);
                        let pc = abs(p - c);
                        let pr = if pa <= pb && pa <= pc { a }
                            else if pb <= pc { b }
                            else { c };
                        self.scanline[i] = pr as u8 + x;
                        i += 1;
                        px_pos += 1;
                        if px_pos >= self.pixel_bytes_raw {
                            px_pos -= self.pixel_bytes_raw;
                        }
                    }
                }
            }

            data = data.slice_from(line.len());

            self.update_scanline(i - line.len(), i);

            scanline_pos = if i < self.scanline_bytes_raw {
                Some(i)
            } else {
                None
            };
        }

        self.scanline_pos = scanline_pos;
        self.filter = filter;

        Ok(())
    }

    fn interlace_params(&self) -> (/*x0*/ uint, /*y0*/ uint, /*dx*/ uint, /*dy*/ uint) {
        match self.interlace {
            // interlace_method = 0:
            0 => (0, 0, 1, 1),

            // interlace_method = 1 (7 steps):
            1 => (0, 0, 8, 8),

            /* NOTE these seem to follow the pattern:
             * (i, 0, 2*i, 2*i);
             * (0, i,   i, 2*i);
             * with i in [4, 2, 1].
             */
            2 => (4, 0, 8, 8),
            3 => (0, 4, 4, 8),

            4 => (2, 0, 4, 4),
            5 => (0, 2, 2, 4),

            6 => (1, 0, 2, 2),
            7 => (0, 1, 1, 2),
            _ => fail!("unreacheable (interlace step)")
        }
    }

    fn update_scanline(&mut self, from: uint, to: uint) {
        let data = self.scanline.slice(from, to);
        let (_, _, dx, _) = self.interlace_params();

        let mut i = self.y_byte_pos + self.x_byte_pos;
        let next_line = self.y_byte_pos + self.scanline_bytes;

        // Grayscale + Alpha.
        let pixel_ka = |k, alpha| {
            self.image.pixels[i] = k;
            self.image.pixels[i + 1] = alpha;
            i += dx * 2;
        };
        // Grayscale.
        let pixel_k = |k: u8, multiplier| {
            let alpha = match self.transparent_color {
                Some(v) if v[0] == k as u16 => 0x00,
                _ => 0xff
            };
            pixel_ka(k * multiplier, alpha);
        };
        // Palette.
        let pixel_pal = |j: u8| {
            let palette = self.palette.as_ref().unwrap();
            let j = j as uint * 4;
            self.image.pixels[i] = palette[j];
            self.image.pixels[i + 1] = palette[j + 1];
            self.image.pixels[i + 2] = palette[j + 2];
            self.image.pixels[i + 3] = palette[j + 3];
            i += dx * 4;
        };
        // One byte of RGBA or KA.
        let pixel_byte = |byte, pixel_bytes| {
            self.image.pixels[i] = byte;

            i += 1;
            if i % pixel_bytes == 0 {
                i += (dx - 1) * pixel_bytes;
            }
        };
        match self.color_type {
            K1 | K2 | K4 => for &x in data.iter() {
                let bits = self.color_type.pixel_bits();
                let multiplier = match bits {
                    2 => 0x55,
                    4 => 0x11,
                    _ => 0xff
                };
                for bit in range_step_inclusive(8 - bits, 0, bits) {
                    pixel_k((x >> bit) & ((1 << bits) - 1), multiplier);
                    if i > next_line { // TODO(eddyb) optimize (check only for last byte).
                        break;
                    }
                }
            },
            K8 => for &x in data.iter() {
                pixel_k(x, 0x01);
            },
            KA8 => for &x in data.iter() {
                pixel_byte(x, 2);
            },
            Pal1 | Pal2 | Pal4 => for &x in data.iter() {
                let bits = self.color_type.pixel_bits();
                for bit in range_step_inclusive(8 - bits, 0, bits) {
                    pixel_pal((x >> bit) & ((1 << bits) - 1));
                    if i > next_line { // TODO(eddyb) optimize (check only for last byte).
                        break;
                    }
                }
            },
            Pal8 => for &x in data.iter() {
                pixel_pal(x);
            },
            RGB8 => for &x in data.iter() {
                pixel_byte(x, 4);

                if i % 4 == 3 {
                    let alpha = match self.transparent_color {
                        Some([r, g, b]) if r == self.image.pixels[i - 3] as u16
                                        && g == self.image.pixels[i - 2] as u16
                                        && b == self.image.pixels[i - 1] as u16 => 0x00,
                        _ => 0xff
                    };
                    pixel_byte(alpha, 4);
                }
            },
            RGBA8 => for &x in data.iter() {
                pixel_byte(x, 4);
            },
            _ => fail!("unreacheable (TODO implement 16-bit depth)")
        }

        if i < next_line {
            self.x_byte_pos = i - self.y_byte_pos;
        } else {
            let (x0, _, _, dy) = self.interlace_params();
            let pixel_bytes = self.image.color_type.pixel_bits() / 8;

            self.x_byte_pos = x0 * pixel_bytes;
            self.y_byte_pos += dy * self.scanline_bytes;
            if self.y_byte_pos >= self.image.pixels.len() {
                match self.interlace {
                    0 | 7 => {
                        // FIXME(eddyb) free all temporary structures.
                        self.palette = None;
                        self.idat_inflate_stream = None;
                    }
                    _ => {
                        self.interlace += 1;
                        let (x0, y0, _, _) = self.interlace_params();
                        self.x_byte_pos = x0 * pixel_bytes;
                        self.y_byte_pos = y0 * pixel_bytes;
                    }
                }
            }
        }
    }
}

enum State {
    CheckMagic(/*offset*/ u8),
    U16(U16Next),
    U16Byte1(U16Next, /*value*/ u8),
    U32(U32Next, /*offset*/ u8, /*value*/ u32),
    Chunk4CC(/*size*/ u32),
    Chunk4CC1(/*size*/ u32, [u8, ..1]),
    Chunk4CC2(/*size*/ u32, [u8, ..2]),
    Chunk4CC3(/*size*/ u32, [u8, ..3]),
    IgnoreChunk(/*left*/ u32),
    IhdrBits(/*width*/ u32, /*height*/ u32),
    IhdrColorType(/*width*/ u32, /*height*/ u32, /*bits*/ u8),
    IhdrCompressionMethod(/*width*/ u32, /*height*/ u32, /*bits*/ u8, /*color_type*/ u8),
    IhdrFilterMethod(/*width*/ u32, /*height*/ u32, /*bits*/ u8, /*color_type*/ u8, /*compression_method*/ u8),
    IhdrInterlaceMethod(/*width*/ u32, /*height*/ u32, /*bits*/ u8, /*color_type*/ u8, /*compression_method*/ u8, /*filter_method*/ u8),
    Plte(/*left*/ u32),
    Trns(/*left*/ u32, /*index*/ u32),
    IdatInflate(/*left*/ u32)
}

enum U16Next {
    U16TrnsK,
    U16TrnsR,
    U16TrnsG(/*red*/ u16),
    U16TrnsB(/*red*/ u16, /*green*/ u16)
}

enum U32Next {
    U32ChunkSize,
    U32ChunkCRC(/*last_chunk*/ bool),
    U32IhdrWidth,
    U32IhdrHeight(/*width*/ u32)
}

pub struct Decoder {
    priv state: Option<State>,
    priv image: Option<PartialImage>
}

impl Decoder {
    pub fn new() -> Decoder {
        Decoder {
            state: Some(CheckMagic(0)),
            image: None
        }
    }

    fn next_state(&mut self, data: &[u8]) -> Result<uint, ~str> {
        let b = data[0];
        let ok2 = |n: u32, state| {
            self.state = Some(state);
            Ok(n as uint)
        };
        let ok = |state| ok2(1, state);
        let ok_u32 = |next| ok(U32(next, 0, 0));
        let skip_crc = U32(U32ChunkCRC(false), 0, 0);

        let state = match self.state {
            Some(state) => state,
            None => return Err(~"called png::Decoder::next_state with non-existent state")
        };

        match state {
            CheckMagic(i) => {
                if b != MAGIC[i] {
                    Err(format!("PNG header mismatch, expected {:#02x} but found {:#02x} for byte {}", MAGIC[i], b, i))
                } else if i < 7 {
                    ok(CheckMagic(i + 1))
                } else {
                    ok_u32(U32ChunkSize)
                }
            }
            U16(next) => ok(U16Byte1(next, b)),
            U16Byte1(next, value) => {
                let value = (value as u16 << 8) | b as u16;
                match (next, value) {
                    (U16TrnsK, k) => {
                        let image = self.image.as_mut().unwrap();
                        image.transparent_color = Some([k, k, k]);
                        ok(skip_crc)
                    }
                    (U16TrnsR, r) => ok(U16(U16TrnsG(r))),
                    (U16TrnsG(r), g) => ok(U16(U16TrnsB(r, g))),
                    (U16TrnsB(r, g), b) => {
                        let image = self.image.as_mut().unwrap();
                        image.transparent_color = Some([r, g, b]);
                        ok(skip_crc)
                    }
                }
            }
            U32(next, i, value) => {
                let value = (value << 8) | b as u32;
                if i < 3 {
                    ok(U32(next, i + 1, value))
                } else {
                    match next {
                        U32ChunkSize => ok(Chunk4CC(value)),
                        U32ChunkCRC(last_chunk) => {
                            // TODO(eddyb) check the CRC.
                            if last_chunk {
                                self.state = None;
                                Ok(1)
                            } else {
                                ok_u32(U32ChunkSize)
                            }
                        }
                        U32IhdrWidth => ok_u32(U32IhdrHeight(value)),
                        U32IhdrHeight(w) => ok(IhdrBits(w, value))
                    }
                }
            }
            Chunk4CC(size) => ok(Chunk4CC1(size, [b])),
            Chunk4CC1(size, [b0]) => ok(Chunk4CC2(size, [b0, b])),
            Chunk4CC2(size, [b0, b1]) => ok(Chunk4CC3(size, [b0, b1, b])),
            Chunk4CC3(size, [b0, b1, b2]) => {
                let name = [b0, b1, b2, b];
                match from_utf8(name) {
                    "IHDR" => {
                        if self.image.is_some() {
                            Err(~"duplicate IHDR")
                        } else if size != size_of::<Ihdr>() as u32 {
                            Err(format!("IHDR size mismatch, expected {} but found {}", size_of::<Ihdr>(), size))
                        } else {
                            ok_u32(U32IhdrWidth)
                        }
                    }
                    "PLTE" => {
                        if size > 0 && size % 3 != 0 {
                            Err(format!("PLTE has non multiple of 3 size {}", size))
                        } else {
                            match self.image {
                                None => Err(~"PLTE before IHDR"),
                                Some(ref mut image) => {
                                    if image.idat_inflate_stream.is_some() {
                                        Err(~"PLTE after IDAT")
                                    } else if image.palette.is_some() {
                                        Err(~"duplicate PLTE")
                                    } else if !image.color_type.is_palette() {
                                        // Ignore a palette that's not used to decode the image.
                                        ok(IgnoreChunk(size))
                                    } else {
                                        image.palette = Some(vec::with_capacity(size as uint / 3 * 4));
                                        ok(Plte(size))
                                    }
                                }
                            }
                        }
                    }
                    "tRNS" => {
                        match self.image {
                            None => Err(~"tRNS before IHDR"),
                            Some(ref mut image) => {
                                if image.idat_inflate_stream.is_some() {
                                    Err(~"tRNS after IDAT")
                                } else {
                                    match image.color_type {
                                        K1 | K2 | K4 | K8 | K16 => ok(U16(U16TrnsK)),
                                        Pal1 | Pal2 | Pal4 | Pal8 => ok(Trns(size, 3)),
                                        RGB8 | RGB16 => ok(U16(U16TrnsR)),
                                        _ => ok(IgnoreChunk(size))
                                    }
                                }
                            }
                        }
                    }
                    "IDAT" => {
                        if self.image.is_none() {
                            Err(~"IDAT before IHDR")
                        } else if self.image.as_ref().unwrap().color_type.is_palette()
                            && self.image.as_ref().unwrap().palette.is_none() {
                            Err(~"IDAT before PLTE")
                        } else {
                            let stream = &mut self.image.as_mut().unwrap().idat_inflate_stream;
                            if stream.is_none() {
                                // FIXME(eddyb) harcoded buffer size.
                                *stream = Some(~InflateStream::new(0x1000));
                            }
                            ok(IdatInflate(size))
                        }
                    }
                    "IEND" => ok_u32(U32ChunkCRC(true)),
                    "tEXt" | "iTXt" | "iCCP" => ok(IgnoreChunk(size)), // TODO(eddyb) maybe save the data?
                    name => {
                        error!("skipping unrecognized PNG chunk `{}` (size={})", name, size);
                        ok(IgnoreChunk(size))
                    }
                }
            }
            IgnoreChunk(left) => {
                let n = min(left, data.len() as u32);
                if left > n {
                    ok2(n, IgnoreChunk(left - n))
                } else {
                    ok2(n, skip_crc)
                }
            }
            IhdrBits(w, h) => ok(IhdrColorType(w, h, b)),
            IhdrColorType(w, h, bits) => ok(IhdrCompressionMethod(w, h, bits, b)),
            IhdrCompressionMethod(w, h, bits, c) => ok(IhdrFilterMethod(w, h, bits, c, b)),
            IhdrFilterMethod(w, h, bits, c, z) => ok(IhdrInterlaceMethod(w, h, bits, c, z, b)),
            IhdrInterlaceMethod(w, h, bits, c, z, f) => {
                let header = Ihdr {
                    width: w,
                    height: h,
                    bits: bits,
                    color_type: c,
                    compression_method: z,
                    filter_method: f,
                    interlace_method: b
                };
                match header.to_image() {
                    Ok(image) => {
                        self.image = Some(image);
                        ok(skip_crc)
                    }
                    Err(m) => Err(m)
                }
            }
            Plte(left) => {
                let n = min(left, data.len() as u32);
                let image = self.image.as_mut().unwrap();
                let palette = image.palette.as_mut().unwrap();
                for &x in data.slice_to(n as uint).iter() {
                    palette.push(x);

                    if palette.len() % 4 == 3 {
                        palette.push(0xff);
                    }
                }
                if left > n {
                    ok2(n, Plte(left - n))
                } else {
                    ok2(n, skip_crc)
                }
            }
            Trns(left, mut i) => {
                let n = min(left, data.len() as u32);
                let image = self.image.as_mut().unwrap();
                let palette = image.palette.as_mut().unwrap();
                for &x in data.slice_to(n as uint).iter() {
                    palette[i] = x;
                    i += 4;
                }
                if left > n {
                    ok2(n, Trns(left - n, i))
                } else {
                    ok2(n, skip_crc)
                }
            }
            IdatInflate(left) => {
                let mut n = min(left, data.len() as u32);
                let image = self.image.as_mut().unwrap();
                let mut stream = image.idat_inflate_stream.take_unwrap();
                match stream.update(data.slice_to(n as uint)) {
                    Ok((used, output)) => {
                        let used = used as u32;
                        image.update_idat(output);
                        n = used;
                    }
                    Err(m) => return Err(format!("IDAT error: {:s}", m))
                }
                // FIXME(eddyb) don't put back if it's no longer required.
                image.idat_inflate_stream = Some(stream);
                if left > n {
                    ok2(n, IdatInflate(left - n))
                } else {
                    ok2(n, skip_crc)
                }
            }
        }
    }

    pub fn update<'a>(&'a mut self, mut data: &[u8]) -> ImageState<'a> {
        while data.len() > 0 {
            match self.next_state(data) {
                Ok(n) => { data = data.slice_from(n); }
                Err(m) => return Error(m)
            }
        }
        Partial(self.image.as_ref().map(|partial| &partial.image))
    }
}

pub trait DecoderRef {
    fn update<'a>(&'a mut self, data: &[u8]) -> ImageState<'a>;
}

impl DecoderRef for Option<~Decoder> {
    fn update<'a>(&'a mut self, data: &[u8]) -> ImageState<'a> {
        match self.take() {
            Some(mut decoder) => {
                match decoder.update(data) {
                    Partial(_) if decoder.state.is_some() => {
                        *self = Some(decoder);
                        Partial(self.as_ref().unwrap().image.as_ref().map(|partial| &partial.image))
                    }
                    Error(m) => Error(m),
                    _ => Complete(decoder.image.take_unwrap().image)
                }
            }
            None => Error(~"called Option<~png::Decoder>::update on None")
        }
    }
}

/*
pub fn is_png(image: &[u8]) -> bool {
    do image.as_imm_buf |bytes, _len| {
        unsafe {
            ffi::png_sig_cmp(bytes, 0, 8) == 0
        }
    }
}*/

fn load_png(path: &Path) -> Result<Image, ~str> {
    match File::open_mode(path, io::Open, io::Read) {
        Some(mut r) => load_png_from_memory(r.read_to_end()),
        None => Err(~"could not open file"),
    }
}

fn load_png_from_memory(image: &[u8]) -> Result<Image, ~str> {
    let mut decoder = Some(~Decoder::new());
    match decoder.update(image) {
        Partial(_) => Err(~"incomplete PNG file"),
        Complete(image) => Ok(image),
        Error(m) => Err(m)
    }
}

#[cfg(test)]
mod test {
    extern mod extra;

    use std::io;
    use std::io::File;
    use std::vec;
    use super::{load_png, load_png_from_memory, RGBA8, Decoder, Partial, Complete, Error};

    fn load_rgba8(file: &'static str, w: u32, h: u32) {
        match load_png(&Path::new(file)) {
            Err(m) => fail!(m),
            Ok(image) => {
                assert_eq!(image.color_type, RGBA8);
                assert_eq!(image.width, w);
                assert_eq!(image.height, h);
            }
        }
    }

    #[test]
    fn test_load() {
        load_rgba8("test.png", 831, 624);
        load_rgba8("test_store.png", 10, 10);
    }

    fn load_rgba8_in_chunks(file: &'static str, chunk_size: uint, w: u32, h: u32) {
        do spawn {
            let mut reader = match File::open_mode(&Path::new(file), io::Open, io::Read) {
                Some(r) => r,
                None => fail!("could not open file"),
            };

            let mut buf = vec::from_elem(chunk_size, 0u8);
            let mut decoder = Some(~Decoder::new());
            loop {
                match reader.read(buf.mut_slice(0, chunk_size)) {
                    Some(count) => match decoder.update(buf.slice_to(count)) {
                        Partial(_) => {}
                        Complete(image) => {
                            assert_eq!(image.color_type, RGBA8);
                            assert_eq!(image.width, w);
                            assert_eq!(image.height, h);
                            break;
                        },
                        Error(m) => fail!(m)
                    },
                    None => fail!("incomplete PNG file")
                }
            }
        }
    }

    #[test]
    fn test_load_big_parallel() {
        // HACK(eddyb) arbitrary values.
        for _ in range(0, 128) {
            load_rgba8_in_chunks("test.png", 1024, 831, 624);
        }
    }

    fn load_rgba8_from_memory(buf: &[u8], w: u32, h: u32) {
        match load_png_from_memory(buf) {
            Err(m) => fail!(m),
            Ok(image) => {
                assert_eq!(image.color_type, RGBA8);
                assert_eq!(image.width, w);
                assert_eq!(image.height, h);
            }
        }
    }

    #[test]
    fn test_load_big_100() {
        let mut reader = match File::open_mode(&Path::new("test.png"), io::Open, io::Read) {
            Some(r) => r,
            None => fail!("could not open file")
        };
        let buf = reader.read_to_end();

        for _ in range(0, 100) {
            load_rgba8_from_memory(buf, 831, 624);
        }
    }

    fn bench_file_rgba(b: &mut extra::test::BenchHarness, file: &'static str, w: u32, h: u32) {
        let mut reader = match File::open_mode(&Path::new(file), io::Open, io::Read) {
            Some(r) => r,
            None => fail!("could not open file")
        };
        let buf = reader.read_to_end();

        b.iter(|| load_rgba8_from_memory(buf, w, h));
        b.bytes = (w * h * 4) as u64;
    }

    #[bench]
    fn bench_small(b: &mut extra::test::BenchHarness) {
        bench_file_rgba(b, "test_store.png", 10, 10);
    }

    #[bench]
    fn bench_big(b: &mut extra::test::BenchHarness) {
        bench_file_rgba(b, "test.png", 831, 624);
    }
}
