// Copyright 2013 The Servo Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link(name = "png",
       vers = "0.1")];
#[crate_type = "lib"];
#[feature(link_args)];

extern mod std;
use std::cast;
use std::io;
use std::io::File;
use std::ptr;
use std::vec;
use std::libc::{c_int, size_t};

pub mod ffi;

#[nolink]
#[link_args="-L. -lpng -lz -lshim"]
extern {}

#[deriving(Eq)]
pub enum ColorType {
    K1, K2, K4, K8, K16,
    KA8, KA16,
    Pal1, Pal2, Pal4, Pal8,
    RGB8, RGB16,
    RGBA8, RGBA16,
}

pub struct Image {
    width: u32,
    height: u32,
    color_type: ColorType,
    pixels: ~[u8],
}

// This intermediate data structure is used to read
// an image data from 'offset' position, and store it
// to the data vector.
struct ImageData<'self> {
    data: &'self [u8],
    offset: uint,
}

#[fixed_stack_segment]
pub fn is_png(image: &[u8]) -> bool {
    image.as_imm_buf(|bytes, _len| {
        unsafe {
            ffi::png_sig_cmp(bytes, 0, 8) == 0
        }
    })
}

pub extern fn read_data(png_ptr: *ffi::png_struct, data: *mut u8, length: size_t) {
    unsafe {
        let io_ptr = ffi::png_get_io_ptr(png_ptr);
        let image_data: &mut ImageData = cast::transmute(io_ptr);
        let len = length as uint;
        vec::raw::mut_buf_as_slice(data, len, |buf| {
            let end_pos = std::num::min(image_data.data.len()-image_data.offset, len);
            vec::raw::copy_memory(buf,
                                  image_data.data.slice(image_data.offset, image_data.offset+end_pos),
                                  end_pos);
            image_data.offset += end_pos;
        });
    }
}

pub fn load_png(path: &Path) -> Result<Image,~str> {
    let mut reader = match File::open_mode(path, io::Open, io::Read) {
        Some(r) => r,
        None => return Err(~"could not open file"),
    };
    let buf = reader.read_to_end();
    load_png_from_memory(buf)
}

#[fixed_stack_segment]
pub fn load_png_from_memory(image: &[u8]) -> Result<Image,~str> {
    unsafe {
        let png_ptr = ffi::png_create_read_struct(ffi::png_get_header_ver(ptr::null()),
                                                  ptr::null(),
                                                  ptr::null(),
                                                  ptr::null());
        if png_ptr.is_null() {
            return Err(~"could not create read struct");
        }
        let info_ptr = ffi::png_create_info_struct(&*png_ptr);
        if info_ptr.is_null() {
            let png_ptr: *ffi::png_struct = &*png_ptr;
            ffi::png_destroy_read_struct(&png_ptr, ptr::null(), ptr::null());
            return Err(~"could not create info struct");
        }
        let res = ffi::setjmp(ffi::pngshim_jmpbuf(png_ptr));
        if res != 0 {
            let png_ptr: *ffi::png_struct = &*png_ptr;
            let info_ptr: *ffi::png_info = &*info_ptr;
            ffi::png_destroy_read_struct(&png_ptr, &info_ptr, ptr::null());
            return Err(~"error reading png");
        }

        let mut image_data = ImageData {
            data: image,
            offset: 0,
        };

        ffi::png_set_read_fn(png_ptr, cast::transmute(&mut image_data), read_data);
        ffi::png_read_info(png_ptr, info_ptr);

        let width = ffi::png_get_image_width(&*png_ptr, &*info_ptr);
        let height = ffi::png_get_image_height(&*png_ptr, &*info_ptr);
        let bit_depth = ffi::png_get_bit_depth(&*png_ptr, &*info_ptr);
        let color_type = ffi::png_get_color_type(&*png_ptr, &*info_ptr);

        // we convert palette to rgb
        if color_type as c_int == ffi::COLOR_TYPE_PALETTE {
            ffi::png_set_palette_to_rgb(png_ptr);
        }
        // make each channel use 1 byte
        if (color_type as c_int == ffi::COLOR_TYPE_GRAY) && (bit_depth < 8) {
            ffi::png_set_expand_gray_1_2_4_to_8(png_ptr);
        }
        // add alpha channels to palette and rgb
        if (color_type as c_int == ffi::COLOR_TYPE_PALETTE) ||
            (color_type as c_int == ffi::COLOR_TYPE_RGB) {
            ffi::png_set_tRNS_to_alpha(png_ptr);
            ffi::png_set_filler(png_ptr, 0xff, ffi::FILLER_AFTER);
        }

        ffi::png_set_packing(png_ptr);
        ffi::png_set_interlace_handling(png_ptr);
        ffi::png_read_update_info(png_ptr, info_ptr);

        let (color_type, pixel_width) = match (color_type as c_int, bit_depth) {
            (ffi::COLOR_TYPE_RGB, 8) |
            (ffi::COLOR_TYPE_RGBA, 8) |
            (ffi::COLOR_TYPE_PALETTE, 8) => (RGBA8, 4),
            (ffi::COLOR_TYPE_GRAY, 8) => (K8, 1),
            (ffi::COLOR_TYPE_GA, 8) => (KA8, 2),
            _ => fail!(~"color type not supported"),
        };

        let mut image_data = vec::from_elem((width * height * pixel_width) as uint, 0u8);
        let image_buf = vec::raw::to_mut_ptr(image_data);
        let row_pointers: ~[*mut u8] = vec::from_fn(height as uint, |idx| {
            ptr::mut_offset(image_buf, (((width * pixel_width) as uint) * idx) as int)
        });

        ffi::png_read_image(png_ptr, vec::raw::to_ptr(row_pointers));

        let png_ptr: *ffi::png_struct = &*png_ptr;
        let info_ptr: *ffi::png_info = &*info_ptr;
        ffi::png_destroy_read_struct(&png_ptr, &info_ptr, ptr::null());

        Ok(Image {
            width: width,
            height: height,
            color_type: color_type,
            pixels: image_data,
        })
    }
}

pub extern fn write_data(png_ptr: *ffi::png_struct, data: *u8, length: size_t) {
    unsafe {
        let io_ptr = ffi::png_get_io_ptr(png_ptr);
        let writer: &mut &mut io::Writer = cast::transmute(io_ptr);
        vec::raw::buf_as_slice(data, length as uint, |buf| {
            writer.write(buf);
        });
    }
}

pub extern fn flush_data(png_ptr: *ffi::png_struct) {
    unsafe {
        let io_ptr = ffi::png_get_io_ptr(png_ptr);
        let writer: &mut &mut io::Writer = cast::transmute(io_ptr);
        writer.flush();
    }
}

#[fixed_stack_segment]
pub fn store_png(img: &Image, path: &Path) -> Result<(),~str> {
    let mut file = match File::open_mode(path, io::Open, io::Write) {
        Some(file) => file,
        None => return Err(~"could not open file")
    };

    let mut writer = &mut file as &mut io::Writer;

    // Box it again because a &Trait is too big to fit in a void*.
    let writer = &mut writer;

    unsafe {
        let png_ptr = ffi::png_create_write_struct(ffi::png_get_header_ver(ptr::null()),
                                                   ptr::null(),
                                                   ptr::null(),
                                                   ptr::null());
        if png_ptr.is_null() {
            return Err(~"could not create write struct");
        }
        let info_ptr = ffi::png_create_info_struct(&*png_ptr);
        if info_ptr.is_null() {
            let png_ptr: *ffi::png_struct = &*png_ptr;
            ffi::png_destroy_write_struct(&png_ptr, ptr::null());
            return Err(~"could not create info struct");
        }
        let res = ffi::setjmp(ffi::pngshim_jmpbuf(png_ptr));
        if res != 0 {
            let png_ptr: *ffi::png_struct = &*png_ptr;
            let info_ptr: *ffi::png_info = &*info_ptr;
            ffi::png_destroy_write_struct(&png_ptr, &info_ptr);
            return Err(~"error writing png");
        }

        ffi::png_set_write_fn(png_ptr, cast::transmute(writer), write_data, flush_data);

        let (bit_depth, color_type, pixel_width) = match img.color_type {
            RGB8 => (8, ffi::COLOR_TYPE_RGB, 3),
            RGBA8 => (8, ffi::COLOR_TYPE_RGBA, 4),
            K8 => (8, ffi::COLOR_TYPE_GRAY, 1),
            KA8 => (8, ffi::COLOR_TYPE_GA, 2),
            _ => fail!("bad color type"),
        };

        ffi::png_set_IHDR(&*png_ptr, info_ptr, img.width, img.height, bit_depth, color_type,
                          ffi::INTERLACE_NONE, ffi::COMPRESSION_TYPE_DEFAULT, ffi::FILTER_NONE);

        let image_buf = vec::raw::to_ptr(img.pixels);
        let row_pointers: ~[*u8] = vec::from_fn(img.height as uint, |idx| {
            ptr::offset(image_buf, (((img.width * pixel_width) as uint) * idx) as int)
        });
        ffi::png_set_rows(&*png_ptr, info_ptr, vec::raw::to_ptr(row_pointers));

        ffi::png_write_png(png_ptr, info_ptr, ffi::TRANSFORM_IDENTITY, ptr::null());

        let png_ptr: *ffi::png_struct = &*png_ptr;
        let info_ptr: *ffi::png_info = &*info_ptr;
        ffi::png_destroy_write_struct(&png_ptr, &info_ptr);
    }
    Ok(())
}

#[cfg(test)]
mod test {
    extern mod extra;

    use std::io;
    use std::io::File;
    use std::vec;
    use super::{ffi, load_png, load_png_from_memory, store_png, RGB8, RGBA8, Image};

    #[test]
    #[fixed_stack_segment]
    fn test_valid_png() {
        let mut reader = match File::open_mode(&Path::new("test.png"), io::Open, io::Read) {
            Some(r) => r,
            None => fail!("could not open file"),
        };

        let mut buf = vec::from_elem(1024, 0u8);
        let count = reader.read(buf.mut_slice(0, 1024)).unwrap();
        assert!(count >= 8);
        unsafe {
            let res = ffi::png_sig_cmp(vec::raw::to_ptr(buf), 0, 8);
            assert!(res == 0);
        }
    }

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

        test_store();
        load_rgba8("test_store.png", 10, 10);
    }

    #[test]
    fn test_load_big_parallel() {
        // HACK(eddyb) arbitrary values.
        for _ in range(0, 128) {
            do spawn {
                load_rgba8("test.png", 831, 624);
            }
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
        test_store();
        bench_file_rgba(b, "test_store.png", 10, 10);
    }

    #[bench]
    fn bench_big(b: &mut extra::test::BenchHarness) {
        bench_file_rgba(b, "test.png", 831, 624);
    }

    #[test]
    fn test_store() {
        let img = Image {
            width: 10,
            height: 10,
            color_type: RGB8,
            pixels: vec::from_elem(10 * 10 * 3, 100u8),
        };
        let res = store_png(&img, &Path::new("test_store.png"));
        assert!(res.is_ok());
    }
}
