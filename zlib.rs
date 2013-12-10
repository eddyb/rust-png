// Copyright 2013 The Servo Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::libc::{c_void, c_char, c_int, c_uint, c_ulong};
use std::ptr::{mut_null, null};
use std::vec;

type alloc_func = extern "C" fn(/*opaque*/ *c_void, /*items*/ c_uint, /*size*/ c_uint);
type free_func = extern "C" fn(/*opaque*/ *c_void, /*address*/ *c_void);

struct z_stream {
    next_in: *u8,       /* next input byte */
    avail_in: c_uint,   /* number of bytes available at next_in */
    total_in: c_ulong,   /* total number of input bytes read so far */

    next_out: *mut u8,  /* next output byte should be put there */
    avail_out: c_uint,  /* remaining free space at next_out */
    total_out: c_ulong, /* total number of bytes output so far */

    msg: *c_char,       /* last error message, NULL if no error */
    state: *c_void,     /* not visible by applications */

    zalloc: alloc_func, /* used to allocate the internal state */
    zfree: free_func,   /* used to free the internal state */
    opaque: *c_void,    /* private data object passed to zalloc and zfree */

    data_type: c_int,   /* best guess about the data type: binary or text */
    adler: c_ulong,      /* adler32 value of the uncompressed data */
    reserved: c_ulong    /* reserved for future use */
}

#[link(name="z")]
extern {
    fn zlibVersion() -> *c_char;
    fn zError(err: c_int) -> *c_char;

    fn inflateInit_(stream: *mut z_stream, version: *c_char, stream_size: c_int) -> c_int;
    fn inflate(stream: *mut z_stream, flush: c_int) -> c_int;
    fn inflateEnd(stream: *mut z_stream) -> c_int;
}

unsafe fn inflateInit(stream: *mut z_stream) -> c_int {
    // HACK(eddyb) zlib does this:
    // #define inflateInit(strm) inflateInit_((strm), ZLIB_VERSION, (int)sizeof(z_stream))
    let ZLIB_VERSION = zlibVersion();
    inflateInit_(stream, ZLIB_VERSION, ::std::mem::size_of::<z_stream>() as c_int)
}

unsafe fn error_to_str(err: c_int) -> ~str {
    ::std::str::raw::from_c_str(zError(err))
}

pub struct InflateStream {
    priv buffer: ~[u8],
    priv stream: z_stream
}

impl InflateStream {
    pub fn new(buffer_size: uint) -> InflateStream {
        let mut stream = InflateStream {
            buffer: vec::with_capacity(buffer_size),
            stream: z_stream {
                next_in: null(),
                avail_in: 0,
                total_in: 0,

                next_out: mut_null(),
                avail_out: 0,
                total_out: 0,

                msg: null(),
                state: null(),

                zalloc: unsafe {::std::cast::transmute(0u)},
                zfree: unsafe {::std::cast::transmute(0u)},
                opaque: null(),

                data_type: 0,
                adler: 0,
                reserved: 0
            }
        };

        let err = unsafe {
            inflateInit(&mut stream.stream as *mut z_stream)
        };
        // TODO(eddyb) handle errors.
        if err != 0 {
            fail!("zlib::inflateInit error `{}` ({})", err, unsafe { error_to_str(err) });
        }

        stream
    }

    pub fn update<'a>(&'a mut self, data: &[u8]) -> Result<(uint, &'a [u8]), ~str> {
        data.as_imm_buf(|ptr, len| {
            self.stream.next_in = ptr;
            self.stream.avail_in = len as c_uint;
        });

        self.buffer.as_mut_buf(|ptr, _len| {
            self.stream.next_out = ptr;
        });
        self.stream.avail_out = self.buffer.capacity() as c_uint;

        let err = unsafe {
            // TODO(eddyb) do proper flushing.
            inflate(&mut self.stream as *mut z_stream, 0)
        };
        if err < 0 {
            return Err(format!("inflate error `{}` ({})", err, unsafe { error_to_str(err) }));
        }

        let used_in = data.len() - self.stream.avail_in as uint;
        let used_out = self.buffer.capacity() - self.stream.avail_out as uint;

        unsafe {
            vec::raw::set_len(&mut self.buffer, used_out);
        }

        Ok((used_in, self.buffer.as_slice()))
    }
}

impl Drop for InflateStream {
    fn drop(&mut self) {
        let err = unsafe {
            inflateEnd(&mut self.stream as *mut z_stream)
        };
        // TODO(eddyb) handle errors properly.
        if err != 0 {
            fail!("zlib::inflateEnd error `{}` ({})", err, unsafe { error_to_str(err) });
        }
    }
}
