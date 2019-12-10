use crate::util::bit_util;
use std::alloc::{alloc, dealloc, Layout};
use std::fmt::{Debug, Formatter};
use std::mem;
use std::ptr::copy_nonoverlapping;
use std::sync::Arc;

extern "C" {
    fn memcmp(p1: *const u8, p2: *const u8, len: usize) -> i32;
}

const ALIGNMENT: usize = 64;

/// A contiguous memory region of fixed size.
#[derive(Clone, Debug, PartialEq)]
pub struct Buffer {
    data: Arc<BufferData>,
    offset: usize,
}

impl Buffer {
    /// Creates a buffer from a byte slice.
    ///
    /// # Safety
    ///
    /// The size of the slice should not overflow when aligned on a 64-byte
    /// boundary, i.e., `vec.len() <= usize::max_value() - 63`.
    pub(crate) unsafe fn from_small_slice(vec: &[u8]) -> Self {
        let len = vec.len() * mem::size_of::<u8>();
        let capacity = bit_util::round_upto_multiple_of_64(len);
        let ptr = alloc(Layout::from_size_align_unchecked(capacity, ALIGNMENT));
        copy_nonoverlapping(vec.as_ptr(), ptr, len);
        let buf_data = BufferData {
            ptr,
            len,
            owned: true,
        };
        Self {
            data: Arc::new(buf_data),
            offset: 0,
        }
    }

    /// Returns the number of bytes in the buffer.
    pub fn len(&self) -> usize {
        self.data.len - self.offset
    }

    /// Returns the raw pointer to the beginning of this buffer.
    pub fn raw_data(&self) -> *const u8 {
        unsafe { self.data.ptr.add(self.offset) }
    }
}

struct BufferData {
    ptr: *const u8, // Must be 64-byte aligned.
    len: usize,
    owned: bool,
}

/// Release the underlying memory when the current buffer goes out of scope
impl Drop for BufferData {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.owned {
            unsafe {
                dealloc(
                    self.ptr as *mut u8,
                    Layout::from_size_align_unchecked(self.len, ALIGNMENT),
                );
            }
        }
    }
}

impl PartialEq for BufferData {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        unsafe { memcmp(self.ptr, other.ptr, self.len) == 0 }
    }
}

impl Debug for BufferData {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "BufferData {{ ptr: {:?}, len: {}, data: ",
            self.ptr, self.len
        )?;

        unsafe {
            f.debug_list()
                .entries(std::slice::from_raw_parts(self.ptr, self.len).iter())
                .finish()?;
        }

        write!(f, " }}")
    }
}

#[cfg(test)]
mod tests {
    use super::Buffer;

    #[test]
    fn buffer_eq() {
        let buf1 = unsafe { Buffer::from_small_slice(&[0, 1, 2, 3, 4]) };
        let mut buf2 = unsafe { Buffer::from_small_slice(&[0, 1, 2, 3, 4]) };
        assert_eq!(buf1, buf2);

        buf2 = unsafe { Buffer::from_small_slice(&[0, 0, 2, 3, 4]) };
        assert_ne!(buf1, buf2);

        buf2 = unsafe { Buffer::from_small_slice(&[0, 1, 2, 3]) };
        assert_ne!(buf1, buf2);
    }
}
