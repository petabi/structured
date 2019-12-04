use crate::memory;
use crate::util::bit_util;
use std::fmt::{Debug, Formatter};
use std::mem;
use std::ptr::copy_nonoverlapping;
use std::sync::Arc;

/// Buffer is a contiguous memory region of fixed size and is aligned at a 64-byte
/// boundary. Buffer is immutable.
#[derive(PartialEq, Debug)]
pub struct Buffer {
    /// Reference-counted pointer to the internal byte buffer.
    data: Arc<BufferData>,

    /// The offset into the buffer.
    offset: usize,
}

struct BufferData {
    /// The raw pointer into the buffer bytes
    ptr: *const u8,

    /// The length (num of bytes) of the buffer
    len: usize,

    /// Whether this piece of memory is owned by this object
    owned: bool,
}

impl PartialEq for BufferData {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        unsafe { memory::memcmp(self.ptr, other.ptr, self.len) == 0 }
    }
}

/// Release the underlying memory when the current buffer goes out of scope
impl Drop for BufferData {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.owned {
            memory::free_aligned(self.ptr as *mut u8, self.len);
        }
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

impl Buffer {
    /// Creates a buffer from an existing memory region (must already be byte-aligned), and this
    /// buffer will free this piece of memory when dropped.
    pub fn from_raw_parts(ptr: *const u8, len: usize) -> Self {
        Self::build_with_arguments(ptr, len, true)
    }

    /// Creates a buffer from an existing memory region (must already be byte-aligned)
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to raw parts.
    /// * `len` - Length of raw parts in bytes
    /// * `owned` - Whether the raw parts is owned by this buffer. If true, this buffer will free
    /// this memory when dropped, otherwise it will skip freeing the raw parts.
    fn build_with_arguments(ptr: *const u8, len: usize, owned: bool) -> Self {
        assert!(
            memory::is_aligned(ptr, memory::ALIGNMENT),
            "memory not aligned"
        );
        let buf_data = BufferData { ptr, len, owned };
        Self {
            data: Arc::new(buf_data),
            offset: 0,
        }
    }

    /// Returns the number of bytes in the buffer
    pub fn len(&self) -> usize {
        self.data.len - self.offset
    }

    /// Returns a raw pointer for this buffer.
    ///
    /// # Safety
    ///
    /// Note that this should be used cautiously, and the returned pointer should not be
    /// stored anywhere, to avoid dangling pointers.
    pub unsafe fn raw_data(&self) -> *const u8 {
        self.data.ptr.add(self.offset)
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            offset: self.offset,
        }
    }
}

/// Creating a `Buffer` instance by copying the memory from a `AsRef<[u8]>` into a newly
/// allocated memory region.
impl<T: AsRef<[u8]>> From<T> for Buffer {
    fn from(p: T) -> Self {
        // allocate aligned memory buffer
        let slice = p.as_ref();
        let len = slice.len() * mem::size_of::<u8>();
        let capacity = bit_util::round_upto_multiple_of_64(len);
        let buffer = memory::allocate_aligned(capacity);
        unsafe {
            copy_nonoverlapping(slice.as_ptr(), buffer, len);
        }
        Self::from_raw_parts(buffer, len)
    }
}
