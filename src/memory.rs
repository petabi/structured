use crate::datatypes::PrimitiveType;
use crate::util::bit_util;
use std::alloc::{alloc, dealloc, realloc, Layout};
use std::cmp;
use std::fmt::{Debug, Formatter};
use std::io::{self, Write};
use std::marker::PhantomData;
use std::mem;
use std::ptr::{self, copy_nonoverlapping};
use std::slice;
use std::sync::Arc;
use thiserror::Error;

extern "C" {
    fn memcmp(p1: *const u8, p2: *const u8, len: usize) -> i32;
}

const ALIGNMENT: usize = 64;

/// A contiguous, immutable memory region of fixed size.
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
        let data = if capacity == 0 {
            ptr::null()
        } else {
            let data = alloc(Layout::from_size_align_unchecked(capacity, ALIGNMENT));
            copy_nonoverlapping(vec.as_ptr(), data, len);
            data
        };
        let buf_data = BufferData {
            ptr: data,
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

    #[allow(dead_code)]
    pub fn data(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.raw_data(), self.len()) }
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

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

/// A contiguous, mutable, and growable memory region.
#[derive(Debug)]
pub struct BufferMut {
    data: *mut u8,
    len: usize,
    capacity: usize,
}

impl BufferMut {
    pub fn with_capacity(capacity: usize) -> Result<Self, AllocationError> {
        if capacity > usize::max_value() - 63 {
            return Err(AllocationError::TooLarge);
        }
        let capacity = bit_util::round_upto_multiple_of_64(capacity);
        debug_assert!(capacity <= usize::max_value() - 63);
        let ptr = unsafe { alloc(Layout::from_size_align_unchecked(capacity, ALIGNMENT)) };
        if ptr.is_null() {
            return Err(AllocationError::Other);
        }
        Ok(Self {
            data: ptr,
            len: 0,
            capacity,
        })
    }

    /// Returns the total capacity of this buffer.
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns whether this buffer is empty or not.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of bytes written in this buffer.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the data stored in this buffer as a slice.
    #[allow(dead_code)]
    pub fn data(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.raw_data(), self.len()) }
    }

    /// Returns the data stored in this buffer as a mutable slice.
    #[allow(dead_code)]
    pub fn data_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.raw_data() as *mut u8, self.len()) }
    }

    /// Returns the raw pointer to the beginning of this buffer.
    #[allow(dead_code)]
    pub fn raw_data(&self) -> *const u8 {
        self.data
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.len = 0
    }

    pub fn try_reserve(&mut self, additional: usize) -> Result<(), AllocationError> {
        if additional > usize::max_value() - 63 - self.len {
            return Err(AllocationError::TooLarge);
        }

        let capacity = bit_util::round_upto_multiple_of_64(self.len + additional);
        if capacity <= self.capacity() {
            return Ok(());
        }
        let capacity = if self.capacity > usize::max_value() / 2 {
            usize::max_value() - 63
        } else {
            cmp::max(capacity, self.capacity * 2)
        };
        debug_assert!(0 < capacity && capacity <= usize::max_value() - 63);
        let data = unsafe {
            if self.data.is_null() {
                alloc(Layout::from_size_align_unchecked(capacity, ALIGNMENT))
            } else {
                realloc(
                    self.data,
                    Layout::from_size_align_unchecked(self.capacity, ALIGNMENT),
                    capacity,
                )
            }
        };
        if data.is_null() {
            return Err(AllocationError::Other);
        }
        self.data = data as *mut u8;
        self.capacity = capacity;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn build(self) -> Buffer {
        let buffer_data = BufferData {
            ptr: self.data,
            len: self.len,
            owned: true,
        };
        mem::forget(self);
        Buffer {
            data: Arc::new(buffer_data),
            offset: 0,
        }
    }
}

impl Drop for BufferMut {
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe {
                dealloc(
                    self.data as *mut u8,
                    Layout::from_size_align_unchecked(self.capacity, ALIGNMENT),
                );
            }
        }
    }
}

impl Write for BufferMut {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let remaining_capacity = self.capacity - self.len;
        if buf.len() > remaining_capacity {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                format!("cannot write more than {} bytes", remaining_capacity),
            ));
        }
        #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
        let dst = if self.len > isize::max_value() as usize {
            unsafe {
                self.data
                    .offset(isize::max_value())
                    .add(self.len - isize::max_value() as usize)
            }
        } else {
            unsafe { self.data.add(self.len) }
        };
        unsafe {
            copy_nonoverlapping(buf.as_ptr(), dst, buf.len());
        }
        self.len += buf.len();
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum AllocationError {
    #[error("cannot allocate memory larger than usize::max_value() - 63 bytes")]
    TooLarge,
    #[error("allocation failed")]
    Other,
}

/// Buffer builder.
#[allow(dead_code)]
pub struct BufferBuilder<T: PrimitiveType> {
    buffer: BufferMut,
    len: usize,
    _marker: PhantomData<T>,
}

#[allow(dead_code)]
impl<T> BufferBuilder<T>
where
    T: PrimitiveType,
    [T::Native]: AsRef<[u8]>,
{
    pub fn with_capacity(capacity: usize) -> Result<Self, AllocationError> {
        let buffer = BufferMut::with_capacity(capacity * mem::size_of::<T::Native>())?;
        Ok(Self {
            buffer,
            len: 0,
            _marker: PhantomData,
        })
    }

    pub fn try_push(&mut self, v: T::Native) -> Result<(), Error> {
        self.try_reserve(1)?;
        self.buffer.write_all(v.as_ref())?;
        self.len += 1;
        Ok(())
    }

    pub fn extend_from_slice(&mut self, slice: &[T::Native]) -> Result<(), Error> {
        self.try_reserve(slice.len())?;
        self.buffer.write_all(slice.as_ref())?;
        self.len += slice.len();
        Ok(())
    }

    fn try_reserve(&mut self, additional: usize) -> Result<(), AllocationError> {
        if usize::max_value() / mem::size_of::<T::Native>() < additional {
            return Err(AllocationError::TooLarge);
        }
        self.buffer
            .try_reserve(mem::size_of::<T::Native>() * additional)?;
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("not enough space in buffer: {0}")]
    BufferError(#[from] io::Error),
    #[error("memory error: {0}")]
    MemoryError(#[from] AllocationError),
}

#[cfg(test)]
mod tests {
    use super::{Buffer, BufferMut};
    use std::io::Write;

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

    #[test]
    fn buffer_mut_with_capacity() {
        let buf = BufferMut::with_capacity(63).unwrap();
        assert_eq!(64, buf.capacity());
        assert_eq!(0, buf.len());
        assert!(buf.is_empty());
    }

    #[test]
    fn buffer_mut_write() {
        let mut buf = BufferMut::with_capacity(100).unwrap();
        buf.write("hello".as_bytes()).unwrap();
        assert_eq!(5, buf.len());
        assert_eq!("hello".as_bytes(), buf.data());

        buf.write(" world".as_bytes()).unwrap();
        assert_eq!(11, buf.len());
        assert_eq!("hello world".as_bytes(), buf.data());

        buf.clear();
        assert_eq!(0, buf.len());
        buf.write("hello arrow".as_bytes()).unwrap();
        assert_eq!(11, buf.len());
        assert_eq!("hello arrow".as_bytes(), buf.data());
    }

    #[test]
    #[should_panic(expected = "cannot write")]
    fn buffer_mut_write_overflow() {
        let mut buf = BufferMut::with_capacity(1).unwrap();
        assert_eq!(64, buf.capacity());
        for _ in 0..10 {
            buf.write(&[0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        }
    }

    #[test]
    fn buffer_mut_try_reserve() {
        let mut buf = BufferMut::with_capacity(1).unwrap();
        assert_eq!(64, buf.capacity());

        buf.try_reserve(10).unwrap();
        assert_eq!(64, buf.capacity());

        buf.try_reserve(100).unwrap();
        assert_eq!(128, buf.capacity());
    }

    #[test]
    fn buffer_mut_build() {
        let mut buf = BufferMut::with_capacity(1).unwrap();
        buf.write(b"aaaa bbbb cccc dddd").unwrap();
        assert_eq!(19, buf.len());
        assert_eq!(b"aaaa bbbb cccc dddd", buf.data());

        let immutable_buf = buf.build();
        assert_eq!(19, immutable_buf.len());
        assert_eq!(b"aaaa bbbb cccc dddd", immutable_buf.data());
    }
}
