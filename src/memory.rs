use crate::datatypes::{NativeType, PrimitiveType, RawBytes};
use crate::util::bit_util;
use std::alloc::{alloc, dealloc, realloc, Layout};
use std::cmp;
use std::fmt::{Debug, Formatter};
use std::io::{self, Write};
use std::marker::PhantomData;
use std::mem;
use std::ptr::{self, copy_nonoverlapping, NonNull};
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
    /// The size of the slice should not overflow when aligned on a
    /// 64-byte boundary, i.e., `vec.len() <= usize::max_value() - 63`.
    #[allow(dead_code)]
    pub(crate) unsafe fn from_small_slice(vec: &[u8]) -> Self {
        debug_assert!(vec.len() <= usize::max_value() - 63);
        let capacity = bit_util::round_upto_multiple_of_64(vec.len());
        let data = if capacity == 0 {
            ptr::null()
        } else {
            let data = alloc(Layout::from_size_align_unchecked(capacity, ALIGNMENT));
            copy_nonoverlapping(vec.as_ptr(), data, vec.len());
            data
        };
        let buf_data = BufferData::new(data, vec.len(), true);
        Self {
            data: Arc::new(buf_data),
            offset: 0,
        }
    }

    /// Returns the number of bytes in the buffer.
    pub fn len(&self) -> usize {
        self.data.len() - self.offset
    }

    #[allow(dead_code)]
    pub fn data(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.raw_data(), self.len()) }
    }

    /// Returns the raw pointer to the beginning of this buffer.
    pub fn raw_data(&self) -> *const u8 {
        unsafe { self.data.ptr().add(self.offset) }
    }

    /// Returns a typed slice.
    ///
    /// # Safety
    ///
    /// The stored data should be valid values of type `T`.
    #[allow(dead_code)]
    pub unsafe fn as_slice<T: NativeType>(&self) -> &[T] {
        assert_eq!(self.len() % mem::size_of::<T>(), 0);
        slice::from_raw_parts(
            self.raw_data() as *const T,
            self.len() / mem::size_of::<T>(),
        )
    }
}

struct BufferData {
    ptr: *const u8, // Must be non-null and 64-byte aligned.
    len: usize,
    owned: bool,
}

impl BufferData {
    /// Creates a new `BufferData`.
    ///
    /// # Safety
    ///
    /// The `Drop` implementation for `BufferData` requries that `ptr` must be
    /// non-null and aligned on a 64-byte boundary, and that `len` is the number
    /// of bytes allocated for the memory at `ptr`. If `owned` is `true`, the
    /// memory at `ptr` must be deallocated by this `BufferData`'s
    /// implementation only.
    #[allow(dead_code)]
    unsafe fn new(ptr: *const u8, len: usize, owned: bool) -> Self {
        debug_assert!(!ptr.is_null());
        Self { ptr, len, owned }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn ptr(&self) -> *const u8 {
        self.ptr
    }
}

impl Drop for BufferData {
    /// Releases the underlying memory.
    fn drop(&mut self) {
        if self.owned {
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
    data: NonNull<u8>,
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
        let data = unsafe {
            let ptr = alloc(Layout::from_size_align_unchecked(capacity, ALIGNMENT));
            if ptr.is_null() {
                return Err(AllocationError::Other);
            }
            NonNull::new_unchecked(ptr)
        };
        Ok(Self {
            data,
            len: 0,
            capacity,
        })
    }

    /// Returns the total capacity of this buffer.
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
    pub fn raw_data(&self) -> *const u8 {
        self.data.as_ptr()
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
        self.data = unsafe {
            let ptr = realloc(
                self.data.as_ptr(),
                Layout::from_size_align_unchecked(self.capacity, ALIGNMENT),
                capacity,
            );
            if ptr.is_null() {
                return Err(AllocationError::Other);
            }
            NonNull::new_unchecked(ptr)
        };
        self.capacity = capacity;
        Ok(())
    }
}

impl Drop for BufferMut {
    fn drop(&mut self) {
        unsafe {
            dealloc(
                self.data.as_ptr(),
                Layout::from_size_align_unchecked(self.capacity, ALIGNMENT),
            );
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
                    .as_ptr()
                    .offset(isize::max_value())
                    .add(self.len - isize::max_value() as usize)
            }
        } else {
            unsafe { self.data.as_ptr().add(self.len) }
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

impl Into<Buffer> for BufferMut {
    fn into(self) -> Buffer {
        let buffer_data = BufferData {
            ptr: self.raw_data(),
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

#[derive(Debug, Error)]
pub enum AllocationError {
    #[error("cannot allocate memory larger than usize::max_value() - 63 bytes")]
    TooLarge,
    #[error("allocation failed")]
    Other,
}

/// Buffer builder.
pub struct BufferBuilder<T: PrimitiveType> {
    buffer: BufferMut,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> BufferBuilder<T>
where
    [T::Native]: RawBytes,
    T: PrimitiveType,
{
    pub fn new() -> Result<Self, AllocationError> {
        Self::with_capacity(ALIGNMENT)
    }

    pub fn with_capacity(capacity: usize) -> Result<Self, AllocationError> {
        let buffer = BufferMut::with_capacity(capacity * mem::size_of::<T::Native>())?;
        Ok(Self {
            buffer,
            len: 0,
            _marker: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn try_reserve(&mut self, additional: usize) -> Result<(), AllocationError> {
        if usize::max_value() / mem::size_of::<T::Native>() < additional {
            return Err(AllocationError::TooLarge);
        }
        self.buffer
            .try_reserve(mem::size_of::<T::Native>() * additional)?;
        Ok(())
    }

    pub fn try_push(&mut self, v: T::Native) -> Result<(), AllocationError> {
        self.try_reserve(1)?;
        self.buffer
            .write_all(v.as_raw_bytes())
            .expect("should have enough space reserved");
        self.len += 1;
        Ok(())
    }

    pub fn extend_from_slice(&mut self, slice: &[T::Native]) -> Result<(), AllocationError> {
        self.try_reserve(slice.len())?;
        self.buffer
            .write_all(slice.as_raw_bytes())
            .expect("should have enough space reserved");
        self.len += slice.len();
        Ok(())
    }

    pub fn build(self) -> Buffer {
        self.buffer.into()
    }
}

#[cfg(test)]
mod tests {
    use super::{Buffer, BufferBuilder, BufferMut};
    use crate::datatypes;
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

        let immutable_buf: Buffer = buf.into();
        assert_eq!(19, immutable_buf.len());
        assert_eq!(b"aaaa bbbb cccc dddd", immutable_buf.data());
    }

    macro_rules! check_as_typed_data {
        ($input: expr, $primitive_t: ty) => {{
            let mut builder: BufferBuilder<$primitive_t> = BufferBuilder::with_capacity(1).unwrap();
            builder.extend_from_slice($input).unwrap();
            let buf = builder.build();
            let slice =
                unsafe { buf.as_slice::<<$primitive_t as datatypes::PrimitiveType>::Native>() };
            assert_eq!($input, slice);
        }};
    }

    #[test]
    fn buffer_builder_extend_from_slice() {
        check_as_typed_data!(&[1_i32, 3_i32, 6_i32], datatypes::Int32Type);
        check_as_typed_data!(&[1_i64, 3_i64, 6_i64], datatypes::Int64Type);
        check_as_typed_data!(&[1_u8, 3_u8, 6_u8], datatypes::UInt8Type);
        check_as_typed_data!(&[1_u32, 3_u32, 6_u32], datatypes::UInt32Type);
        check_as_typed_data!(&[1_f64, 3_f64, 6_f64], datatypes::Float64Type);
        check_as_typed_data!(&[1_i64, 3_i64, 6_i64], datatypes::TimestampSecondType);
    }
}
