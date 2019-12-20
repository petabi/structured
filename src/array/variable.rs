use super::Array as _Array;
use super::{Data, RawPtrBox};
use crate::datatypes::{DataType, Int32Type, UInt8Type};
use crate::memory::{AllocationError, BufferBuilder};
use num_traits::FromPrimitive;
use std::any::Any;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::marker::PhantomData;
use std::ops::Index;
use std::slice;
use std::str;
use std::sync::Arc;
use thiserror::Error;

/// An array whose elements are UTF-8 strings.
pub struct StringArray {
    data: Arc<Data>,
    offsets: RawPtrBox<i32>,
    values: RawPtrBox<u8>,
}

impl StringArray {
    pub fn iter(&self) -> StringArrayIter {
        let begin = self.offsets.get();
        StringArrayIter {
            cur: begin,
            end: unsafe { begin.add(self.data.len) },
            values: self.values.get(),
            _marker: PhantomData,
        }
    }
}

impl super::Array for StringArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.data.len
    }

    fn data(&self) -> &Data {
        &self.data
    }
}

impl fmt::Debug for StringArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StringArray")?;
        f.debug_list().entries(self.iter()).finish()
    }
}

impl Index<usize> for StringArray {
    type Output = str;

    /// Returns a reference to an element.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bound.
    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.len() {
            panic!("index out of bound");
        }
        let cur = unsafe { self.offsets.get().add(index) };
        let next = unsafe { cur.add(1) };
        let len = unsafe { *next - *cur };
        let ptr = unsafe {
            self.values
                .get()
                .offset((*cur).try_into().expect("invalid offset"))
        };
        unsafe {
            str::from_utf8_unchecked(slice::from_raw_parts(
                ptr,
                len.try_into().expect("invalid string length"),
            ))
        }
    }
}

impl<'a> IntoIterator for &'a StringArray {
    type Item = <StringArrayIter<'a> as Iterator>::Item;
    type IntoIter = StringArrayIter<'a>;

    #[must_use]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl TryFrom<&[&str]> for StringArray {
    type Error = AllocationError;

    fn try_from(slice: &[&str]) -> Result<Self, Self::Error> {
        let mut builder = StringBuilder::with_capacity(slice.len())?;
        for s in slice {
            builder.try_push(s)?;
        }
        Ok(builder.into_array())
    }
}

pub struct StringArrayIter<'a> {
    cur: *const i32,
    end: *const i32,
    values: *const u8,
    _marker: PhantomData<&'a i32>,
}

impl<'a> Iterator for StringArrayIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == self.end {
            None
        } else {
            let offset = unsafe { *self.cur };
            let ptr = unsafe {
                self.values
                    .offset(offset.try_into().expect("invalid offset"))
            };
            let next = unsafe { self.cur.add(1) };
            let len = unsafe { *next - *self.cur };
            self.cur = next;
            Some(unsafe {
                str::from_utf8_unchecked(slice::from_raw_parts(
                    ptr,
                    len.try_into().expect("invalid string length"),
                ))
            })
        }
    }
}

pub struct StringBuilder {
    offsets: BufferBuilder<Int32Type>,
    values: BufferBuilder<UInt8Type>,
}

impl StringBuilder {
    pub fn with_capacity(capacity: usize) -> Result<Self, AllocationError> {
        let mut offsets = BufferBuilder::<Int32Type>::with_capacity(capacity)?;
        offsets.try_push(0)?;
        Ok(Self {
            offsets,
            values: BufferBuilder::<UInt8Type>::new()?,
        })
    }

    pub fn try_push(&mut self, val: &str) -> Result<(), AllocationError> {
        if val.len() > i32::max_value() as usize - self.values.len() {
            return Err(AllocationError::TooLarge);
        }
        self.values.extend_from_slice(val.as_bytes())?;
        debug_assert!(i32::try_from(self.values.len()).is_ok());
        self.offsets
            .try_push(i32::from_usize(self.values.len()).expect("should not exceed 2^31 - 1"))?;
        Ok(())
    }

    fn into_array(self) -> StringArray {
        let len = self.offsets.len() - 1;
        let offsets_buffer = self.offsets.build();
        let values_buffer = self.values.build();
        let offsets = offsets_buffer.raw_data();
        let values = values_buffer.raw_data();
        let buffers = vec![offsets_buffer, values_buffer];
        let data = Arc::new(Data {
            data_type: DataType::Utf8,
            len,
            buffers,
        });
        #[allow(clippy::cast_ptr_alignment)]
        StringArray {
            data,
            offsets: RawPtrBox::new(offsets as *const i32),
            values: RawPtrBox::new(values),
        }
    }
}

impl super::Builder for StringBuilder {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn build(self) -> Arc<dyn super::Array> {
        Arc::new(self.into_array())
    }
}

pub struct BinaryArray {
    data: Arc<Data>,
    offsets: RawPtrBox<i32>,
    values: RawPtrBox<u8>,
}

impl BinaryArray {
    pub fn iter(&self) -> BinaryArrayIter {
        let begin = self.offsets.get();
        BinaryArrayIter {
            cur: begin,
            end: unsafe { begin.add(self.data.len) },
            values: self.values.get(),
            _marker: PhantomData,
        }
    }
}

impl fmt::Debug for BinaryArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BinaryArray")?;
        f.debug_list().entries(self.iter()).finish()
    }
}

impl Index<usize> for BinaryArray {
    type Output = [u8];

    /// Returns a reference to an element.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bound.
    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.len() {
            panic!("index out of bound");
        }
        let cur = unsafe { self.offsets.get().add(index) };
        let next = unsafe { cur.add(1) };
        let len = unsafe { *next - *cur };
        let ptr = unsafe {
            self.values
                .get()
                .offset((*cur).try_into().expect("invalid offset"))
        };
        unsafe { slice::from_raw_parts(ptr, len.try_into().expect("invalid byte string length")) }
    }
}

impl<'a> IntoIterator for &'a BinaryArray {
    type Item = <BinaryArrayIter<'a> as Iterator>::Item;
    type IntoIter = BinaryArrayIter<'a>;

    #[must_use]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl TryFrom<&[&[u8]]> for BinaryArray {
    type Error = AllocationError;

    fn try_from(slice: &[&[u8]]) -> Result<Self, Self::Error> {
        let mut builder = BinaryBuilder::with_capacity(slice.len())?;
        for s in slice {
            builder.try_push(s)?;
        }
        Ok(builder.into_array())
    }
}

impl super::Array for BinaryArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.data.len
    }

    fn data(&self) -> &Data {
        &self.data
    }
}

pub struct BinaryArrayIter<'a> {
    cur: *const i32,
    end: *const i32,
    values: *const u8,
    _marker: PhantomData<&'a i32>,
}

impl<'a> Iterator for BinaryArrayIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == self.end {
            None
        } else {
            let offset = unsafe { *self.cur };
            let ptr = unsafe {
                self.values
                    .offset(offset.try_into().expect("invalid offset"))
            };
            let next = unsafe { self.cur.add(1) };
            let len = unsafe { *next - *self.cur };
            self.cur = next;
            Some(unsafe {
                slice::from_raw_parts(ptr, len.try_into().expect("invalid string length"))
            })
        }
    }
}

pub struct BinaryBuilder {
    offsets: BufferBuilder<Int32Type>,
    values: BufferBuilder<UInt8Type>,
}

impl BinaryBuilder {
    pub fn with_capacity(capacity: usize) -> Result<Self, AllocationError> {
        let mut offsets = BufferBuilder::<Int32Type>::with_capacity(capacity)?;
        offsets.try_push(0)?;
        Ok(Self {
            offsets,
            values: BufferBuilder::<UInt8Type>::new()?,
        })
    }

    pub fn try_push(&mut self, val: &[u8]) -> Result<(), AllocationError> {
        if val.len() > i32::max_value() as usize - self.values.len() {
            return Err(AllocationError::TooLarge);
        }
        self.values.extend_from_slice(val)?;
        debug_assert!(i32::try_from(self.values.len()).is_ok());
        self.offsets
            .try_push(i32::from_usize(self.values.len()).expect("should not exceed 2^31 - 1"))?;
        Ok(())
    }

    fn into_array(self) -> BinaryArray {
        let len = self.offsets.len() - 1;
        let offsets_buffer = self.offsets.build();
        let values_buffer = self.values.build();
        let offsets = offsets_buffer.raw_data();
        let values = values_buffer.raw_data();
        let buffers = vec![offsets_buffer, values_buffer];
        let data = Arc::new(Data {
            data_type: DataType::Binary,
            len,
            buffers,
        });
        #[allow(clippy::cast_ptr_alignment)]
        BinaryArray {
            data,
            offsets: RawPtrBox::new(offsets as *const i32),
            values: RawPtrBox::new(values),
        }
    }
}

impl super::Builder for BinaryBuilder {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn build(self) -> Arc<dyn super::Array> {
        Arc::new(self.into_array())
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("memory error: {0}")]
    MemoryError(#[from] AllocationError),
    #[error("parse error: {0}")]
    ParseError(#[from] std::str::Utf8Error),
}

#[cfg(test)]
mod tests {
    use super::{BinaryArray, StringArray};
    use crate::array::Array;
    use std::convert::TryInto;

    #[test]
    fn array_from_u8_slice() {
        let values: Vec<&str> = vec!["hello", "", "parquet"];
        let string_array: StringArray = values.as_slice().try_into().expect("memory error");

        assert_eq!(3, string_array.len());
        assert_eq!(*"hello", string_array[0]);
        assert_eq!(*"", string_array[1]);
        assert_eq!(*"parquet", string_array[2]);
    }

    #[test]
    fn array_from_byte_slice() {
        let values: Vec<&[u8]> = vec![b"hello", b"", b"parquet"];
        let bytes_array: BinaryArray = values.as_slice().try_into().expect("memory error");

        assert_eq!(3, bytes_array.len());
        assert_eq!(*b"hello", bytes_array[0]);
        assert_eq!(*b"", bytes_array[1]);
        assert_eq!(*b"parquet", bytes_array[2]);
    }
}
