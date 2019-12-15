use super::{Data, RawPtrBox};
use crate::datatypes::PrimitiveType;
use crate::memory::{AllocationError, BufferBuilder};
use std::any::Any;
use std::convert::TryFrom;
use std::fmt;
use std::mem;
use std::ops::Deref;
use std::ops::Index;
use std::slice;
use std::sync::Arc;

/// An array whose elements are of primitive types.
pub struct Array<T: PrimitiveType> {
    data: Arc<Data>,
    raw_values: RawPtrBox<T::Native>,
}

impl<T> Array<T>
where
    T: PrimitiveType,
{
    pub fn iter(&self) -> slice::Iter<T::Native> {
        self.deref().iter()
    }
}

impl<T: PrimitiveType> super::Array for Array<T> {
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

impl<T: PrimitiveType> fmt::Debug for Array<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PrimitiveArray<{:?}>", T::get_data_type())?;
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: PrimitiveType> Deref for Array<T> {
    type Target = [T::Native];

    fn deref(&self) -> &[T::Native] {
        unsafe { slice::from_raw_parts(self.raw_values.get(), self.data.len) }
    }
}

impl<T: PrimitiveType> Index<usize> for Array<T> {
    type Output = T::Native;

    /// Returns a reference to an element.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bound.
    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.len() {
            panic!("index out of bound");
        }
        unsafe { &*self.raw_values.get().add(index) }
    }
}

impl<'a, T: PrimitiveType> IntoIterator for &'a Array<T> {
    type Item = <PrimitiveArrayIter<'a, T> as Iterator>::Item;
    type IntoIter = PrimitiveArrayIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

type PrimitiveArrayIter<'a, T> = slice::Iter<'a, <T as PrimitiveType>::Native>;

impl<T: PrimitiveType> TryFrom<&[T::Native]> for Array<T> {
    type Error = AllocationError;

    fn try_from(slice: &[T::Native]) -> Result<Self, Self::Error> {
        let mut builder = Builder::with_capacity(slice.len())?;
        for s in slice {
            builder.try_push(*s)?;
        }
        Ok(builder.into_array())
    }
}

/// An array builder for primitive types.
pub struct Builder<T: PrimitiveType> {
    values: BufferBuilder<T>,
}

impl<T: PrimitiveType> Builder<T> {
    pub fn with_capacity(capacity: usize) -> Result<Self, AllocationError> {
        Ok(Self {
            values: BufferBuilder::<T>::with_capacity(capacity)?,
        })
    }

    pub fn try_push(&mut self, v: T::Native) -> Result<(), AllocationError> {
        self.values.try_push(v)
    }

    fn into_array(self) -> Array<T> {
        let buffer = self.values.build();
        let raw_values = buffer.raw_data();
        let data = Data {
            data_type: T::get_data_type(),
            len: buffer.len() / mem::size_of::<T::Native>(),
            buffers: vec![buffer],
        };
        Array::<T> {
            data: Arc::new(data),
            raw_values: RawPtrBox::new(raw_values as *const T::Native),
        }
    }
}

impl<T: PrimitiveType> super::Builder for Builder<T> {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn build(self) -> Arc<dyn super::Array> {
        Arc::new(self.into_array())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::Int64Type;

    #[test]
    fn array_debug() {
        let array = Builder::<Int64Type>::with_capacity(1).unwrap().into_array();
        assert_eq!(format!("{:?}", array), "PrimitiveArray<Int64>[]");
    }
}
