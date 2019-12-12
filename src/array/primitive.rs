use super::{Data, RawPtrBox};
use crate::datatypes::PrimitiveType;
use crate::memory::{AllocationError, BufferBuilder};
use std::fmt;
use std::ops::Deref;
use std::slice;
use std::sync::Arc;

/// An array whose elements are of primitive types.
#[allow(dead_code)]
pub struct Array<T: PrimitiveType> {
    data: Arc<Data>,
    raw_values: RawPtrBox<T::Native>,
}

impl<T: PrimitiveType> super::Array for Array<T> {
    fn len(&self) -> usize {
        self.data.len
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

/// An array builder for primitive types.
#[allow(dead_code)]
pub struct Builder<T: PrimitiveType> {
    values: BufferBuilder<T>,
}

#[allow(dead_code)]
impl<T: PrimitiveType> Builder<T> {
    pub fn try_push(&mut self, v: T::Native) -> Result<(), AllocationError> {
        self.values.try_push(v)
    }
}

impl<T: PrimitiveType> super::Builder for Builder<T> {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn build(self) -> Arc<dyn super::Array> {
        let buffer = self.values.build();
        let raw_values = buffer.raw_data();
        let data = Data {
            data_type: T::get_data_type(),
            len: buffer.len(),
            buffers: vec![buffer],
        };
        Arc::new(Array::<T> {
            data: Arc::new(data),
            raw_values: RawPtrBox::new(raw_values as *const T::Native),
        })
    }
}
