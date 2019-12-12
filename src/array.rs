mod primitive;
mod string;

pub use primitive::Array as PrimitiveArray;
pub use string::Array as StringArray;

use crate::datatypes::DataType;
use crate::memory::Buffer;
use std::fmt;
use std::sync::Arc;

/// A dynamically-typed array.
pub trait Array: fmt::Debug {
    /// Returns the number of elements of this array.
    fn len(&self) -> usize;
}

/// An `Array` builder.
pub trait Builder {
    /// Returns the number of array slots in the builder.
    fn len(&self) -> usize;
    /// Converts inself into an `Array`.
    fn build(self) -> Arc<dyn Array>;
}

/// A generic representation of array data.
#[derive(PartialEq, Debug, Clone)]
struct Data {
    /// The element type for this array data.
    data_type: DataType,
    /// The number of elements in this array data.
    len: usize,
    buffers: Vec<Buffer>,
}

struct RawPtrBox<T> {
    inner: *const T,
}

#[allow(dead_code)]
impl<T> RawPtrBox<T> {
    fn new(inner: *const T) -> Self {
        Self { inner }
    }

    fn get(&self) -> *const T {
        self.inner
    }
}

unsafe impl<T> Send for RawPtrBox<T> {}
unsafe impl<T> Sync for RawPtrBox<T> {}
