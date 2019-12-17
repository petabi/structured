pub(crate) mod primitive;
pub(crate) mod variable;

pub use primitive::Array as PrimitiveArray;
pub use primitive::Builder as PrimitiveBuilder;
pub use variable::{StringArray, StringArrayIter, StringBuilder};

use crate::datatypes::*;
use crate::memory::Buffer;
use std::any::Any;
use std::fmt;
use std::sync::Arc;

/// A dynamically-typed array.
pub trait Array: fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    /// Returns the number of elements of this array.
    fn len(&self) -> usize;
    fn data(&self) -> &Data;
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
pub struct Data {
    /// The element type for this array data.
    data_type: DataType,
    /// The number of elements in this array data.
    len: usize,
    buffers: Vec<Buffer>,
}

impl Data {
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }
}

struct RawPtrBox<T> {
    inner: *const T,
}

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
