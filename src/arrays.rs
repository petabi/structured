mod data;

pub use data::ArrayDataRef;

use crate::datatypes::PrimitiveType;
use std::fmt;
use std::sync::Arc;

/// Trait for dealing with different types of array at runtime when the type of the
/// array is not known in advance
pub trait Array: fmt::Debug {}

pub type ArrayRef = Arc<dyn Array>;

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

/// Array whose elements are of primitive types.
#[allow(dead_code)]
pub struct PrimitiveArray<T: PrimitiveType> {
    data: ArrayDataRef,
    /// Pointer to the value array. The lifetime of this must be <= to the value buffer
    /// stored in `data`, so it's safe to store.
    /// Also note that boolean arrays are bit-packed, so although the underlying pointer
    /// is of type bool it should be cast back to u8 before being used.
    /// i.e. `self.raw_values.get() as *const u8`
    raw_values: RawPtrBox<T::Native>,
}

/// A list array where each element is a variable-sized sequence of values with the same
/// type.
#[allow(dead_code)]
pub struct ListArray {
    data: ArrayDataRef,
    values: ArrayRef,
    value_offsets: RawPtrBox<i32>,
}

/// A type of `ListArray` whose elements are UTF8 strings.
#[allow(dead_code)]
pub struct StringArray {
    data: ArrayDataRef,
    value_offsets: RawPtrBox<i32>,
    value_data: RawPtrBox<u8>,
}
