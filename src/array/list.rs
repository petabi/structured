use super::{DataRepr, RawPtrBox};
use std::sync::Arc;

/// An array where each element is a variable-sized sequence of values of the
/// same type.
#[allow(dead_code)]
pub struct Array {
    data: Arc<DataRepr>,
    values: Arc<dyn super::Array>,
    value_offsets: RawPtrBox<i32>,
}

/// An array whose elements are UTF-8 strings.
#[allow(dead_code)]
pub struct StringArray {
    data: Arc<DataRepr>,
    value_offsets: RawPtrBox<i32>,
    value_data: RawPtrBox<u8>,
}
