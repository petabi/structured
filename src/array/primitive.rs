use super::{DataRepr, RawPtrBox};
use crate::datatypes::PrimitiveType;
use std::sync::Arc;

/// An array whose elements are of primitive types.
#[allow(dead_code)]
pub struct Array<T: PrimitiveType> {
    data: Arc<DataRepr>,
    raw_values: RawPtrBox<T::Native>,
}
