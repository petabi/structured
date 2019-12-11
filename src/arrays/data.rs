use crate::bitmap::Bitmap;
use crate::datatypes::DataType;
use crate::memory::Buffer;
use std::sync::Arc;

/// An generic representation of Arrow array data which encapsulates common attributes and
/// operations for Arrow array. Specific operations for different arrays types (e.g.,
/// primitive, list, struct) are implemented in `Array`.
#[derive(PartialEq, Debug, Clone)]
pub struct ArrayDataRepr {
    /// The data type for this array data
    data_type: DataType,

    /// The number of elements in this array data
    pub(crate) len: usize,

    /// The number of null elements in this array data
    pub(crate) null_count: usize,

    /// The offset into this array data
    pub(crate) offset: usize,

    /// The buffers for this array data. Note that depending on the array types, this
    /// could hold different kinds of buffers (e.g., value buffer, value offset buffer)
    /// at different positions.
    buffers: Vec<Buffer>,

    /// The child(ren) of this array. Only non-empty for nested types, currently
    /// `ListArray` and `StructArray`.
    child_data: Vec<ArrayDataRef>,

    /// The null bitmap. A `None` value for this indicates all values are non-null in
    /// this array.
    null_bitmap: Option<Bitmap>,
}

pub type ArrayDataRef = Arc<ArrayDataRepr>;
