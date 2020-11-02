use crate::memory::Buffer;
use crate::util::bit_util;

#[derive(PartialEq, Debug, Clone)]
pub struct Bitmap {
    pub(crate) bits: Buffer,
}

#[allow(dead_code)]
impl Bitmap {
    pub fn new(num_bits: usize) -> Self {
        let num_bytes = num_bits / 8 + if num_bits % 8 > 0 { 1 } else { 0 };
        let r = num_bytes % 64;
        let len = if r == 0 {
            num_bytes
        } else {
            num_bytes + 64 - r
        };
        let v = vec![255; len];
        debug_assert!(v.len() <= usize::max_value() / 8 + 1);
        // The following is safe because the assertion above holds.
        let bits = unsafe { Buffer::from_small_slice(&v[..]) };
        Self { bits }
    }

    pub fn len(&self) -> usize {
        self.bits.len()
    }

    pub fn is_set(&self, i: usize) -> bool {
        assert!(i < (self.bits.len() << 3));
        unsafe { bit_util::get_bit_raw(self.bits.raw_data(), i) }
    }
}

impl From<Buffer> for Bitmap {
    fn from(buf: Buffer) -> Self {
        Self { bits: buf }
    }
}
