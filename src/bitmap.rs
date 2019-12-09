use crate::buffer::Buffer;
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
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(255); // 1 is not null
        }
        Self {
            bits: Buffer::from(&v[..]),
        }
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