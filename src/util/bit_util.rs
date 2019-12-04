static BIT_MASK: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Returns the nearest number that is `>=` than `num` and is a multiple of 64
#[inline]
pub fn round_upto_multiple_of_64(num: usize) -> usize {
    round_upto_power_of_2(num, 64)
}

/// Returns the nearest multiple of `factor` that is `>=` than `num`. Here `factor` must
/// be a power of 2.
fn round_upto_power_of_2(num: usize, factor: usize) -> usize {
    debug_assert!(factor > 0 && (factor & (factor - 1)) == 0);
    (num + (factor - 1)) & !(factor - 1)
}

/// Returns whether bit at position `i` in `data` is set or not.
///
/// # Safety
///
/// Note this doesn't do any bound checking, for performance reason. The caller is
/// responsible to guarantee that `i` is within bounds.
#[inline]
pub unsafe fn get_bit_raw(data: *const u8, i: usize) -> bool {
    (*data.add(i >> 3) & BIT_MASK[i & 7]) != 0
}
