use std::alloc::Layout;

pub const ALIGNMENT: usize = 64;

pub fn allocate_aligned(size: usize) -> *mut u8 {
    unsafe {
        let layout = Layout::from_size_align_unchecked(size, ALIGNMENT);
        ::std::alloc::alloc(layout)
    }
}

pub fn free_aligned(p: *mut u8, size: usize) {
    unsafe {
        ::std::alloc::dealloc(p, Layout::from_size_align_unchecked(size, ALIGNMENT));
    }
}

extern "C" {
    pub fn memcmp(p1: *const u8, p2: *const u8, len: usize) -> i32;
}

/// Check if the pointer `p` is aligned to offset `a`.
pub fn is_aligned<T>(p: *const T, a: usize) -> bool {
    let a_minus_one = a.wrapping_sub(1);
    let pmoda = p as usize & a_minus_one;
    pmoda == 0
}
