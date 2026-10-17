use crate::gpu::device::MAX_PUSH_CONSTANT_BYTES;

#[repr(C, align(4))]
pub struct PushConstants {
    data: [u8; MAX_PUSH_CONSTANT_BYTES],
    len: usize,
}

impl PushConstants {
    pub const fn empty() -> Self {
        Self {
            data: [0; MAX_PUSH_CONSTANT_BYTES],
            len: 0,
        }
    }

    pub fn from_struct<T: Sized>(val: &T) -> Self {
        let size = std::mem::size_of::<T>();
        assert!(
            size <= MAX_PUSH_CONSTANT_BYTES,
            "Push constants size {size} exceeds {MAX_PUSH_CONSTANT_BYTES}-byte limit",
        );
        let mut data = [0u8; MAX_PUSH_CONSTANT_BYTES];
        let bytes = crate::utils::as_bytes(val);
        data[..size].copy_from_slice(bytes);
        Self { data, len: size }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data[..self.len]
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
