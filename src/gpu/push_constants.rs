// if ever needed, vk 1.4 does give us 256 as the new minimum
#[repr(C, align(4))]
pub struct PushConstants {
    data: [u8; 128],
    len: u32,
}

impl PushConstants {
    pub const fn empty() -> Self {
        Self {
            data: [0; 128],
            len: 0,
        }
    }

    pub fn from_struct<T: Sized>(val: &T) -> Self {
        let size = std::mem::size_of::<T>();
        assert!(
            size <= 128,
            "Push constants size {size} exceeds 128-byte limit",
        );
        let mut data = [0u8; 128];
        let bytes = crate::utils::as_bytes(val);
        data[..size].copy_from_slice(bytes);
        Self {
            data,
            len: size as u32,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data[..self.len as usize]
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
