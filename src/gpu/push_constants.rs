use crate::{ComputeManager, gpu::device::MAX_PUSH_CONSTANT_BYTES, tensor_graph::TensorId};

const MAX_BUFFERS: usize = 4;

pub struct PushConstants {
    data: [u8; MAX_PUSH_CONSTANT_BYTES],
    len: usize,
    buffers: [Option<TensorId>; MAX_BUFFERS],
    num_buffers: usize,
}

impl PushConstants {
    pub fn new() -> Self {
        Self {
            data: [0; MAX_PUSH_CONSTANT_BYTES],
            len: 0,
            buffers: [None; MAX_BUFFERS],
            num_buffers: 0,
        }
    }

    pub fn buffer(mut self, tensor: impl Into<Option<TensorId>>) -> Self {
        assert!(
            self.num_buffers < MAX_BUFFERS,
            "Push constants buffer limit exceeded"
        );
        self.buffers[self.num_buffers] = tensor.into();
        self.num_buffers += 1;
        self.push(0u64)
    }

    pub fn push<T: Copy>(mut self, val: T) -> Self {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        let start = (self.len + align - 1) & !(align - 1);
        assert!(
            start + size <= MAX_PUSH_CONSTANT_BYTES,
            "Push constants limit exceeded"
        );
        unsafe {
            std::ptr::write_unaligned(self.data.as_mut_ptr().add(start) as *mut T, val);
        }
        self.len = start + size;
        self
    }

    pub fn resolve_pointers(&mut self, cm: &ComputeManager) {
        for (i, &tensor_id) in self.buffers[..self.num_buffers].iter().enumerate() {
            let addr = tensor_id
                .map(|id| cm.tensor_read(id).get_gpu_memory_or_panic().device_address)
                .unwrap_or(0);
            self.data[i * 8..(i + 1) * 8].copy_from_slice(&addr.to_ne_bytes());
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data[..self.len]
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
