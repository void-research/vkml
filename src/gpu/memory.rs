use std::sync::Arc;

use vulkanalia::{vk, vk::DeviceV1_0};

use crate::VKMLError;

use super::device::Gpu;

pub struct GpuMemory {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub properties: vk::MemoryPropertyFlags,
    gpu: Arc<Gpu>,
}

impl GpuMemory {
    pub fn new(
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        size: vk::DeviceSize,
        properties: vk::MemoryPropertyFlags,
        gpu: &Arc<Gpu>,
    ) -> Self {
        gpu.memory_allocate_usage(size);
        Self {
            buffer,
            memory,
            size,
            properties,
            gpu: Arc::clone(gpu),
        }
    }

    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }

    /// Copy raw bytes into GPU memory.
    pub fn copy_into(&self, data: &[u8]) -> Result<(), VKMLError> {
        self.gpu.write_memory(self, data)
    }

    /// Read raw bytes from GPU memory into CPU memory.
    pub fn read_memory(&self) -> Result<Box<[u8]>, VKMLError> {
        self.gpu.read_memory(self)
    }
}

impl Drop for GpuMemory {
    fn drop(&mut self) {
        unsafe {
            let device = self.gpu.get_device();
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
        self.gpu.memory_free_usage(self.size);
    }
}
