use std::{
    cmp,
    sync::{Arc, Mutex},
};
use vulkanalia::vk::{self, DeviceV1_0, InstanceV1_0};

use crate::{
    gpu::{Gpu, HostAccessMode, memory::GpuMemory},
    utils::error::VKMLError,
};

pub(crate) struct StagingResources {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub properties: vk::MemoryPropertyFlags,
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
}

impl Gpu {
    /// Allocate uninitialised GPU memory based on this GPU's HostAccessMode.
    pub fn allocate_uninitialised(self: &Arc<Self>, bytes: usize) -> Result<GpuMemory, VKMLError> {
        let size_in_bytes = bytes as vk::DeviceSize;

        unsafe {
            let buffer_info = vk::BufferCreateInfo {
                s_type: vk::StructureType::BUFFER_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::BufferCreateFlags::empty(),
                size: size_in_bytes,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 0,
                queue_family_indices: std::ptr::null(),
            };

            let buffer = self.get_device().create_buffer(&buffer_info, None)?;
            let mem_requirements = self.get_device().get_buffer_memory_requirements(buffer);

            let properties = match self.host_access_mode {
                HostAccessMode::DirectAllHostVisible => {
                    vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT
                        | vk::MemoryPropertyFlags::DEVICE_LOCAL
                }
                HostAccessMode::DeviceLocalWithStaging => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            };

            let memory_type =
                self.find_memory_type(mem_requirements.memory_type_bits, properties)?;

            let alloc_info = vk::MemoryAllocateInfo {
                s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                next: std::ptr::null(),
                allocation_size: mem_requirements.size,
                memory_type_index: memory_type,
            };

            let memory = self.get_device().allocate_memory(&alloc_info, None)?;
            self.get_device().bind_buffer_memory(buffer, memory, 0)?;

            Ok(GpuMemory::new(
                buffer,
                memory,
                size_in_bytes,
                properties,
                self,
            ))
        }
    }

    /// Allocate GPU memory and upload raw bytes into it.
    pub fn allocate(self: &Arc<Self>, bytes: &[u8]) -> Result<GpuMemory, VKMLError> {
        let dest = self.allocate_uninitialised(bytes.len())?;
        self.write_memory(&dest, bytes)?;
        Ok(dest)
    }

    /// Write raw bytes into GPU memory, choosing direct mapping or staging appropriately.
    pub fn write_memory(&self, dest: &GpuMemory, data: &[u8]) -> Result<(), VKMLError> {
        let data_size = data.len() as vk::DeviceSize;
        if data_size > dest.size {
            return Err(VKMLError::Gpu(format!(
                "Data size {} exceeds GPU buffer size {}",
                data_size, dest.size
            )));
        }

        if dest
            .properties
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            unsafe {
                let data_ptr = self.get_device().map_memory(
                    dest.memory,
                    0,
                    data_size,
                    vk::MemoryMapFlags::empty(),
                )? as *mut u8;

                std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());

                self.get_device().unmap_memory(dest.memory);
            }
            Ok(())
        } else {
            self.write_through_staging(dest, data)
        }
    }

    /// Read raw bytes from GPU memory into CPU memory, choosing direct mapping or staging appropriately.
    pub fn read_memory(&self, source: &GpuMemory) -> Result<Box<[u8]>, VKMLError> {
        if source
            .properties
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            let mut buffer = Box::new_uninit_slice(source.size as usize);

            unsafe {
                let data_ptr = self.get_device().map_memory(
                    source.memory,
                    0,
                    source.size,
                    vk::MemoryMapFlags::empty(),
                )? as *const u8;

                let buffer_ptr = buffer.as_mut_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(data_ptr, buffer_ptr, buffer.len());

                self.get_device().unmap_memory(source.memory);
            }

            Ok(unsafe { buffer.assume_init() })
        } else {
            self.read_through_staging(source)
        }
    }

    fn write_through_staging(&self, dest: &GpuMemory, data: &[u8]) -> Result<(), VKMLError> {
        if data.len() as vk::DeviceSize > dest.size {
            return Err(VKMLError::Gpu(format!(
                "Attempted to write {} bytes into buffer sized {}",
                data.len(),
                dest.size
            )));
        }

        let staging_mutex = self.get_or_create_staging_resources();
        let staging_buffer = staging_mutex.lock().unwrap();
        let staging_size = staging_buffer.size as usize;

        if staging_size == 0 {
            return Err(VKMLError::Gpu(
                "Staging buffer must be at least 1 byte".to_string(),
            ));
        }

        let command_buffer = staging_buffer.command_buffer;
        let fence = staging_buffer.fence;

        unsafe {
            let mut offset = 0usize;
            while offset < data.len() {
                let remaining = data.len() - offset;
                let chunk_size = cmp::min(staging_size, remaining);
                let chunk = &data[offset..offset + chunk_size];

                let data_ptr = self.get_device().map_memory(
                    staging_buffer.memory,
                    0,
                    chunk_size as vk::DeviceSize,
                    vk::MemoryMapFlags::empty(),
                )? as *mut u8;
                std::ptr::copy_nonoverlapping(chunk.as_ptr(), data_ptr, chunk_size);
                self.get_device().unmap_memory(staging_buffer.memory);

                self.begin_command_buffer(
                    command_buffer,
                    vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                )?;

                let copy_region = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: offset as vk::DeviceSize,
                    size: chunk_size as vk::DeviceSize,
                };

                self.get_device().cmd_copy_buffer(
                    command_buffer,
                    staging_buffer.buffer,
                    dest.buffer,
                    &[copy_region],
                );

                self.end_command_buffer(command_buffer)?;

                self.submit_with_fence(&[command_buffer], Some(fence))?;
                self.wait_and_reset_fence(fence)?;

                self.get_device().reset_command_buffer(
                    command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )?;

                offset += chunk_size;
            }
        }

        Ok(())
    }

    fn read_through_staging(&self, source: &GpuMemory) -> Result<Box<[u8]>, VKMLError> {
        let total_bytes = source.size as usize;
        let mut buffer = Box::new_uninit_slice(total_bytes);

        let staging_mutex = self.get_or_create_staging_resources();
        let staging_buffer = staging_mutex.lock().unwrap();
        let staging_size = staging_buffer.size as usize;

        if staging_size == 0 {
            return Err(VKMLError::Gpu(
                "Staging buffer must be at least 1 byte".to_string(),
            ));
        }

        let command_buffer = staging_buffer.command_buffer;
        let fence = staging_buffer.fence;

        unsafe {
            let mut offset = 0usize;
            while offset < total_bytes {
                let remaining = total_bytes - offset;
                let chunk_size = cmp::min(staging_size, remaining);

                self.begin_command_buffer(
                    command_buffer,
                    vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                )?;

                let copy_region = vk::BufferCopy {
                    src_offset: offset as vk::DeviceSize,
                    dst_offset: 0,
                    size: chunk_size as vk::DeviceSize,
                };

                self.get_device().cmd_copy_buffer(
                    command_buffer,
                    source.buffer,
                    staging_buffer.buffer,
                    &[copy_region],
                );

                self.end_command_buffer(command_buffer)?;

                self.submit_with_fence(&[command_buffer], Some(fence))?;
                self.wait_and_reset_fence(fence)?;

                let data_ptr = self.get_device().map_memory(
                    staging_buffer.memory,
                    0,
                    chunk_size as vk::DeviceSize,
                    vk::MemoryMapFlags::empty(),
                )? as *const u8;

                let buffer_ptr = buffer.as_mut_ptr().add(offset) as *mut u8;
                std::ptr::copy_nonoverlapping(data_ptr, buffer_ptr, chunk_size);
                self.get_device().unmap_memory(staging_buffer.memory);

                self.get_device().reset_command_buffer(
                    command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )?;

                offset += chunk_size;
            }
        }

        let output = unsafe { buffer.assume_init() };

        Ok(output)
    }

    fn plan_staging(&self) -> (usize, bool) {
        let total_memory = self.memory_total().max(1);
        let mut size_bytes = (total_memory / 20).max(1); // target: 5% of total memory
        let mut device_local = false;

        if self.host_visible_device_local_bytes > 0 {
            let visibility_threshold = (total_memory / 100).max(1); // require ~1% before preferring device-local
            if self.host_visible_device_local_bytes >= visibility_threshold {
                size_bytes = self.host_visible_device_local_bytes.min(size_bytes);
                device_local = true;
            }
        }

        (size_bytes.max(1) as usize, device_local)
    }

    // Staging buffer is host-visible and sized to 5% of the tracked maximum memory.
    fn get_or_create_staging_resources(&self) -> &Mutex<StagingResources> {
        self.staging_resources.get_or_init(|| unsafe {
            let (staging_size, device_local_staging) = self.plan_staging();

            let buffer_info = vk::BufferCreateInfo {
                s_type: vk::StructureType::BUFFER_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::BufferCreateFlags::empty(),
                size: staging_size as vk::DeviceSize,
                usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 0,
                queue_family_indices: std::ptr::null(),
            };

            let buffer = self
                .get_device()
                .create_buffer(&buffer_info, None)
                .expect("Failed to create staging buffer");
            let mem_requirements = self.get_device().get_buffer_memory_requirements(buffer);

            let requested_properties = if device_local_staging {
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::DEVICE_LOCAL
            } else {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            };

            let memory_type = self
                .find_memory_type(mem_requirements.memory_type_bits, requested_properties)
                .expect("Failed to find suitable memory type for staging buffer");

            let alloc_info = vk::MemoryAllocateInfo {
                s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                next: std::ptr::null(),
                allocation_size: mem_requirements.size,
                memory_type_index: memory_type,
            };

            let memory = self
                .get_device()
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate staging memory");
            self.get_device()
                .bind_buffer_memory(buffer, memory, 0)
                .expect("Failed to bind staging buffer memory");

            self.memory_allocate_usage(staging_size as vk::DeviceSize);

            let command_buffer_info = vk::CommandBufferAllocateInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                next: std::ptr::null(),
                command_pool: self.get_command_pool(),
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
            };

            let command_buffers = self
                .get_device()
                .allocate_command_buffers(&command_buffer_info)
                .expect("Failed to allocate staging command buffer");
            let command_buffer = command_buffers[0];

            let fence_info = vk::FenceCreateInfo {
                s_type: vk::StructureType::FENCE_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::FenceCreateFlags::empty(),
            };
            let fence = self
                .get_device()
                .create_fence(&fence_info, None)
                .expect("Failed to create staging fence");

            Mutex::new(StagingResources {
                buffer,
                memory,
                size: staging_size as vk::DeviceSize,
                properties: requested_properties,
                command_buffer,
                fence,
            })
        })
    }

    fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, VKMLError> {
        unsafe {
            let mem_properties = self
                .instance
                .get_physical_device_memory_properties(self.physical_device);

            for i in 0..mem_properties.memory_type_count {
                if (type_filter & (1 << i)) != 0
                    && mem_properties.memory_types[i as usize]
                        .property_flags
                        .contains(properties)
                {
                    return Ok(i);
                }
            }

            Err(VKMLError::Gpu(format!(
                "Failed to find suitable memory type for filter {:#b} with properties {:?}",
                type_filter, properties
            )))
        }
    }
}
