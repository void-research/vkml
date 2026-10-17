use std::{ffi::c_void, ptr};
use vulkanalia::vk::{self, DeviceV1_0, DeviceV1_3, DeviceV1_4, Handle};

use crate::{ComputeManager, gpu::Gpu, instruction::VkOperation, tensor_graph::TensorId};

impl Gpu {
    fn create_pipeline(&self, shader_code: &[u8], local_size: [u32; 3]) -> vk::Pipeline {
        unsafe {
            let shader_info = vk::ShaderModuleCreateInfo {
                s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::ShaderModuleCreateFlags::empty(),
                code_size: shader_code.len(),
                code: shader_code.as_ptr() as *const u32,
            };

            let shader_module = self
                .device
                .create_shader_module(&shader_info, None)
                .expect("Failed to create shader module");

            // Prepare specialization constants so shaders compiled to use
            // specialization IDs 0,1,2 for local_size_x/y/z will receive
            // the chosen local workgroup sizes at pipeline creation time.
            let spec_entries = [
                vk::SpecializationMapEntry {
                    constant_id: 0,
                    offset: 0,
                    size: 4,
                },
                vk::SpecializationMapEntry {
                    constant_id: 1,
                    offset: 4,
                    size: 4,
                },
                vk::SpecializationMapEntry {
                    constant_id: 2,
                    offset: 8,
                    size: 4,
                },
            ];

            let spec_info = vk::SpecializationInfo {
                map_entry_count: spec_entries.len() as u32,
                map_entries: spec_entries.as_ptr(),
                data_size: (local_size.len() * std::mem::size_of::<u32>()),
                data: local_size.as_ptr() as *const c_void,
            };

            let pipeline_info = vk::ComputePipelineCreateInfo {
                s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::PipelineCreateFlags::empty(),
                stage: vk::PipelineShaderStageCreateInfo {
                    s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                    next: std::ptr::null(),
                    flags: vk::PipelineShaderStageCreateFlags::empty(),
                    stage: vk::ShaderStageFlags::COMPUTE,
                    module: shader_module,
                    name: c"main".as_ptr(),
                    specialization_info: &spec_info,
                },
                layout: self.pipeline_layout,
                base_pipeline_handle: vk::Pipeline::null(),
                base_pipeline_index: -1,
            };

            let pipeline = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .expect("Failed to create compute pipeline")
                .0[0];

            self.device.destroy_shader_module(shader_module, None);

            pipeline
        }
    }

    fn get_or_create_pipeline(&self, spv: &'static [u8], local_size: [u32; 3]) -> vk::Pipeline {
        let key = (spv.as_ptr() as usize, local_size);
        let mut pipelines = self.pipelines.lock().unwrap();

        if let Some(&pipeline) = pipelines.get(&key) {
            return pipeline;
        }

        let pipeline = self.create_pipeline(spv, local_size);
        pipelines.insert(key, pipeline);
        pipeline
    }

    fn bind_compute_pipeline(
        &self,
        command_buffer: vk::CommandBuffer,
        spv: &'static [u8],
        local_size: [u32; 3],
    ) {
        let pipeline = self.get_or_create_pipeline(spv, local_size);
        unsafe {
            self.device
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
        }
    }

    // bind GPU storage buffers to descriptor set bindings
    fn bind_storage_buffers(
        &self,
        command_buffer: vk::CommandBuffer,
        storage_buffers: &[Option<TensorId>],
        cm: &ComputeManager,
    ) {
        let buffer_infos: Box<_> = storage_buffers
            .iter()
            .map(|id_opt| match id_opt {
                Some(id) => {
                    let mem = cm.tensor_read(*id).get_gpu_memory_or_panic();
                    vk::DescriptorBufferInfo {
                        buffer: mem.buffer,
                        offset: 0,
                        range: mem.size,
                    }
                }
                None => vk::DescriptorBufferInfo {
                    buffer: vk::Buffer::null(),
                    offset: 0,
                    range: 0,
                },
            })
            .collect();

        let write_descriptor_sets: Box<_> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, info)| vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                next: ptr::null(),
                dst_set: vk::DescriptorSet::null(),
                dst_binding: i as u32,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                buffer_info: info,
                image_info: ptr::null(),
                texel_buffer_view: ptr::null(),
            })
            .collect();

        unsafe {
            self.device.cmd_push_descriptor_set(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &write_descriptor_sets,
            );
        }
    }

    fn bind_push_constants(&self, command_buffer: vk::CommandBuffer, data: &[u8]) {
        unsafe {
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                data,
            );
        }
    }

    fn dispatch(
        &self,
        command_buffer: vk::CommandBuffer,
        local_size: [u32; 3],
        work_size: [u32; 3],
    ) {
        let dispatch_x = work_size[0].div_ceil(local_size[0]);
        let dispatch_y = work_size[1].div_ceil(local_size[1]);
        let dispatch_z = work_size[2].div_ceil(local_size[2]);

        unsafe {
            self.device
                .cmd_dispatch(command_buffer, dispatch_x, dispatch_y, dispatch_z);
        }
    }

    pub fn record_vk_operation(
        &self,
        command_buffer: vk::CommandBuffer,
        op: VkOperation,
        cm: &ComputeManager,
    ) {
        match op {
            VkOperation::Compute {
                spv,
                local_size,
                work_size,
                push_constants,
                storage_buffers,
            } => {
                self.bind_compute_pipeline(command_buffer, spv, local_size);
                self.bind_storage_buffers(command_buffer, &storage_buffers, cm);

                if !push_constants.is_empty() {
                    self.bind_push_constants(command_buffer, push_constants.as_bytes());
                }

                self.dispatch(command_buffer, local_size, work_size);
            }
            VkOperation::ClearAndCompute {
                clear_dst,
                spv,
                local_size,
                work_size,
                push_constants,
                storage_buffers,
            } => {
                let dst_mem = cm.tensor_read(clear_dst).get_gpu_memory_or_panic();

                unsafe {
                    self.device
                        .cmd_fill_buffer(command_buffer, dst_mem.buffer, 0, dst_mem.size, 0);

                    let buffer_barrier = vk::BufferMemoryBarrier2 {
                        s_type: vk::StructureType::BUFFER_MEMORY_BARRIER_2,
                        next: ptr::null(),
                        src_stage_mask: vk::PipelineStageFlags2::ALL_TRANSFER,
                        src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                        dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        dst_access_mask: vk::AccessFlags2::SHADER_READ
                            | vk::AccessFlags2::SHADER_WRITE,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        buffer: dst_mem.buffer,
                        offset: 0,
                        size: dst_mem.size,
                    };
                    let dep_info = vk::DependencyInfo {
                        s_type: vk::StructureType::DEPENDENCY_INFO,
                        next: ptr::null(),
                        dependency_flags: vk::DependencyFlags::empty(),
                        memory_barrier_count: 0,
                        memory_barriers: ptr::null(),
                        buffer_memory_barrier_count: 1,
                        buffer_memory_barriers: &buffer_barrier,
                        image_memory_barrier_count: 0,
                        image_memory_barriers: ptr::null(),
                    };
                    self.device.cmd_pipeline_barrier2(command_buffer, &dep_info);
                }

                self.record_vk_operation(
                    command_buffer,
                    VkOperation::Compute {
                        spv,
                        local_size,
                        work_size,
                        push_constants,
                        storage_buffers,
                    },
                    cm,
                );
            }
            VkOperation::Copy { src, dst } => {
                let src_mem = cm.tensor_read(src).get_gpu_memory_or_panic();
                let dst_mem = cm.tensor_read(dst).get_gpu_memory_or_panic();

                let copy_region = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: src_mem.size,
                };

                unsafe {
                    self.device.cmd_copy_buffer(
                        command_buffer,
                        src_mem.buffer,
                        dst_mem.buffer,
                        &[copy_region],
                    );
                }
            }
        }
    }
}
