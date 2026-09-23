use onnx_extractor::DataType;
use std::{ffi::c_void, ptr};
use vulkanalia::vk::{self, DeviceV1_0, DeviceV1_4, Handle};

use crate::{
    gpu::{Gpu, memory::GpuMemory},
    instruction::GpuShader,
    utils::error::VKMLError,
};

impl Gpu {
    fn create_pipeline(
        &self,
        shader_code: &[u8],
        local_size: [u32; 3],
        binding_count: usize,
    ) -> Result<vk::Pipeline, VKMLError> {
        unsafe {
            // ensure the shader byte length is a multiple of 4 (SPIR-V is in 32-bit words)
            if !shader_code.len().is_multiple_of(4) {
                return Err(VKMLError::Gpu(
                    "shader byte length must be a multiple of 4".to_string(),
                ));
            }

            let shader_info = vk::ShaderModuleCreateInfo {
                s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::ShaderModuleCreateFlags::empty(),
                code_size: shader_code.len(),
                code: shader_code.as_ptr() as *const u32,
            };

            let shader_module = self.device.create_shader_module(&shader_info, None)?;

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

            let pipeline_layout = self.get_pipeline_layout(binding_count);

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
                layout: pipeline_layout,
                base_pipeline_handle: vk::Pipeline::null(),
                base_pipeline_index: -1,
            };

            let pipeline = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)?
                .0[0];

            self.device.destroy_shader_module(shader_module, None);

            Ok(pipeline)
        }
    }

    pub fn get_or_create_slang_pipeline(
        &self,
        op: GpuShader,
        dtype: DataType,
        local_size: [u32; 3],
    ) -> vk::Pipeline {
        let key = (op, dtype, local_size);

        if let Some(&pipeline) = self.pipelines_slang.read().unwrap().get(&key) {
            return pipeline;
        }

        let compiled_blob = self
            .slang
            .compile(op, dtype)
            .unwrap_or_else(|e| panic!("Slang compilation failed for {:?}: {}", op, e));

        let pipeline = self
            .create_pipeline(compiled_blob.as_slice(), local_size, op.binding_count())
            .unwrap_or_else(|_| {
                panic!(
                    "Slang Pipeline creation failed for {:?} with workgroup {:?}",
                    op, local_size
                )
            });

        self.pipelines_slang
            .write()
            .unwrap()
            .entry(key)
            .or_insert(pipeline);
        pipeline
    }

    pub fn get_pipeline_layout(&self, binding_count: usize) -> vk::PipelineLayout {
        assert!(
            binding_count < self.pipeline_layouts.len(),
            "Binding count {} exceeds maximum {}",
            binding_count,
            self.pipeline_layouts.len() - 1
        );

        *self.pipeline_layouts[binding_count].get_or_init(|| unsafe {
            let descriptor_set_layout = self.get_descriptor_set_layout(binding_count);

            // 128 bytes is the minimum guaranteed push constant space for the vulkan spec
            let push_constant_range = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: 128,
            };

            let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
                s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::PipelineLayoutCreateFlags::empty(),
                set_layout_count: 1,
                set_layouts: &descriptor_set_layout,
                push_constant_range_count: 1,
                push_constant_ranges: &push_constant_range,
            };

            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Failed to create pipeline layout")
        })
    }

    fn get_descriptor_set_layout(&self, binding_count: usize) -> vk::DescriptorSetLayout {
        *self.descriptor_set_layouts[binding_count].get_or_init(|| unsafe {
            // N identical storage buffer bindings
            let bindings: Box<[vk::DescriptorSetLayoutBinding]> = (0..binding_count)
                .map(|i| vk::DescriptorSetLayoutBinding {
                    binding: i as u32,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    immutable_samplers: ptr::null(),
                })
                .collect();

            let descriptor_layout_info = vk::DescriptorSetLayoutCreateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                next: ptr::null(),
                flags: vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR,
                binding_count: bindings.len() as u32,
                bindings: bindings.as_ptr(),
            };

            self.device
                .create_descriptor_set_layout(&descriptor_layout_info, None)
                .expect("Failed to create descriptor set layout")
        })
    }

    pub fn bind_slang_compute_pipeline(
        &self,
        command_buffer: vk::CommandBuffer,
        op: GpuShader,
        dtype: DataType,
        local_size: [u32; 3],
    ) {
        unsafe {
            let pipeline = self.get_or_create_slang_pipeline(op, dtype, local_size);
            self.device
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
        }
    }

    // bind GPU storage buffers to descriptor set bindings
    pub fn bind_storage_buffers(&self, command_buffer: vk::CommandBuffer, buffers: &[&GpuMemory]) {
        unsafe {
            let buffer_infos: Box<_> = buffers
                .iter()
                .map(|mem| vk::DescriptorBufferInfo {
                    buffer: mem.buffer,
                    offset: 0,
                    range: mem.size,
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

            self.device.cmd_push_descriptor_set(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.get_pipeline_layout(buffers.len()),
                0,
                &write_descriptor_sets,
            );
        }
    }

    /// Bind GPU storage buffers (supporting Option<&GPUMemory>) to descriptor set bindings.
    /// Optional buffers will be bound as null buffers with size 0.
    pub fn bind_storage_buffers_optional(
        &self,
        command_buffer: vk::CommandBuffer,
        buffers: &[Option<&GpuMemory>],
    ) {
        unsafe {
            let buffer_infos: Box<_> = buffers
                .iter()
                .map(|mem_opt| {
                    if let Some(mem) = mem_opt {
                        vk::DescriptorBufferInfo {
                            buffer: mem.buffer,
                            offset: 0,
                            range: mem.size,
                        }
                    } else {
                        vk::DescriptorBufferInfo {
                            buffer: vk::Buffer::null(),
                            offset: 0,
                            range: 0,
                        }
                    }
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

            self.device.cmd_push_descriptor_set(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.get_pipeline_layout(buffers.len()),
                0,
                &write_descriptor_sets,
            );
        }
    }

    pub fn bind_push_constants(
        &self,
        command_buffer: vk::CommandBuffer,
        op: GpuShader,
        data: &[u8],
    ) {
        unsafe {
            self.device.cmd_push_constants(
                command_buffer,
                self.get_pipeline_layout(op.binding_count()),
                vk::ShaderStageFlags::COMPUTE,
                0,
                data,
            );
        }
    }
}
