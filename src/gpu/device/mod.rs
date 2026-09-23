use onnx_extractor::DataType;
use std::{
    collections::HashMap,
    ffi::{CStr, c_void},
    ptr,
    sync::{Arc, Mutex, OnceLock, RwLock},
};
use vulkanalia::{
    Device, Instance,
    vk::{self, DeviceV1_0, DeviceV1_3, Handle, InstanceV1_0, InstanceV1_1},
};

use crate::{
    compute::memory_tracker::MemoryTracker,
    gpu::{VkExtensions, device::allocator::StagingResources},
    instruction::GpuShader,
    slang::SlangCompiler,
    utils::error::VKMLError,
};

pub mod allocator;
pub mod pipeline;
pub mod workgroup;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostAccessMode {
    /// Tensors remain device local and host access occurs through staging.
    DeviceLocalWithStaging,
    /// All tensors are host visible; staging is unnecessary.
    DirectAllHostVisible,
}

pub struct Gpu {
    device_name: String,
    properties: vk::PhysicalDeviceProperties,
    subgroup_properties: vk::PhysicalDeviceSubgroupProperties,
    push_descriptor_properties: vk::PhysicalDevicePushDescriptorProperties,

    has_compute: bool,
    max_compute_queue_count: u32,
    host_visible_device_local_bytes: u64, // bytes available on the device that satisfy DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT

    physical_device: vk::PhysicalDevice,
    compute_queue: vk::Queue,
    memory_tracker: MemoryTracker,
    extensions: VkExtensions,

    host_access_mode: HostAccessMode,
    staging_resources: OnceLock<Mutex<StagingResources>>,

    // Drop order matters: fields drop top-to-bottom
    pipelines_slang: RwLock<HashMap<(GpuShader, DataType, [u32; 3]), vk::Pipeline>>,
    descriptor_set_layouts: Box<[OnceLock<vk::DescriptorSetLayout>]>,
    pipeline_layouts: Box<[OnceLock<vk::PipelineLayout>]>,
    command_pool: vk::CommandPool,
    device: Arc<Device>,
    instance: Arc<Instance>,
    slang: Arc<SlangCompiler>,
}

impl Gpu {
    pub fn new(
        instance: Arc<Instance>,
        physical_device: vk::PhysicalDevice,
        slang: Arc<SlangCompiler>,
    ) -> Result<Self, VKMLError> {
        unsafe {
            let queue_families =
                instance.get_physical_device_queue_family_properties(physical_device);

            let device_extensions =
                instance.enumerate_device_extension_properties(physical_device, None)?;
            let vk_extensions = VkExtensions::from_extension_properties(
                &instance,
                physical_device,
                &device_extensions,
            )?;

            let queue_family_index = queue_families
                .iter()
                .enumerate()
                .find(|(_, properties)| properties.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(index, _)| index as u32)
                .expect("No compute queue family found on device");

            // Request a single compute queue
            let queue_info = vk::DeviceQueueCreateInfo {
                s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::DeviceQueueCreateFlags::empty(),
                queue_family_index,
                queue_count: 1,
                queue_priorities: &1.0f32,
            };

            let device_features = vk::PhysicalDeviceFeatures::default();

            // Prepare extension name pointers and p_next feature chain.
            // Keep extras alive until after device creation so the pointer arrays and structs remain valid.
            let extras = vk_extensions.prepare_device_create();

            // currently only send to single compute queue
            //driver will still parallelise work if possible
            let device_create_info = vk::DeviceCreateInfo {
                s_type: vk::StructureType::DEVICE_CREATE_INFO,
                next: extras.device_create_next(),
                flags: vk::DeviceCreateFlags::empty(),
                queue_create_info_count: 1,
                queue_create_infos: &queue_info,
                enabled_layer_count: 0,
                enabled_layer_names: std::ptr::null(),
                enabled_extension_count: extras.name_ptrs.len() as u32,
                enabled_extension_names: extras.name_ptrs.as_ptr(),
                enabled_features: &device_features,
            };

            let device =
                Arc::new(instance.create_device(physical_device, &device_create_info, None)?);

            // get the single compute queue
            let compute_queue = device.get_device_queue(queue_family_index, 0);

            // query device properties
            let command_pool_info = vk::CommandPoolCreateInfo {
                s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
                next: std::ptr::null(),
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                queue_family_index,
            };
            let command_pool = device.create_command_pool(&command_pool_info, None)?;

            let mut subgroup_properties = vk::PhysicalDeviceSubgroupProperties {
                s_type: vk::StructureType::PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
                next: std::ptr::null_mut(),
                subgroup_size: 0,
                supported_stages: vk::ShaderStageFlags::empty(),
                supported_operations: vk::SubgroupFeatureFlags::empty(),
                quad_operations_in_all_stages: vk::FALSE,
            };

            let mut push_props = vk::PhysicalDevicePushDescriptorProperties {
                s_type: vk::StructureType::PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES,
                next: &mut subgroup_properties as *mut _ as *mut c_void,
                max_push_descriptors: 0,
            };

            let mut props2 = vk::PhysicalDeviceProperties2 {
                s_type: vk::StructureType::PHYSICAL_DEVICE_PROPERTIES_2,
                next: &mut push_props as *mut _ as *mut c_void,
                properties: Default::default(),
            };

            instance.get_physical_device_properties2(physical_device, &mut props2);
            let properties = props2.properties;
            let device_name = CStr::from_ptr(properties.device_name.as_ptr())
                .to_string_lossy()
                .into_owned();

            // Memory properties, memory_budget if extension is available
            let mut budget_props = vk::PhysicalDeviceMemoryBudgetPropertiesEXT {
                s_type: vk::StructureType::PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
                next: std::ptr::null_mut(),
                heap_budget: [0; vk::MAX_MEMORY_HEAPS],
                heap_usage: [0; vk::MAX_MEMORY_HEAPS],
            };

            let mut memory_props2 = vk::PhysicalDeviceMemoryProperties2 {
                s_type: vk::StructureType::PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
                next: if vk_extensions.has_memory_budget() {
                    &mut budget_props as *mut _ as *mut c_void
                } else {
                    std::ptr::null_mut()
                },
                memory_properties: Default::default(),
            };

            instance.get_physical_device_memory_properties2(physical_device, &mut memory_props2);
            let memory_properties = memory_props2.memory_properties;

            // caches for descriptor set layouts and pipeline layouts indexed by binding count
            // uses max_push_descriptors as a reasonable upper bound
            let max_bindings = push_props.max_push_descriptors as usize;
            let descriptor_set_layouts = (0..=max_bindings).map(|_| OnceLock::new()).collect();
            let pipeline_layouts = (0..=max_bindings).map(|_| OnceLock::new()).collect();

            let target_props = vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT;

            let mut host_visible_device_local_bytes: u64 = 0;
            for memory_type in memory_properties.memory_types.iter() {
                if memory_type.property_flags.contains(target_props) {
                    let heap_index = memory_type.heap_index as usize;
                    let heap_size = memory_properties.memory_heaps[heap_index].size;
                    host_visible_device_local_bytes =
                        host_visible_device_local_bytes.max(heap_size);
                }
            }

            // check compute capability
            let (has_compute, max_compute_queue_count) = queue_families
                .iter()
                .find(|props| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|props| (true, props.queue_count))
                .unwrap_or((false, 0));

            // calculate available memory budget
            let device_local_heap_index = (0..memory_properties.memory_type_count)
                .find(|&i| {
                    let memory_type = memory_properties.memory_types[i as usize];
                    memory_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .map(|i| memory_properties.memory_types[i as usize].heap_index)
                .unwrap_or(0);

            let memory_budget = if vk_extensions.has_memory_budget() {
                // from VK_EXT_memory_budget, scaled to 95% to account for overhead
                let reported_budget =
                    budget_props.heap_budget[device_local_heap_index as usize] as u128;
                ((reported_budget * 95) / 100) as u64
            } else {
                // use 80% of total device memory
                let total_memory =
                    memory_properties.memory_heaps[device_local_heap_index as usize].size as u128;
                ((total_memory * 80) / 100) as u64
            };

            let host_access_mode = if host_visible_device_local_bytes >= memory_budget {
                HostAccessMode::DirectAllHostVisible
            } else {
                HostAccessMode::DeviceLocalWithStaging
            };

            Ok(Self {
                device_name,
                properties,
                subgroup_properties,
                push_descriptor_properties: push_props,

                has_compute,
                max_compute_queue_count,
                host_visible_device_local_bytes,

                physical_device,
                compute_queue,
                memory_tracker: MemoryTracker::new(memory_budget),
                extensions: vk_extensions,

                host_access_mode,
                staging_resources: OnceLock::new(),

                pipelines_slang: RwLock::new(HashMap::new()),
                descriptor_set_layouts,
                pipeline_layouts,
                command_pool,
                device,
                instance,
                slang,
            })
        }
    }

    pub fn host_access_mode(&self) -> HostAccessMode {
        self.host_access_mode
    }

    pub fn get_device(&self) -> &Device {
        &self.device
    }

    pub fn get_command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    pub(crate) fn memory_allocate_usage(&self, bytes: vk::DeviceSize) {
        self.memory_tracker.allocate(bytes);
    }

    pub(crate) fn memory_free_usage(&self, bytes: vk::DeviceSize) {
        self.memory_tracker.deallocate(bytes);
    }

    pub fn memory_available(&self) -> u64 {
        self.memory_tracker.get_available()
    }

    pub fn memory_total(&self) -> u64 {
        self.memory_tracker.get_maximum()
    }

    pub fn device_type(&self) -> vk::PhysicalDeviceType {
        self.properties.device_type
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn max_workgroup_size(&self) -> [u32; 3] {
        self.properties.limits.max_compute_work_group_size
    }

    pub fn max_workgroup_invocations(&self) -> u32 {
        self.properties.limits.max_compute_work_group_invocations
    }

    pub fn max_shared_memory_size(&self) -> u32 {
        self.properties.limits.max_compute_shared_memory_size
    }

    pub fn subgroup_size(&self) -> u32 {
        self.subgroup_properties.subgroup_size.max(1)
    }

    pub fn supports_subgroup_operations(&self) -> bool {
        self.subgroup_properties
            .supported_operations
            .contains(vk::SubgroupFeatureFlags::ARITHMETIC)
    }

    pub fn get_physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn extensions(&self) -> &VkExtensions {
        &self.extensions
    }

    /// for reusable command buffers, pass `vk::CommandBufferUsageFlags::empty()`
    /// for staging-style record/submit/reset loops, pass `vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT`
    pub fn begin_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        flags: vk::CommandBufferUsageFlags,
    ) -> Result<(), VKMLError> {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo {
                flags,
                ..Default::default()
            };
            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;
        }
        Ok(())
    }

    pub fn end_command_buffer(&self, command_buffer: vk::CommandBuffer) -> Result<(), VKMLError> {
        unsafe {
            self.device.end_command_buffer(command_buffer)?;
        }
        Ok(())
    }

    pub fn submit_with_fence(
        &self,
        command_buffers: &[vk::CommandBuffer],
        fence: Option<vk::Fence>,
    ) -> Result<(), VKMLError> {
        unsafe {
            let submit_info = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                next: ptr::null(),
                wait_semaphore_count: 0,
                wait_semaphores: ptr::null(),
                wait_dst_stage_mask: ptr::null(),
                command_buffer_count: command_buffers.len() as u32,
                command_buffers: command_buffers.as_ptr(),
                signal_semaphore_count: 0,
                signal_semaphores: ptr::null(),
            };

            self.device.queue_submit(
                self.compute_queue,
                &[submit_info],
                fence.unwrap_or(vk::Fence::null()),
            )?;
        }

        Ok(())
    }

    pub fn wait_and_reset_fence(&self, fence: vk::Fence) -> Result<(), VKMLError> {
        unsafe {
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.reset_fences(&[fence])?;
        }

        Ok(())
    }

    pub fn create_fence(&self) -> Result<vk::Fence, VKMLError> {
        let fence_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            next: ptr::null(),
            flags: vk::FenceCreateFlags::empty(),
        };

        unsafe { Ok(self.device.create_fence(&fence_info, None)?) }
    }

    // insert a memory barrier ensuring previous compute writes are visible to subsequent compute dispatches
    pub fn barrier_compute_shader_access(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer_barriers: &[vk::BufferMemoryBarrier2],
    ) {
        unsafe {
            let dependency_info = vk::DependencyInfo {
                s_type: vk::StructureType::DEPENDENCY_INFO,
                next: ptr::null(),
                dependency_flags: vk::DependencyFlags::empty(),
                memory_barrier_count: 0,
                memory_barriers: ptr::null(),
                buffer_memory_barrier_count: buffer_barriers.len() as u32,
                buffer_memory_barriers: buffer_barriers.as_ptr(),
                image_memory_barrier_count: 0,
                image_memory_barriers: ptr::null(),
            };

            self.device
                .cmd_pipeline_barrier2(command_buffer, &dependency_info);
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            if let Some(staging_mutex) = self.staging_resources.get()
                && let Ok(res) = staging_mutex.lock()
            {
                self.device.destroy_fence(res.fence, None);
                self.device.destroy_buffer(res.buffer, None);
                self.device.free_memory(res.memory, None);
            }

            self.device.destroy_command_pool(self.command_pool, None);

            if let Ok(pipelines) = self.pipelines_slang.read() {
                for &pipeline in pipelines.values() {
                    self.device.destroy_pipeline(pipeline, None);
                }
            }

            for cell in self.descriptor_set_layouts.iter() {
                if let Some(&dsl) = cell.get() {
                    self.device.destroy_descriptor_set_layout(dsl, None);
                }
            }
            for cell in self.pipeline_layouts.iter() {
                if let Some(&pl) = cell.get() {
                    self.device.destroy_pipeline_layout(pl, None);
                }
            }
        }
    }
}

impl std::fmt::Debug for Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let staging_desc = match self.staging_resources.get().and_then(|m| m.lock().ok()) {
            Some(res) => format!(
                "Some {{ size_bytes: {}, properties: {:?} }}",
                res.size, res.properties
            ),
            None => "None".to_string(),
        };
        let in_use = self.memory_tracker.get_current();
        let total = self.memory_tracker.get_maximum();
        let mem_percent = if total == 0 {
            0.0
        } else {
            (in_use as f64 / total as f64) * 100.0
        };

        f.debug_struct("Gpu")
            .field("name", &self.device_name)
            .field("device_type", &self.properties.device_type)
            .field("has_compute", &self.has_compute)
            .field("memory_budget", &self.memory_tracker.get_available())
            .field("memory_in_use", &in_use)
            .field(
                "memory_in_use_as_percent",
                &format_args!("{:.2}%", mem_percent),
            )
            .field(
                "max_workgroup_count",
                &self.properties.limits.max_compute_work_group_count,
            )
            .field(
                "max_workgroup_size",
                &self.properties.limits.max_compute_work_group_size,
            )
            .field(
                "max_workgroup_invocations",
                &self.properties.limits.max_compute_work_group_invocations,
            )
            .field("max_compute_queue_count", &self.max_compute_queue_count)
            .field(
                "max_shared_memory_size",
                &self.properties.limits.max_compute_shared_memory_size,
            )
            .field(
                "max_push_descriptors",
                &self.push_descriptor_properties.max_push_descriptors,
            )
            .field("subgroup_size", &self.subgroup_properties.subgroup_size)
            .field(
                "host_visible_device_local_bytes",
                &self.host_visible_device_local_bytes,
            )
            .field("host_access_mode", &self.host_access_mode)
            .field("staging_buffer", &staging_desc)
            .field("extensions", &self.extensions)
            .finish()
    }
}
