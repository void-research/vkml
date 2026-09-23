use std::{collections::HashSet, ptr, sync::Arc};

use vulkanalia::{
    Entry, Instance,
    loader::{LIBRARY, LibloadingLoader},
    vk::{self, InstanceV1_0},
};
use zero_pool::global_pool;

use crate::{VKMLError, gpu::device::Gpu, slang::SlangCompiler};

pub struct GpuPool {
    gpus: Vec<Arc<Gpu>>,
    _entry: Entry,
}

impl GpuPool {
    pub fn new(selected: Option<Vec<usize>>) -> Result<Self, VKMLError> {
        unsafe {
            let loader = LibloadingLoader::new(LIBRARY).expect("Failed to load Vulkan library");
            let entry = Entry::new(loader).expect("Failed to create Vulkan entry point");

            let appinfo = vk::ApplicationInfo {
                s_type: vk::StructureType::APPLICATION_INFO,
                next: ptr::null(),
                application_name: c"vkml".as_ptr(),
                application_version: vk::make_version(0, 0, 1),
                engine_name: c"vkml".as_ptr(),
                engine_version: vk::make_version(0, 0, 1),
                api_version: vk::make_version(1, 4, 0),
            };

            let create_info = vk::InstanceCreateInfo {
                s_type: vk::StructureType::INSTANCE_CREATE_INFO,
                next: ptr::null(),
                flags: vk::InstanceCreateFlags::empty(),
                application_info: &appinfo,
                enabled_layer_count: 0,
                enabled_layer_names: ptr::null(),
                enabled_extension_count: 0,
                enabled_extension_names: ptr::null(),
            };

            let instance = Arc::new(entry.create_instance(&create_info, None)?);
            let slang = Arc::new(SlangCompiler::new()?);

            let physical_devices = instance.enumerate_physical_devices()?;

            let devices: Vec<vk::PhysicalDevice> = match &selected {
                Some(indices) => {
                    let mut seen = HashSet::with_capacity(indices.len());
                    let mut list = Vec::with_capacity(indices.len());
                    for &idx in indices {
                        if idx >= physical_devices.len() {
                            return Err(VKMLError::GpuPool(format!(
                                "Selected GPU index {idx} out of range"
                            )));
                        }
                        if !seen.insert(idx) {
                            return Err(VKMLError::GpuPool(format!(
                                "Duplicate GPU index {idx} in selection"
                            )));
                        }
                        list.push(physical_devices[idx]);
                    }
                    list
                }
                None => physical_devices,
            };

            let count = devices.len();
            let mut gpus = Vec::with_capacity(count);

            let tasks: Vec<GpuInitParams> = devices
                .iter()
                .enumerate()
                .map(|(i, &physical_device)| GpuInitParams {
                    instance: instance.clone(),
                    physical_device,
                    slang: slang.clone(),
                    index: i,
                    out_ptr: gpus.as_mut_ptr(),
                })
                .collect();

            global_pool().run(gpu_init_task, &tasks);
            gpus.set_len(count);

            if selected.is_none() {
                // Sort GPUs: discrete GPUs first, then by total memory (descending)
                gpus.sort_by_key(|gpu| {
                    (
                        gpu.device_type() != vk::PhysicalDeviceType::DISCRETE_GPU,
                        std::cmp::Reverse(gpu.memory_total()),
                    )
                });
            }

            Ok(Self {
                gpus,
                _entry: entry,
            })
        }
    }

    pub fn gpus(&self) -> &[Arc<Gpu>] {
        &self.gpus
    }
}

impl std::fmt::Debug for GpuPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuPool").field("gpus", &self.gpus).finish()
    }
}

struct GpuInitParams {
    instance: Arc<Instance>,
    physical_device: vk::PhysicalDevice,
    slang: Arc<SlangCompiler>,
    index: usize,
    out_ptr: *mut Arc<Gpu>,
}

fn gpu_init_task(params: &GpuInitParams) {
    let gpu = Arc::new(
        Gpu::new(
            params.instance.clone(),
            params.physical_device,
            params.slang.clone(),
        )
        .expect("Failed to initialise GPU"),
    );

    unsafe {
        let slot = params.out_ptr.add(params.index);
        ptr::write(slot, gpu);
    }
}
