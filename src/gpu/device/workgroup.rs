use vulkanalia::vk::{self, DeviceV1_0};

use crate::gpu::Gpu;

impl Gpu {
    pub fn workgroup_size_1d(&self) -> [u32; 3] {
        let sg = self
            .subgroup_size()
            .min(self.max_workgroup_invocations())
            .min(self.max_workgroup_size()[0])
            .max(1);
        [sg, 1, 1]
    }

    pub fn workgroup_size_2d(&self) -> [u32; 3] {
        let max_inv = self.max_workgroup_invocations();
        let max_x = self.max_workgroup_size()[0];
        let max_y = self.max_workgroup_size()[1];

        let mut dim_x = 8.min(max_x);
        let mut dim_y = 8.min(max_y);
        while dim_x * dim_y > max_inv {
            if dim_x >= dim_y {
                dim_x = (dim_x / 2).max(1);
            } else {
                dim_y = (dim_y / 2).max(1);
            }
        }
        [dim_x, dim_y, 1]
    }

    pub fn workgroup_size_3d(&self) -> [u32; 3] {
        let max_inv = self.max_workgroup_invocations();
        let max_x = self.max_workgroup_size()[0];
        let max_y = self.max_workgroup_size()[1];
        let max_z = self.max_workgroup_size()[2];

        let mut dim_x = 4.min(max_x);
        let mut dim_y = 4.min(max_y);
        let mut dim_z = 4.min(max_z);
        while dim_x * dim_y * dim_z > max_inv {
            if dim_x >= dim_y && dim_x >= dim_z {
                dim_x = (dim_x / 2).max(1);
            } else if dim_y >= dim_z {
                dim_y = (dim_y / 2).max(1);
            } else {
                dim_z = (dim_z / 2).max(1);
            }
        }
        [dim_x, dim_y, dim_z]
    }

    pub fn optimal_tiled_matrix_size(&self, m: u32, n: u32, bytes_per_thread: usize) -> u32 {
        let max_shmem = self.max_shared_memory_size() as u64;
        let max_inv = self.max_workgroup_invocations();
        let max_x = self.max_workgroup_size()[0];
        let max_y = self.max_workgroup_size()[1];

        for tile in [32, 16, 8] {
            let threads = tile * tile;
            let shmem_needed = (threads as u64) * (bytes_per_thread as u64);

            if m >= tile
                && n >= tile
                && threads <= max_inv
                && tile <= max_x
                && tile <= max_y
                && shmem_needed <= max_shmem
            {
                return tile;
            }
        }

        8
    }

    pub fn dispatch(&self, cb: vk::CommandBuffer, local_size: [u32; 3], work_size: [u32; 3]) {
        let dispatch_x = work_size[0].div_ceil(local_size[0]);
        let dispatch_y = work_size[1].div_ceil(local_size[1]);
        let dispatch_z = work_size[2].div_ceil(local_size[2]);
        unsafe {
            self.device
                .cmd_dispatch(cb, dispatch_x, dispatch_y, dispatch_z);
        }
    }
}
