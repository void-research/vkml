mod f32_f32_f32_cpu;
mod push_constants;

use crate::VKMLError;
use crate::instruction::matmul::f32_f32_f32_cpu::f32_f32_f32_cpu;
use crate::instruction::matmul::push_constants::{
    MatMul1D2DPushConstants, MatMul1D3DPushConstants, MatMul2D1DPushConstants,
    MatMul2D2DPushConstants, MatMul3D1DPushConstants, MatMulTiledPushConstants,
};
use crate::utils::bytes::as_bytes;
use crate::{
    ComputeManager,
    gpu::Gpu,
    instruction::{GpuShader, Instruction},
    tensor::{ComputeTarget, Tensor},
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

pub struct MatMulInstruction {
    pub src1: TensorId,
    pub src2: TensorId,
    pub dst: TensorId,
}

impl MatMulInstruction {
    pub fn select_shader(&self, gpu: &Gpu, cm: &ComputeManager) -> Option<GpuShader> {
        let src1_desc = cm.tensor_desc(self.src1);
        let src2_desc = cm.tensor_desc(self.src2);
        let dst_desc = cm.tensor_desc(self.dst);

        let src1_dtype = src1_desc.data_type();
        let src2_dtype = src2_desc.data_type();
        let dst_dtype = dst_desc.data_type();

        if src1_dtype != src2_dtype || src2_dtype != dst_dtype {
            return None;
        }

        let a_rank = src1_desc.dims().len();
        let b_rank = src2_desc.dims().len();

        let mut op = match (a_rank, b_rank) {
            (1, 2) => GpuShader::MatMul_1D2D,
            (2, 1) => GpuShader::MatMul_2D1D,
            (2, 2) => GpuShader::MatMul_2D2D,
            (2, 3) | (3, 2) | (3, 3) => GpuShader::MatMul_Tiled,
            (3, 1) => GpuShader::MatMul_3D1D,
            (1, 3) => GpuShader::MatMul_1D3D,
            _ => return None,
        };

        if op == GpuShader::MatMul_2D2D {
            let m = src1_desc.dims()[0] as u64;
            let n = src2_desc.dims()[1] as u64;
            if m == 1 {
                op = GpuShader::MatMul_1D2D;
            } else if n == 1 {
                op = GpuShader::MatMul_2D1D;
            } else if gpu.max_shared_memory_size() >= 512 && m >= 8 && n >= 8 {
                op = GpuShader::MatMul_Tiled;
            }
        }

        if op.info().can_run_on(gpu, dst_dtype) {
            Some(op)
        } else {
            None
        }
    }
}

impl Debug for MatMulInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "MatMul(src1={}, src2={}, dst={})",
            self.src1, self.src2, self.dst
        )
    }
}

impl Instruction for MatMulInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.src1, self.src2]
    }

    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst]
    }

    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]) {
        if !new_inputs.is_empty() {
            self.src1 = new_inputs[0];
        }

        if new_inputs.len() > 1 {
            self.src2 = new_inputs[1];
        }

        if !new_outputs.is_empty() {
            self.dst = new_outputs[0];
        }
    }

    fn can_run_on(&self, target: &ComputeTarget, cm: &ComputeManager) -> Result<bool, VKMLError> {
        let src1_desc = cm.tensor_desc(self.src1);
        let src2_desc = cm.tensor_desc(self.src2);
        let dst_desc = cm.tensor_desc(self.dst);

        let a_rank = src1_desc.dims().len();
        let b_rank = src2_desc.dims().len();

        if a_rank == 0 || b_rank == 0 {
            return Err(VKMLError::Instruction(format!(
                "MatMul instruction {:?}: 0-rank tensors unsupported (a_rank={}, b_rank={})",
                self, a_rank, b_rank
            )));
        }

        if a_rank == 1 && b_rank == 1 {
            return Err(VKMLError::Instruction(format!(
                "MatMul instruction {:?}: 1D x 1D dot product unsupported (a_rank=1, b_rank=1)",
                self
            )));
        }

        match target {
            ComputeTarget::Gpu(gpu) => Ok(self.select_shader(gpu, cm).is_some()),
            ComputeTarget::Cpu => {
                let compatible = src1_desc.data_type() == DataType::Float
                    && src2_desc.data_type() == DataType::Float
                    && dst_desc.data_type() == DataType::Float;
                Ok(compatible)
            }
        }
    }

    fn record_into_command_buffer(
        &self,
        gpu: &Gpu,
        command_buffer: vk::CommandBuffer,
        cm: &ComputeManager,
    ) -> Result<(), VKMLError> {
        let op_name = self.select_shader(gpu, cm).ok_or_else(|| {
            VKMLError::Instruction(format!(
                "GPU MatMul has no compatible shader for instruction {:?}",
                self
            ))
        })?;

        let src1_tensor = cm.tensor_read(self.src1);
        let src2_tensor = cm.tensor_read(self.src2);
        let dst_tensor = cm.tensor_read(self.dst);

        execute_gpu_matmul(
            gpu,
            command_buffer,
            src1_tensor,
            src2_tensor,
            dst_tensor,
            op_name,
        )
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        let src1_tensor = cm.tensor_read(self.src1);
        let src2_tensor = cm.tensor_read(self.src2);
        let dst_tensor = cm.tensor_write(self.dst);

        let src1_dtype = src1_tensor.desc().data_type();
        let src2_dtype = src2_tensor.desc().data_type();
        let dst_dtype = dst_tensor.desc().data_type();

        let src1_dims = src1_tensor.desc().dims_usize();
        let src2_dims = src2_tensor.desc().dims_usize();
        let dst_dims = dst_tensor.desc().dims_usize();

        let src1_bytes = src1_tensor.get_cpu_memory_slice_or_panic();
        let src2_bytes = src2_tensor.get_cpu_memory_slice_or_panic();
        let dst_bytes = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        match (src1_dtype, src2_dtype, dst_dtype) {
            (DataType::Float, DataType::Float, DataType::Float) => {
                f32_f32_f32_cpu(
                    src1_dims, src2_dims, dst_dims, src1_bytes, src2_bytes, dst_bytes,
                );
            }
            _ => unimplemented!(
                "CPU MatMul: unimplemented for DataType src1:{:?}, src2:{:?}, dst:{:?}",
                src1_dtype,
                src2_dtype,
                dst_dtype
            ),
        }
    }
}

/// Execute GPU MatMul operation using specialised shaders
fn execute_gpu_matmul(
    gpu: &Gpu,
    command_buffer: vk::CommandBuffer,
    src1_tensor: &Tensor,
    src2_tensor: &Tensor,
    dst_tensor: &Tensor,
    operation: GpuShader,
) -> Result<(), VKMLError> {
    let src1_mem = src1_tensor.get_gpu_memory_or_panic();
    let src2_mem = src2_tensor.get_gpu_memory_or_panic();
    let dst_mem = dst_tensor.get_gpu_memory_or_panic();

    let src1_dims = src1_tensor.desc().dims();
    let src2_dims = src2_tensor.desc().dims();
    let src1_strides = src1_tensor.desc().strides();
    let src2_strides = src2_tensor.desc().strides();
    let dst_strides = dst_tensor.desc().strides();

    let dst_dtype = dst_tensor.desc().data_type();

    // Configure based on operation type
    // Pass actual output dimensions to dispatch
    let (local_size, push_constants_bytes, work_size) = match operation {
        GpuShader::MatMul_1D2D => {
            // [1, k] or [k] × [k, n] → [1, n] or [n]
            let k = *src1_dims.last().unwrap();
            let n = src2_dims[1];

            let pc = MatMul1D2DPushConstants {
                k: k as u32,
                n: n as u32,
                stride_a: *src1_strides.last().unwrap() as u32,
                stride_b0: src2_strides[0] as u32,
                stride_b1: src2_strides[1] as u32,
                stride_c: *dst_strides.last().unwrap() as u32,
            };

            let sg = gpu.subgroup_size().min(gpu.max_workgroup_size()[0]);
            if (n as u32) >= sg {
                let dim_x = sg.min(64);
                let max_inv = gpu.max_workgroup_invocations().min(256);
                let dim_y = (max_inv / dim_x).max(1);
                (
                    [dim_x, dim_y, 1],
                    as_bytes(&pc).to_vec(),
                    [n as u32, dim_y, 1],
                )
            } else {
                let max_threads = gpu
                    .max_workgroup_invocations()
                    .min(gpu.max_workgroup_size()[1]);
                let reduce_threads = 1u32 << (31 - max_threads.leading_zeros());
                (
                    [1, reduce_threads, 1],
                    as_bytes(&pc).to_vec(),
                    [n as u32, reduce_threads, 1],
                )
            }
        }

        GpuShader::MatMul_2D1D => {
            // [m, k] × [k, 1] or [k] → [m, 1] or [m]
            let m = src1_dims[0];
            let k = src1_dims[1];

            let pc = MatMul2D1DPushConstants {
                m: m as u32,
                k: k as u32,
                stride_a0: src1_strides[0] as u32,
                stride_a1: src1_strides[1] as u32,
                stride_b: *src2_strides.last().unwrap() as u32,
                stride_c: *dst_strides.last().unwrap() as u32,
            };

            let sg = gpu.subgroup_size().min(gpu.max_workgroup_size()[0]);
            if (m as u32) >= sg {
                let dim_x = sg.min(64);
                let max_inv = gpu.max_workgroup_invocations().min(256);
                let dim_y = (max_inv / dim_x).max(1);
                (
                    [dim_x, dim_y, 1],
                    as_bytes(&pc).to_vec(),
                    [m as u32, dim_y, 1],
                )
            } else {
                let max_threads = gpu
                    .max_workgroup_invocations()
                    .min(gpu.max_workgroup_size()[1]);
                let reduce_threads = 1u32 << (31 - max_threads.leading_zeros());
                (
                    [1, reduce_threads, 1],
                    as_bytes(&pc).to_vec(),
                    [m as u32, reduce_threads, 1],
                )
            }
        }

        GpuShader::MatMul_2D2D => {
            // [m,k] × [k,n] → [m,n]
            let m = src1_dims[0];
            let k = src1_dims[1];
            let n = src2_dims[1];

            let pc = MatMul2D2DPushConstants {
                m: m as u32,
                k: k as u32,
                n: n as u32,
                stride_a0: src1_strides[0] as u32,
                stride_a1: src1_strides[1] as u32,
                stride_b0: src2_strides[0] as u32,
                stride_b1: src2_strides[1] as u32,
                stride_c0: dst_strides[0] as u32,
                stride_c1: dst_strides[1] as u32,
            };

            (
                gpu.workgroup_size_2d(),
                as_bytes(&pc).to_vec(),
                [n as u32, m as u32, 1],
            )
        }

        GpuShader::MatMul_Tiled => {
            let a_rank = src1_dims.len();
            let b_rank = src2_dims.len();

            let (
                batch,
                m,
                k,
                n,
                stride_a_batch,
                stride_a0,
                stride_a1,
                stride_b_batch,
                stride_b0,
                stride_b1,
                stride_c_batch,
                stride_c0,
                stride_c1,
            ) = match (a_rank, b_rank) {
                (2, 2) => {
                    let m = src1_dims[0] as u32;
                    let k = src1_dims[1] as u32;
                    let n = src2_dims[1] as u32;
                    (
                        1,
                        m,
                        k,
                        n,
                        0,
                        src1_strides[0] as u32,
                        src1_strides[1] as u32,
                        0,
                        src2_strides[0] as u32,
                        src2_strides[1] as u32,
                        0,
                        dst_strides[0] as u32,
                        dst_strides[1] as u32,
                    )
                }
                (2, 3) => {
                    let m = src1_dims[0] as u32;
                    let k = src1_dims[1] as u32;
                    let batch = src2_dims[0] as u32;
                    let n = src2_dims[2] as u32;
                    (
                        batch,
                        m,
                        k,
                        n,
                        0,
                        src1_strides[0] as u32,
                        src1_strides[1] as u32,
                        src2_strides[0] as u32,
                        src2_strides[1] as u32,
                        src2_strides[2] as u32,
                        dst_strides[0] as u32,
                        dst_strides[1] as u32,
                        dst_strides[2] as u32,
                    )
                }
                (3, 2) => {
                    let batch = src1_dims[0] as u32;
                    let m = src1_dims[1] as u32;
                    let k = src1_dims[2] as u32;
                    let n = src2_dims[1] as u32;
                    (
                        batch,
                        m,
                        k,
                        n,
                        src1_strides[0] as u32,
                        src1_strides[1] as u32,
                        src1_strides[2] as u32,
                        0,
                        src2_strides[0] as u32,
                        src2_strides[1] as u32,
                        dst_strides[0] as u32,
                        dst_strides[1] as u32,
                        dst_strides[2] as u32,
                    )
                }
                (3, 3) => {
                    let batch_a = src1_dims[0] as u32;
                    let batch_b = src2_dims[0] as u32;
                    let batch = batch_a.max(batch_b);
                    let m = src1_dims[1] as u32;
                    let k = src1_dims[2] as u32;
                    let n = src2_dims[2] as u32;
                    let stride_a_batch = if batch_a == 1 {
                        0
                    } else {
                        src1_strides[0] as u32
                    };
                    let stride_b_batch = if batch_b == 1 {
                        0
                    } else {
                        src2_strides[0] as u32
                    };
                    (
                        batch,
                        m,
                        k,
                        n,
                        stride_a_batch,
                        src1_strides[1] as u32,
                        src1_strides[2] as u32,
                        stride_b_batch,
                        src2_strides[1] as u32,
                        src2_strides[2] as u32,
                        dst_strides[0] as u32,
                        dst_strides[1] as u32,
                        dst_strides[2] as u32,
                    )
                }
                _ => {
                    return Err(VKMLError::Instruction(format!(
                        "Unsupported MatMul_Tiled dimensions: a_rank={}, b_rank={}",
                        a_rank, b_rank
                    )));
                }
            };

            let pc = MatMulTiledPushConstants {
                batch,
                m,
                k,
                n,
                stride_a_batch,
                stride_a0,
                stride_a1,
                stride_b_batch,
                stride_b0,
                stride_b1,
                stride_c_batch,
                stride_c0,
                stride_c1,
            };

            let bytes_per_thread = 2 * dst_dtype.size_in_bytes().unwrap_or(4);
            let tile_dim = gpu.optimal_tiled_matrix_size(m, n, bytes_per_thread);

            (
                [tile_dim, tile_dim, 1],
                as_bytes(&pc).to_vec(),
                [n, m, batch],
            )
        }

        GpuShader::MatMul_3D1D => {
            // [batch,m,k] × [k] → [batch,m]
            let batch = src1_dims[0];
            let m = src1_dims[1];
            let k = src1_dims[2];
            let total = (batch * m) as u32;

            let pc = MatMul3D1DPushConstants {
                batch: batch as u32,
                m: m as u32,
                k: k as u32,
                total,
                stride_a0: src1_strides[0] as u32,
                stride_a1: src1_strides[1] as u32,
                stride_a2: src1_strides[2] as u32,
                stride_b: src2_strides[0] as u32,
                stride_c0: dst_strides[0] as u32,
                stride_c1: dst_strides[1] as u32,
            };

            let local_size = gpu.workgroup_size_1d();
            (local_size, as_bytes(&pc).to_vec(), [total, 1, 1])
        }

        GpuShader::MatMul_1D3D => {
            // [k] × [batch,k,n] → [batch,n]
            let k = src1_dims[0];
            let batch = src2_dims[0];
            let n = src2_dims[2];
            let total = (batch * n) as u32;

            let pc = MatMul1D3DPushConstants {
                batch: batch as u32,
                k: k as u32,
                n: n as u32,
                total,
                stride_a: src1_strides[0] as u32,
                stride_b0: src2_strides[0] as u32,
                stride_b1: src2_strides[1] as u32,
                stride_b2: src2_strides[2] as u32,
                stride_c0: dst_strides[0] as u32,
                stride_c1: dst_strides[1] as u32,
            };

            let local_size = gpu.workgroup_size_1d();
            (local_size, as_bytes(&pc).to_vec(), [total, 1, 1])
        }

        _ => {
            return Err(VKMLError::Instruction(format!(
                "Unsupported MatMul operation: {:?}",
                operation
            )));
        }
    };

    gpu.bind_slang_compute_pipeline(command_buffer, operation, dst_dtype, local_size);
    gpu.bind_storage_buffers(command_buffer, &[src1_mem, src2_mem, dst_mem]);
    gpu.bind_push_constants(command_buffer, operation, &push_constants_bytes);
    gpu.dispatch(command_buffer, local_size, work_size);

    Ok(())
}
