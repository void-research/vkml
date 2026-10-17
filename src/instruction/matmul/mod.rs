mod f32_f32_f32_cpu;

use crate::VKMLError;
use crate::gpu::shader::{Shader, shader};
use crate::instruction::matmul::f32_f32_f32_cpu::f32_f32_f32_cpu;
use crate::{
    ComputeManager,
    instruction::{Dispatch, Instruction, PushConstants, VkOperation},
    tensor::ComputeTarget,
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};

pub static MATMUL_1D2D: Shader = shader!("matmul_1d2d");
pub static MATMUL_1D2D_ATOMIC: Shader = shader!("matmul_1d2d_atomic", [Float, Float16]);
pub static MATMUL_2D1D: Shader = shader!("matmul_2d1d");
pub static MATMUL_2D1D_ATOMIC: Shader = shader!("matmul_2d1d_atomic", [Float, Float16]);
pub static MATMUL_2D2D: Shader = shader!("matmul_2d2d");
pub static MATMUL_3D1D: Shader = shader!("matmul_3d1d");
pub static MATMUL_1D3D: Shader = shader!("matmul_1d3d");
pub static MATMUL_TILED: Shader = shader!("matmul_tiled");

#[derive(PartialEq)]
pub enum MatMulVariant {
    D1D2,
    D2D1,
    D2D2,
    D3D1,
    D1D3,
    Tiled,
}

impl MatMulVariant {
    pub fn shader(&self) -> &'static Shader {
        match self {
            Self::D1D2 => &MATMUL_1D2D,
            Self::D2D1 => &MATMUL_2D1D,
            Self::D2D2 => &MATMUL_2D2D,
            Self::D3D1 => &MATMUL_3D1D,
            Self::D1D3 => &MATMUL_1D3D,
            Self::Tiled => &MATMUL_TILED,
        }
    }
}

pub struct MatMulInstruction {
    pub src1: TensorId,
    pub src2: TensorId,
    pub dst: TensorId,
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

    fn select_operation(
        &self,
        target: &ComputeTarget,
        cm: &ComputeManager,
    ) -> Result<Option<Dispatch>, VKMLError> {
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
            ComputeTarget::Cpu => {
                let compatible = src1_desc.data_type() == DataType::Float
                    && src2_desc.data_type() == DataType::Float
                    && dst_desc.data_type() == DataType::Float;
                if compatible {
                    Ok(Some(Dispatch::Cpu))
                } else {
                    Ok(None)
                }
            }
            ComputeTarget::Gpu(gpu) => {
                let src1_dtype = src1_desc.data_type();
                let src2_dtype = src2_desc.data_type();
                let dst_dtype = dst_desc.data_type();

                if src1_dtype != src2_dtype || src2_dtype != dst_dtype {
                    return Ok(None);
                }

                let mut op = match (a_rank, b_rank) {
                    (1, 2) => MatMulVariant::D1D2,
                    (2, 1) => MatMulVariant::D2D1,
                    (2, 2) => MatMulVariant::D2D2,
                    (2, 3) | (3, 2) | (3, 3) => MatMulVariant::Tiled,
                    (3, 1) => MatMulVariant::D3D1,
                    (1, 3) => MatMulVariant::D1D3,
                    _ => return Ok(None),
                };

                if op == MatMulVariant::D2D2 {
                    let m = src1_desc.dims()[0] as u64;
                    let n = src2_desc.dims()[1] as u64;
                    if m == 1 {
                        op = MatMulVariant::D1D2;
                    } else if n == 1 {
                        op = MatMulVariant::D2D1;
                    } else if gpu.max_shared_memory_size() >= 512 && m >= 8 && n >= 8 {
                        op = MatMulVariant::Tiled;
                    }
                }

                let Some(spv) = gpu.get_spirv(op.shader(), dst_dtype) else {
                    return Ok(None);
                };

                let src1_dims = src1_desc.dims();
                let src2_dims = src2_desc.dims();
                let src1_strides = src1_desc.strides();
                let src2_strides = src2_desc.strides();
                let dst_strides = dst_desc.strides();
                let dst_dtype = dst_desc.data_type();

                let (local_size, push_constants, work_size) = match op {
                    MatMulVariant::D1D2 => {
                        let k = *src1_dims.last().unwrap();
                        let n = src2_dims[1];

                        let num_slices = 64u32;
                        let max_y = gpu.max_workgroup_size()[1].max(1);
                        let local_y = 1 << (128u32.min(max_y)).ilog2();
                        // For Split-K to amortize workgroup barrier and atomic add overhead,
                        // each thread in the slice must have at least 2 reduction iterations.
                        let min_k = (num_slices * local_y * 2) as i64;

                        if gpu.extensions().supports_atomic_float(dst_dtype)
                            && n <= gpu.subgroup_size() as i64
                            && k >= min_k
                        {
                            let Some(atomic_spv) = gpu.get_spirv(&MATMUL_1D2D_ATOMIC, dst_dtype)
                            else {
                                return Ok(None);
                            };

                            let pc = PushConstants::new()
                                .buffer(self.src1)
                                .buffer(self.src2)
                                .buffer(self.dst)
                                .push(k as u32)
                                .push(n as u32)
                                .push(num_slices)
                                .push(*src1_strides.last().unwrap() as u32)
                                .push(src2_strides[0] as u32)
                                .push(src2_strides[1] as u32)
                                .push(*dst_strides.last().unwrap() as u32);

                            let local = [1, local_y, 1];
                            let work = [n as u32, num_slices * local_y, 1];

                            return Ok(Some(Dispatch::Gpu(VkOperation::ClearAndCompute {
                                clear_dst: self.dst,
                                spv: atomic_spv,
                                local_size: local,
                                work_size: work,
                                push_constants: pc,
                            })));
                        }

                        let pc = PushConstants::new()
                            .buffer(self.src1)
                            .buffer(self.src2)
                            .buffer(self.dst)
                            .push(k as u32)
                            .push(n as u32)
                            .push(*src1_strides.last().unwrap() as u32)
                            .push(src2_strides[0] as u32)
                            .push(src2_strides[1] as u32)
                            .push(*dst_strides.last().unwrap() as u32);

                        let local = gpu.workgroup_size_gemv(n as u32);
                        let work = [n as u32, local[1], 1];
                        (local, pc, work)
                    }
                    MatMulVariant::D2D1 => {
                        let m = src1_dims[0];
                        let k = src1_dims[1];

                        let num_slices = 64u32;
                        let max_y = gpu.max_workgroup_size()[1].max(1);
                        let local_y = 1 << (128u32.min(max_y)).ilog2();
                        // For Split-K to amortize workgroup barrier and atomic add overhead,
                        // each thread in the slice must have at least 2 reduction iterations.
                        let min_k = (num_slices * local_y * 2) as i64;

                        if gpu.extensions().supports_atomic_float(dst_dtype)
                            && m <= gpu.subgroup_size() as i64
                            && k >= min_k
                        {
                            let Some(atomic_spv) = gpu.get_spirv(&MATMUL_2D1D_ATOMIC, dst_dtype)
                            else {
                                return Ok(None);
                            };

                            let pc = PushConstants::new()
                                .buffer(self.src1)
                                .buffer(self.src2)
                                .buffer(self.dst)
                                .push(m as u32)
                                .push(k as u32)
                                .push(num_slices)
                                .push(src1_strides[0] as u32)
                                .push(src1_strides[1] as u32)
                                .push(*src2_strides.last().unwrap() as u32)
                                .push(*dst_strides.last().unwrap() as u32);

                            let local = [1, local_y, 1];
                            let work = [m as u32, num_slices * local_y, 1];

                            return Ok(Some(Dispatch::Gpu(VkOperation::ClearAndCompute {
                                clear_dst: self.dst,
                                spv: atomic_spv,
                                local_size: local,
                                work_size: work,
                                push_constants: pc,
                            })));
                        }

                        let pc = PushConstants::new()
                            .buffer(self.src1)
                            .buffer(self.src2)
                            .buffer(self.dst)
                            .push(m as u32)
                            .push(k as u32)
                            .push(src1_strides[0] as u32)
                            .push(src1_strides[1] as u32)
                            .push(*src2_strides.last().unwrap() as u32)
                            .push(*dst_strides.last().unwrap() as u32);

                        let local = gpu.workgroup_size_gemv(m as u32);
                        let work = [m as u32, local[1], 1];
                        (local, pc, work)
                    }
                    MatMulVariant::D2D2 => {
                        let m = src1_dims[0];
                        let k = src1_dims[1];
                        let n = src2_dims[1];

                        let pc = PushConstants::new()
                            .buffer(self.src1)
                            .buffer(self.src2)
                            .buffer(self.dst)
                            .push(m as u32)
                            .push(k as u32)
                            .push(n as u32)
                            .push(src1_strides[0] as u32)
                            .push(src1_strides[1] as u32)
                            .push(src2_strides[0] as u32)
                            .push(src2_strides[1] as u32)
                            .push(dst_strides[0] as u32)
                            .push(dst_strides[1] as u32);

                        (gpu.workgroup_size_2d(), pc, [n as u32, m as u32, 1])
                    }
                    MatMulVariant::Tiled => {
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

                        let bytes_per_thread = 2 * dst_dtype.size_in_bytes().unwrap_or(4);
                        let tile_dim = gpu.optimal_tiled_matrix_size(m, n, bytes_per_thread);

                        let pc = PushConstants::new()
                            .buffer(self.src1)
                            .buffer(self.src2)
                            .buffer(self.dst)
                            .push(batch)
                            .push(m)
                            .push(k)
                            .push(n)
                            .push(stride_a_batch)
                            .push(stride_a0)
                            .push(stride_a1)
                            .push(stride_b_batch)
                            .push(stride_b0)
                            .push(stride_b1)
                            .push(stride_c_batch)
                            .push(stride_c0)
                            .push(stride_c1);

                        ([tile_dim, tile_dim, 1], pc, [n, m, batch])
                    }
                    MatMulVariant::D3D1 => {
                        let batch = src1_dims[0];
                        let m = src1_dims[1];
                        let k = src1_dims[2];
                        let total = (batch * m) as u32;

                        let pc = PushConstants::new()
                            .buffer(self.src1)
                            .buffer(self.src2)
                            .buffer(self.dst)
                            .push(batch as u32)
                            .push(m as u32)
                            .push(k as u32)
                            .push(total)
                            .push(src1_strides[0] as u32)
                            .push(src1_strides[1] as u32)
                            .push(src1_strides[2] as u32)
                            .push(src2_strides[0] as u32)
                            .push(dst_strides[0] as u32)
                            .push(dst_strides[1] as u32);

                        let local_size = gpu.workgroup_size_1d();
                        (local_size, pc, [total, 1, 1])
                    }
                    MatMulVariant::D1D3 => {
                        let k = src1_dims[0];
                        let batch = src2_dims[0];
                        let n = src2_dims[2];
                        let total = (batch * n) as u32;

                        let pc = PushConstants::new()
                            .buffer(self.src1)
                            .buffer(self.src2)
                            .buffer(self.dst)
                            .push(batch as u32)
                            .push(k as u32)
                            .push(n as u32)
                            .push(total)
                            .push(src1_strides[0] as u32)
                            .push(src2_strides[0] as u32)
                            .push(src2_strides[1] as u32)
                            .push(src2_strides[2] as u32)
                            .push(dst_strides[0] as u32)
                            .push(dst_strides[1] as u32);

                        let local_size = gpu.workgroup_size_1d();
                        (local_size, pc, [total, 1, 1])
                    }
                };

                Ok(Some(Dispatch::Gpu(VkOperation::Compute {
                    spv,
                    local_size,
                    work_size,
                    push_constants,
                })))
            }
        }
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
