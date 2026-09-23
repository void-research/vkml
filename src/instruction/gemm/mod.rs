mod f32_f32_f32_f32_cpu;
mod push_constants;

use crate::VKMLError;
use crate::instruction::gemm::f32_f32_f32_f32_cpu::f32_f32_f32_f32_cpu;
use crate::instruction::gemm::push_constants::GemmPushConstants;
use crate::utils::bytes::as_bytes;
use crate::{
    ComputeManager,
    gpu::Gpu,
    instruction::{GpuShader, Instruction},
    tensor::ComputeTarget,
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

/// GEMM (General Matrix Multiplication) instruction
/// Computes Y = alpha * op(A) * op(B) + beta * C
/// where op(X) is either X or X^T depending on transpose flags
pub struct GemmInstruction {
    pub a: TensorId,
    pub b: TensorId,
    pub c: Option<TensorId>,
    pub y: TensorId,
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: bool,
    pub trans_b: bool,
}

impl GemmInstruction {
    pub fn select_shader(&self, gpu: &Gpu, cm: &ComputeManager) -> Option<GpuShader> {
        let a_desc = cm.tensor_desc(self.a);
        let b_desc = cm.tensor_desc(self.b);
        let y_desc = cm.tensor_desc(self.y);
        let c_desc = self.c.map(|c| cm.tensor_desc(c));

        let y_dtype = y_desc.data_type();
        if a_desc.data_type() != y_dtype || b_desc.data_type() != y_dtype {
            return None;
        }
        if let Some(c) = c_desc
            && c.data_type() != y_dtype
        {
            return None;
        }

        let (m, _, n) = compute_gemm_dimensions(
            a_desc.dims(),
            b_desc.dims(),
            y_desc.dims(),
            self.trans_a,
            self.trans_b,
        )
        .ok()?;

        let shader = if gpu.max_shared_memory_size() >= 512 && (m as u64) >= 8 && (n as u64) >= 8 {
            GpuShader::Gemm_Tiled
        } else {
            GpuShader::Gemm
        };

        if shader.info().can_run_on(gpu, y_dtype) {
            Some(shader)
        } else {
            None
        }
    }
}

impl Debug for GemmInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Gemm(a={}, b={}, c={:?}, y={}, alpha={}, beta={}, trans_a={}, trans_b={})",
            self.a, self.b, self.c, self.y, self.alpha, self.beta, self.trans_a, self.trans_b
        )
    }
}

impl Instruction for GemmInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        let mut inputs = vec![self.a, self.b];
        if let Some(c) = self.c {
            inputs.push(c);
        }
        inputs
    }

    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.y]
    }

    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]) {
        if new_inputs.len() >= 2 {
            self.a = new_inputs[0];
            self.b = new_inputs[1];
            if new_inputs.len() >= 3 {
                self.c = Some(new_inputs[2]);
            }
        }

        if !new_outputs.is_empty() {
            self.y = new_outputs[0];
        }
    }

    fn can_run_on(&self, target: &ComputeTarget, cm: &ComputeManager) -> Result<bool, VKMLError> {
        let a_desc = cm.tensor_desc(self.a);
        let b_desc = cm.tensor_desc(self.b);
        let y_desc = cm.tensor_desc(self.y);

        compute_gemm_dimensions(
            a_desc.dims(),
            b_desc.dims(),
            y_desc.dims(),
            self.trans_a,
            self.trans_b,
        )?;

        match target {
            ComputeTarget::Gpu(gpu) => Ok(self.select_shader(gpu, cm).is_some()),
            ComputeTarget::Cpu => {
                let c_ok = match self.c {
                    Some(c) => cm.tensor_desc(c).data_type() == DataType::Float,
                    None => true,
                };
                let compatible = a_desc.data_type() == DataType::Float
                    && b_desc.data_type() == DataType::Float
                    && y_desc.data_type() == DataType::Float
                    && c_ok;
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
                "GPU Gemm has no compatible shader for instruction {:?}",
                self
            ))
        })?;

        let a_tensor = cm.tensor_read(self.a);
        let b_tensor = cm.tensor_read(self.b);
        let y_tensor = cm.tensor_read(self.y);
        let c_tensor = self.c.map(|c| cm.tensor_read(c));

        let a_gpu_mem = a_tensor.get_gpu_memory_or_panic();
        let b_gpu_mem = b_tensor.get_gpu_memory_or_panic();
        let y_gpu_mem = y_tensor.get_gpu_memory_or_panic();

        let a_dims = a_tensor.desc().dims();
        let b_dims = b_tensor.desc().dims();
        let y_dims = y_tensor.desc().dims();

        // Determine matrix dimensions based on transpose flags
        // A is (M, K) or (K, M) if transposed
        // B is (K, N) or (N, K) if transposed
        // Y is (M, N)
        let (m, k, n) =
            compute_gemm_dimensions(a_dims, b_dims, y_dims, self.trans_a, self.trans_b)?;

        let a_strides = a_tensor.desc().strides();
        let b_strides = b_tensor.desc().strides();
        let y_strides = y_tensor.desc().strides();

        // Build push constants
        let has_c = c_tensor.is_some();
        let pc = GemmPushConstants {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            stride_a0: a_strides[0] as u32,
            stride_a1: a_strides[1] as u32,
            stride_b0: b_strides[0] as u32,
            stride_b1: b_strides[1] as u32,
            stride_y0: y_strides[0] as u32,
            stride_y1: y_strides[1] as u32,
            trans_a: if self.trans_a { 1u32 } else { 0u32 },
            trans_b: if self.trans_b { 1u32 } else { 0u32 },
            alpha: self.alpha.to_bits(),
            beta: self.beta.to_bits(),
            has_c: if has_c { 1u32 } else { 0u32 },
        };

        // Prepare storage buffers with optional C
        let c_gpu_mem = c_tensor.as_ref().map(|t| t.get_gpu_memory_or_panic());

        // Choose optimal workgroup size for 2D matrix operation
        let local_size = gpu.workgroup_size_2d();

        let y_dtype = y_tensor.desc().data_type();

        match op_name {
            GpuShader::Gemm => {
                gpu.bind_slang_compute_pipeline(command_buffer, op_name, y_dtype, local_size);
                gpu.bind_storage_buffers_optional(
                    command_buffer,
                    &[Some(a_gpu_mem), Some(b_gpu_mem), c_gpu_mem, Some(y_gpu_mem)],
                );
                gpu.bind_push_constants(command_buffer, op_name, as_bytes(&pc));
                gpu.dispatch(command_buffer, local_size, [n as u32, m as u32, 1]);
            }
            GpuShader::Gemm_Tiled => {
                let bytes_per_thread = 2 * y_dtype.size_in_bytes().unwrap_or(4);
                let tile_dim = gpu.optimal_tiled_matrix_size(m as u32, n as u32, bytes_per_thread);
                let tiled_local_size = [tile_dim, tile_dim, 1];

                gpu.bind_slang_compute_pipeline(command_buffer, op_name, y_dtype, tiled_local_size);
                gpu.bind_storage_buffers_optional(
                    command_buffer,
                    &[Some(a_gpu_mem), Some(b_gpu_mem), c_gpu_mem, Some(y_gpu_mem)],
                );
                gpu.bind_push_constants(command_buffer, GpuShader::Gemm, as_bytes(&pc));
                gpu.dispatch(command_buffer, tiled_local_size, [n as u32, m as u32, 1]);
            }
            _ => {
                return Err(VKMLError::Instruction(format!(
                    "Invalid GpuShader {:?} for Gemm",
                    op_name
                )));
            }
        }

        Ok(())
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        let a_tensor = cm.tensor_read(self.a);
        let b_tensor = cm.tensor_read(self.b);
        let c_tensor = self.c.map(|c| cm.tensor_read(c));
        let y_tensor = cm.tensor_write(self.y);

        let a_dims_i64 = a_tensor.desc().dims();
        let b_dims_i64 = b_tensor.desc().dims();
        let y_dims_i64 = y_tensor.desc().dims();

        let a_dims: Vec<usize> = a_dims_i64.iter().map(|&d| d as usize).collect();
        let b_dims: Vec<usize> = b_dims_i64.iter().map(|&d| d as usize).collect();
        let y_dims: Vec<usize> = y_dims_i64.iter().map(|&d| d as usize).collect();

        let a_dtype = a_tensor.desc().data_type();
        let b_dtype = b_tensor.desc().data_type();
        let c_dtype_opt = c_tensor.as_ref().map(|t| t.desc().data_type());
        let y_dtype = y_tensor.desc().data_type();

        let a_bytes = a_tensor.get_cpu_memory_slice_or_panic();
        let b_bytes = b_tensor.get_cpu_memory_slice_or_panic();
        let c_bytes = c_tensor.map(|t| t.get_cpu_memory_slice_or_panic());
        let y_bytes = y_tensor.get_cpu_memory_mut_slice_or_panic();

        match (a_dtype, b_dtype, c_dtype_opt, y_dtype) {
            (DataType::Float, DataType::Float, None, DataType::Float)
            | (DataType::Float, DataType::Float, Some(DataType::Float), DataType::Float) => {
                f32_f32_f32_f32_cpu(
                    a_dims,
                    b_dims,
                    y_dims,
                    a_bytes,
                    b_bytes,
                    c_bytes,
                    y_bytes,
                    self.alpha,
                    self.beta,
                    self.trans_a,
                    self.trans_b,
                );
            }
            _ => unimplemented!(
                "Gemm: unimplemented for DataType a:{:?}, b:{:?}, c:{}, y:{:?}",
                a_dtype,
                b_dtype,
                c_dtype_opt
                    .map(|dt| format!("{:?}", dt))
                    .unwrap_or_else(|| "None".to_string()),
                y_dtype
            ),
        }
    }
}

fn compute_gemm_dimensions(
    a_dims: &[i64],
    b_dims: &[i64],
    y_dims: &[i64],
    trans_a: bool,
    trans_b: bool,
) -> Result<(usize, usize, usize), VKMLError> {
    if a_dims.len() != 2 || b_dims.len() != 2 || y_dims.len() != 2 {
        return Err(VKMLError::Instruction(format!(
            "GEMM requires 2D tensors, got A: {:?}, B: {:?}, Y: {:?}",
            a_dims, b_dims, y_dims
        )));
    }

    // A is (M, K) or (K, M) if trans_a
    let (a_dim0, a_dim1) = (a_dims[0] as usize, a_dims[1] as usize);
    let (m, k_a) = if trans_a {
        (a_dim1, a_dim0)
    } else {
        (a_dim0, a_dim1)
    };

    // B is (K, N) or (N, K) if trans_b
    let (b_dim0, b_dim1) = (b_dims[0] as usize, b_dims[1] as usize);
    let (k_b, n) = if trans_b {
        (b_dim1, b_dim0)
    } else {
        (b_dim0, b_dim1)
    };

    // Verify K dimension matches
    if k_a != k_b {
        return Err(VKMLError::Instruction(format!(
            "GEMM: K dimension mismatch: A gives K={}, B gives K={}",
            k_a, k_b
        )));
    }

    // Verify output dimensions
    if y_dims[0] as usize != m || y_dims[1] as usize != n {
        return Err(VKMLError::Instruction(format!(
            "GEMM: output shape mismatch: expected ({}, {}), got ({}, {})",
            m, n, y_dims[0], y_dims[1]
        )));
    }

    Ok((m, k_a, n))
}
