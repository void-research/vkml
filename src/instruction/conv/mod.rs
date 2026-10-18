mod f32_f32_f32_f32_cpu;

use crate::ComputeManager;
use crate::VKMLError;
use crate::gpu::shader::{Shader, shader};
use crate::instruction::gemm::{GEMM_SHADER, GEMM_TILED_SHADER};
use crate::utils::{OnnxAutoPad, calc_begin_and_end_pads};
use crate::{
    instruction::{
        Dispatch, Instruction, PushConstants, VkOperation,
        conv::f32_f32_f32_f32_cpu::f32_f32_f32_f32_cpu,
    },
    tensor::{ComputeTarget, TensorDesc},
    tensor_graph::TensorId,
};

use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};

pub static CONV_1D_SHADER: Shader = shader!("conv_1d");
pub static CONV_2D_SHADER: Shader = shader!("conv_2d");
pub static CONV_2D_TILED_SHADER: Shader = shader!("conv_2d_tiled");
pub static CONV_3D_SHADER: Shader = shader!("conv_3d");

pub struct ConvInstruction {
    pub src: TensorId,
    pub weights: TensorId,
    pub bias: Option<TensorId>,
    pub dst: TensorId,

    pub auto_pad: OnnxAutoPad,
    pub dilations: Vec<i64>,
    pub group: i64,
    pub kernel_shape: Vec<i64>,
    pub pads: Vec<i64>,
    pub strides: Vec<i64>,
}

impl ConvInstruction {
    fn compute_pads(&self, src_desc: &TensorDesc) -> Vec<i64> {
        let (pb, _pe) = calc_begin_and_end_pads(
            self.auto_pad.clone(),
            &self.pads,
            &self.kernel_shape,
            &self.strides,
            &self.dilations,
            src_desc,
        );
        pb
    }
}

impl Debug for ConvInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Conv(src={}, weights={}, bias={:?}, dst={}, auto_pad={:?}, dilations={:?}, group={:?}, kernel_shape={:?}, pads={:?}, strides={:?})",
            self.src,
            self.weights,
            self.bias,
            self.dst,
            self.auto_pad,
            self.dilations,
            self.group,
            self.kernel_shape,
            self.pads,
            self.strides
        )
    }
}

impl Instruction for ConvInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        let mut inputs = vec![self.src, self.weights];
        if let Some(bias) = self.bias {
            inputs.push(bias);
        }
        inputs
    }

    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst]
    }

    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]) {
        if !new_inputs.is_empty() {
            self.src = new_inputs[0];
        }

        if new_inputs.len() > 1 {
            self.weights = new_inputs[1];
        }

        if new_inputs.len() > 2 && self.bias.is_some() {
            self.bias = Some(new_inputs[2]);
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
        let src_desc = cm.tensor_desc(self.src);
        let weights_desc = cm.tensor_desc(self.weights);
        let dst_desc = cm.tensor_desc(self.dst);

        let spatial_rank = src_desc.ndim().saturating_sub(2);
        if !(1..=3).contains(&spatial_rank) {
            return Err(VKMLError::Instruction(format!(
                "Conv instruction {:?}: spatial rank must be 1, 2, or 3, got {}",
                self, spatial_rank
            )));
        }

        match target {
            ComputeTarget::Cpu => {
                if src_desc.data_type() != DataType::Float
                    || weights_desc.data_type() != DataType::Float
                    || dst_desc.data_type() != DataType::Float
                {
                    return Ok(None);
                }

                if let Some(bias_id) = self.bias
                    && cm.tensor_desc(bias_id).data_type() != DataType::Float
                {
                    return Ok(None);
                }

                Ok(Some(Dispatch::Cpu))
            }
            ComputeTarget::Gpu(gpu) => {
                let src_dims = src_desc.dims();
                let dst_dims = dst_desc.dims();
                if src_dims.len() < 2 || dst_dims.len() < 2 {
                    return Ok(None);
                }

                let c_val = src_dims[1];
                let m_val = dst_dims[1];
                if self.group < 1 || c_val % self.group != 0 || m_val % self.group != 0 {
                    return Ok(None);
                }

                let dtype = src_desc.data_type();
                if weights_desc.data_type() != dtype || dst_desc.data_type() != dtype {
                    return Ok(None);
                }

                if let Some(bias_id) = self.bias
                    && cm.tensor_desc(bias_id).data_type() != dtype
                {
                    return Ok(None);
                }

                let pb = self.compute_pads(src_desc);

                let (shader, local_size, push_constants, work_size) = match spatial_rank {
                    1 => {
                        let input_len = if src_dims.len() >= 3 {
                            src_dims[2] as u32
                        } else {
                            1
                        };
                        let output_len = if dst_dims.len() >= 3 {
                            dst_dims[2] as u32
                        } else {
                            1
                        };

                        let pc_values = PushConstants::new()
                            .buffer(self.src)
                            .buffer(self.weights)
                            .buffer(self.dst)
                            .buffer(self.bias)
                            .push(src_dims[0] as u32)
                            .push(src_dims[1] as u32)
                            .push(dst_dims[1] as u32)
                            .push(input_len)
                            .push(output_len)
                            .push(self.kernel_shape.first().copied().unwrap_or(1) as u32)
                            .push(self.strides.first().copied().unwrap_or(1) as u32)
                            .push(self.dilations.first().copied().unwrap_or(1) as u32)
                            .push(pb.first().copied().unwrap_or(0) as u32)
                            .push(self.group as u32)
                            .push(if self.bias.is_some() { 1u32 } else { 0u32 });

                        let total = (src_dims[0] as u32) * (dst_dims[1] as u32) * output_len;
                        (
                            &CONV_1D_SHADER,
                            gpu.workgroup_size_1d(),
                            pc_values,
                            [total, 1, 1],
                        )
                    }
                    2 => {
                        let batch = src_dims[0] as u32;
                        let in_c = src_dims[1] as u32;
                        let out_c = dst_dims[1] as u32;
                        let in_h = src_dims[2] as u32;
                        let in_w = src_dims[3] as u32;
                        let out_h = dst_dims[2] as u32;
                        let out_w = dst_dims[3] as u32;

                        let k_h = self.kernel_shape.first().copied().unwrap_or(1) as u32;
                        let k_w = self.kernel_shape.get(1).copied().unwrap_or(1) as u32;
                        let s_h = self.strides.first().copied().unwrap_or(1) as u32;
                        let s_w = self.strides.get(1).copied().unwrap_or(1) as u32;
                        let d_h = self.dilations.first().copied().unwrap_or(1) as u32;
                        let d_w = self.dilations.get(1).copied().unwrap_or(1) as u32;
                        let pad_h = pb.first().copied().unwrap_or(0) as u32;
                        let pad_w = pb.get(1).copied().unwrap_or(0) as u32;

                        let s_total = out_h * out_w;
                        let k_total = in_c * k_h * k_w;

                        let is_1x1_direct_gemm = self.group == 1
                            && batch == 1
                            && k_h == 1
                            && k_w == 1
                            && s_h == 1
                            && s_w == 1
                            && d_h == 1
                            && d_w == 1
                            && pad_h == 0
                            && pad_w == 0;

                        if is_1x1_direct_gemm {
                            let m = out_c;
                            let n = s_total;
                            let has_bias = self.bias.is_some();

                            let pc_values = PushConstants::new()
                                .buffer(self.weights)
                                .buffer(self.src)
                                .buffer(self.bias)
                                .buffer(self.dst)
                                .push(m)
                                .push(k_total)
                                .push(n)
                                .push(in_c)
                                .push(1u32)
                                .push(n)
                                .push(1u32)
                                .push(n)
                                .push(1u32)
                                .push(if has_bias { 1u32 } else { 0u32 })
                                .push(0u32)
                                .push(0u32)
                                .push(0u32)
                                .push(1.0f32)
                                .push(1.0f32)
                                .push(if has_bias { 1u32 } else { 0u32 });

                            let (shader, local_size, work_size) = if m == 1 {
                                (&GEMM_SHADER, [gpu.workgroup_size_1d()[0], 1, 1], [n, 1, 1])
                            } else if n == 1 {
                                let wg_y = gpu
                                    .subgroup_size()
                                    .min(gpu.max_workgroup_invocations())
                                    .min(gpu.max_workgroup_size()[1])
                                    .max(1);
                                (&GEMM_SHADER, [1, wg_y, 1], [1, m, 1])
                            } else if gpu.max_shared_memory_size() >= 512 && m >= 8 && n >= 8 {
                                let bytes_per_thread = 2 * dtype.size_in_bytes().unwrap_or(4);
                                let tile_dim =
                                    gpu.optimal_tiled_matrix_size(m, n, bytes_per_thread);
                                (&GEMM_TILED_SHADER, [tile_dim, tile_dim, 1], [n, m, 1])
                            } else {
                                (&GEMM_SHADER, gpu.workgroup_size_2d(), [n, m, 1])
                            };

                            (shader, local_size, pc_values, work_size)
                        } else if self.group == 1
                            && gpu.max_shared_memory_size() >= 512
                            && out_c >= 8
                            && s_total >= 8
                        {
                            let bytes_per_thread = 2 * dtype.size_in_bytes().unwrap_or(4);
                            let tile_dim =
                                gpu.optimal_tiled_matrix_size(out_c, s_total, bytes_per_thread);

                            let pc_values = PushConstants::new()
                                .buffer(self.src)
                                .buffer(self.weights)
                                .buffer(self.dst)
                                .buffer(self.bias)
                                .push(batch)
                                .push(in_c)
                                .push(out_c)
                                .push(in_h)
                                .push(in_w)
                                .push(out_h)
                                .push(out_w)
                                .push(k_h)
                                .push(k_w)
                                .push(s_h)
                                .push(s_w)
                                .push(d_h)
                                .push(d_w)
                                .push(pad_h)
                                .push(pad_w)
                                .push(k_total)
                                .push(s_total)
                                .push(if self.bias.is_some() { 1u32 } else { 0u32 });

                            (
                                &CONV_2D_TILED_SHADER,
                                [tile_dim, tile_dim, 1],
                                pc_values,
                                [s_total, out_c, batch],
                            )
                        } else {
                            let pc_values = PushConstants::new()
                                .buffer(self.src)
                                .buffer(self.weights)
                                .buffer(self.dst)
                                .buffer(self.bias)
                                .push(batch)
                                .push(in_c)
                                .push(out_c)
                                .push(in_h)
                                .push(in_w)
                                .push(out_h)
                                .push(out_w)
                                .push(k_h)
                                .push(k_w)
                                .push(s_h)
                                .push(s_w)
                                .push(d_h)
                                .push(d_w)
                                .push(pad_h)
                                .push(pad_w)
                                .push(self.group as u32)
                                .push(if self.bias.is_some() { 1u32 } else { 0u32 });

                            let batch_nm = batch * out_c;
                            (
                                &CONV_2D_SHADER,
                                gpu.workgroup_size_2d(),
                                pc_values,
                                [out_w, out_h, batch_nm],
                            )
                        }
                    }
                    3 => {
                        let pc_values = PushConstants::new()
                            .buffer(self.src)
                            .buffer(self.weights)
                            .buffer(self.dst)
                            .buffer(self.bias)
                            .push(src_dims[0] as u32)
                            .push(src_dims[1] as u32)
                            .push(dst_dims[1] as u32)
                            .push(src_dims[2] as u32)
                            .push(src_dims[3] as u32)
                            .push(src_dims[4] as u32)
                            .push(dst_dims[2] as u32)
                            .push(dst_dims[3] as u32)
                            .push(dst_dims[4] as u32)
                            .push(self.kernel_shape.first().copied().unwrap_or(1) as u32)
                            .push(self.kernel_shape.get(1).copied().unwrap_or(1) as u32)
                            .push(self.kernel_shape.get(2).copied().unwrap_or(1) as u32)
                            .push(self.strides.first().copied().unwrap_or(1) as u32)
                            .push(self.strides.get(1).copied().unwrap_or(1) as u32)
                            .push(self.strides.get(2).copied().unwrap_or(1) as u32)
                            .push(self.dilations.first().copied().unwrap_or(1) as u32)
                            .push(self.dilations.get(1).copied().unwrap_or(1) as u32)
                            .push(self.dilations.get(2).copied().unwrap_or(1) as u32)
                            .push(pb.first().copied().unwrap_or(0) as u32)
                            .push(pb.get(1).copied().unwrap_or(0) as u32)
                            .push(pb.get(2).copied().unwrap_or(0) as u32)
                            .push(self.group as u32)
                            .push(if self.bias.is_some() { 1u32 } else { 0u32 });

                        let out_w = dst_dims[4] as u32;
                        let out_h = dst_dims[3] as u32;
                        let out_d = dst_dims[2] as u32;
                        let total_z = out_d * (dst_dims[0] as u32) * (dst_dims[1] as u32);

                        (
                            &CONV_3D_SHADER,
                            gpu.workgroup_size_3d(),
                            pc_values,
                            [out_w, out_h, total_z],
                        )
                    }
                    _ => unreachable!(),
                };

                let Some(spv) = gpu.get_spirv(shader, dtype) else {
                    return Ok(None);
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
        // Acquire read guards and extract copies of metadata and input bytes so we can
        // drop the read guards before taking a mutable write guard on dst.
        let src_guard = cm.tensor_read(self.src);
        let weights_guard = cm.tensor_read(self.weights);
        let bias_guard_opt = self.bias.map(|bid| cm.tensor_read(bid));

        let src_desc = src_guard.desc();
        let weight_desc = weights_guard.desc();
        let src_bytes_vec: Vec<u8> = src_guard.get_cpu_memory_slice_or_panic().to_vec();
        let weight_bytes_vec: Vec<u8> = weights_guard.get_cpu_memory_slice_or_panic().to_vec();
        let bias_bytes_vec_opt: Option<Vec<u8>> = bias_guard_opt
            .as_ref()
            .map(|t| t.get_cpu_memory_slice_or_panic().to_vec());

        // Obtain dst as mutable write guard
        let dst_tensor = cm.tensor_write(self.dst);
        let dst_desc = dst_tensor.desc().clone();

        // Get raw bytes as slices referencing our copied vecs
        let src_bytes: &[u8] = src_bytes_vec.as_slice();
        let weight_bytes: &[u8] = weight_bytes_vec.as_slice();
        let bias_bytes_opt: Option<&[u8]> = bias_bytes_vec_opt.as_deref();
        let dst_ptr = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        let pads_begin = self.compute_pads(src_desc);

        // Dispatch based on data type
        let src_dtype = src_desc.data_type();
        let weight_dtype = weight_desc.data_type();
        let bias_dtype_opt = bias_guard_opt.as_ref().map(|t| t.desc().data_type());
        let dst_dtype = dst_desc.data_type();
        match (src_dtype, weight_dtype, bias_dtype_opt, dst_dtype) {
            (DataType::Float, DataType::Float, None, DataType::Float)
            | (DataType::Float, DataType::Float, Some(DataType::Float), DataType::Float) => {
                f32_f32_f32_f32_cpu(
                    src_desc.dims(),
                    weight_desc.dims(),
                    dst_desc.dims(),
                    src_bytes,
                    weight_bytes,
                    bias_bytes_opt,
                    dst_ptr,
                    &self.strides,
                    &pads_begin,
                    &self.dilations,
                    self.group as usize,
                );
            }
            _ => unimplemented!(
                "CPU Conv unimplemented for DataType src:{:?}, weight:{:?}, bias:{:?}, dst:{:?}",
                src_dtype,
                weight_dtype,
                bias_dtype_opt
                    .map(|dt| format!("{:?}", dt))
                    .unwrap_or_else(|| "None".to_string()),
                dst_dtype
            ),
        }
    }
}
