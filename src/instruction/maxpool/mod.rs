mod f32_f32_cpu;
mod push_constants;

use crate::ComputeManager;
use crate::VKMLError;
use crate::instruction::maxpool::push_constants::{
    MaxPool1DPushConstants, MaxPool2DPushConstants, MaxPool3DPushConstants,
};
use crate::utils::{OnnxAutoPad, calc_begin_and_end_pads};
use crate::{
    instruction::{
        Dispatch, Instruction, PushConstants, Shader, VkOperation,
        maxpool::f32_f32_cpu::f32_f32_cpu, slang,
    },
    tensor::{ComputeTarget, TensorDesc},
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};

pub static MAXPOOL_1D_SHADER: Shader = slang!("maxpool_1d.slang", 2);
pub static MAXPOOL_2D_SHADER: Shader = slang!("maxpool_2d.slang", 2);
pub static MAXPOOL_3D_SHADER: Shader = slang!("maxpool_3d.slang", 2);

pub struct MaxPoolInstruction {
    pub src: TensorId,
    pub dst: TensorId,
    pub auto_pad: OnnxAutoPad,
    pub dilations: Vec<i64>,
    pub kernel_shape: Vec<i64>,
    pub pads: Vec<i64>,
    pub strides: Vec<i64>,
    pub ceil_mode: bool,
}

impl MaxPoolInstruction {
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

impl Debug for MaxPoolInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "MaxPool(src={}, dst={}, kernel={:?}, strides={:?}, pads={:?}, dilations={:?}, auto_pad={:?}, ceil_mode={})",
            self.src,
            self.dst,
            self.kernel_shape,
            self.strides,
            self.pads,
            self.dilations,
            self.auto_pad,
            self.ceil_mode
        )
    }
}

impl Instruction for MaxPoolInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.src]
    }

    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst]
    }

    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]) {
        if !new_inputs.is_empty() {
            self.src = new_inputs[0];
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
        let dst_desc = cm.tensor_desc(self.dst);
        let spatial_rank = src_desc.ndim().saturating_sub(2);
        if !(1..=3).contains(&spatial_rank) {
            return Err(VKMLError::Instruction(format!(
                "MaxPool instruction {:?}: spatial rank must be 1, 2, or 3, got {}",
                self, spatial_rank
            )));
        }

        match target {
            ComputeTarget::Cpu => {
                let compatible = src_desc.data_type() == DataType::Float
                    && dst_desc.data_type() == DataType::Float;
                if compatible {
                    Ok(Some(Dispatch::Cpu))
                } else {
                    Ok(None)
                }
            }
            ComputeTarget::Gpu(gpu) => {
                let dtype = src_desc.data_type();
                if dst_desc.data_type() != dtype {
                    return Ok(None);
                }

                let src_dims = src_desc.dims();
                let dst_dims = dst_desc.dims();
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

                        let pc = MaxPool1DPushConstants {
                            n: src_dims[0] as u32,
                            c: src_dims[1] as u32,
                            input_len,
                            output_len,
                            kernel: self.kernel_shape.first().copied().unwrap_or(1) as u32,
                            stride: self.strides.first().copied().unwrap_or(1) as u32,
                            dilation: self.dilations.first().copied().unwrap_or(1) as u32,
                            pad_begin: pb.first().copied().unwrap_or(0) as u32,
                        };

                        let total = (src_dims[0] as u32) * (src_dims[1] as u32) * output_len;
                        (
                            &MAXPOOL_1D_SHADER,
                            gpu.workgroup_size_1d(),
                            PushConstants::from_struct(&pc),
                            [total, 1, 1],
                        )
                    }
                    2 => {
                        let pc = MaxPool2DPushConstants {
                            n: src_dims[0] as u32,
                            c: src_dims[1] as u32,
                            in_h: src_dims[2] as u32,
                            in_w: src_dims[3] as u32,
                            out_h: dst_dims[2] as u32,
                            out_w: dst_dims[3] as u32,
                            k_h: self.kernel_shape.first().copied().unwrap_or(1) as u32,
                            k_w: self.kernel_shape.get(1).copied().unwrap_or(1) as u32,
                            s_h: self.strides.first().copied().unwrap_or(1) as u32,
                            s_w: self.strides.get(1).copied().unwrap_or(1) as u32,
                            d_h: self.dilations.first().copied().unwrap_or(1) as u32,
                            d_w: self.dilations.get(1).copied().unwrap_or(1) as u32,
                            pad_h: pb.first().copied().unwrap_or(0) as u32,
                            pad_w: pb.get(1).copied().unwrap_or(0) as u32,
                        };

                        let out_w = dst_dims[3] as u32;
                        let out_h = dst_dims[2] as u32;
                        let batch_nc = (dst_dims[0] as u32) * (dst_dims[1] as u32);

                        (
                            &MAXPOOL_2D_SHADER,
                            gpu.workgroup_size_2d(),
                            PushConstants::from_struct(&pc),
                            [out_w, out_h, batch_nc],
                        )
                    }
                    3 => {
                        let pc = MaxPool3DPushConstants {
                            n: src_dims[0] as u32,
                            c: src_dims[1] as u32,
                            in_d: src_dims[2] as u32,
                            in_h: src_dims[3] as u32,
                            in_w: src_dims[4] as u32,
                            out_d: dst_dims[2] as u32,
                            out_h: dst_dims[3] as u32,
                            out_w: dst_dims[4] as u32,
                            k_d: self.kernel_shape.first().copied().unwrap_or(1) as u32,
                            k_h: self.kernel_shape.get(1).copied().unwrap_or(1) as u32,
                            k_w: self.kernel_shape.get(2).copied().unwrap_or(1) as u32,
                            s_d: self.strides.first().copied().unwrap_or(1) as u32,
                            s_h: self.strides.get(1).copied().unwrap_or(1) as u32,
                            s_w: self.strides.get(2).copied().unwrap_or(1) as u32,
                            d_d: self.dilations.first().copied().unwrap_or(1) as u32,
                            d_h: self.dilations.get(1).copied().unwrap_or(1) as u32,
                            d_w: self.dilations.get(2).copied().unwrap_or(1) as u32,
                            pad_d: pb.first().copied().unwrap_or(0) as u32,
                            pad_h: pb.get(1).copied().unwrap_or(0) as u32,
                            pad_w: pb.get(2).copied().unwrap_or(0) as u32,
                        };

                        let out_w = dst_dims[4] as u32;
                        let out_h = dst_dims[3] as u32;
                        let out_d = dst_dims[2] as u32;
                        let total_z = out_d * (dst_dims[0] as u32) * (dst_dims[1] as u32);

                        (
                            &MAXPOOL_3D_SHADER,
                            gpu.workgroup_size_3d(),
                            PushConstants::from_struct(&pc),
                            [out_w, out_h, total_z],
                        )
                    }
                    _ => unreachable!(),
                };

                if !gpu.supports_dtype(dtype) || !shader.supported_types.contains(&dtype) {
                    return Ok(None);
                }

                Ok(Some(Dispatch::Gpu(VkOperation::Compute {
                    shader,
                    dtype,
                    local_size,
                    work_size,
                    push_constants,
                    storage_buffers: vec![Some(self.src), Some(self.dst)],
                })))
            }
        }
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        // Acquire read guard and dst write guard
        let src_guard = cm.tensor_read(self.src);
        let src_desc = src_guard.desc();
        let src_bytes = src_guard.get_cpu_memory_slice_or_panic();

        let dst_guard = cm.tensor_write(self.dst);
        let dst_desc = dst_guard.desc().clone();
        let dst_ptr = dst_guard.get_cpu_memory_mut_slice_or_panic();

        let pads_begin = self.compute_pads(src_desc);

        let src_dtype = src_desc.data_type();
        let dst_dtype = dst_desc.data_type();
        match (src_dtype, dst_dtype) {
            (DataType::Float, DataType::Float) => {
                f32_f32_cpu(
                    src_desc.dims(),
                    dst_desc.dims(),
                    src_bytes,
                    dst_ptr,
                    &self.kernel_shape,
                    &self.strides,
                    &pads_begin,
                    &self.dilations,
                );
            }
            _ => unimplemented!(
                "MaxPool: unimplemented CPU for DataType src:{:?}, dst:{:?}",
                src_dtype,
                dst_dtype
            ),
        }
    }
}
