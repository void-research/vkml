mod f32_f32_cpu;

use crate::ComputeManager;
use crate::VKMLError;
use crate::gpu::shader::{Shader, shader};
use crate::utils::{OnnxAutoPad, calc_begin_and_end_pads};
use crate::{
    instruction::{
        Dispatch, Instruction, PushConstants, VkOperation, maxpool::f32_f32_cpu::f32_f32_cpu,
    },
    tensor::{ComputeTarget, TensorDesc},
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};

pub static MAXPOOL_1D_SHADER: Shader = shader!("maxpool_1d");
pub static MAXPOOL_2D_SHADER: Shader = shader!("maxpool_2d");
pub static MAXPOOL_3D_SHADER: Shader = shader!("maxpool_3d");

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

                        let pc = PushConstants::new()
                            .buffer(self.src)
                            .buffer(self.dst)
                            .push(src_dims[0] as u32)
                            .push(src_dims[1] as u32)
                            .push(input_len)
                            .push(output_len)
                            .push(self.kernel_shape.first().copied().unwrap_or(1) as u32)
                            .push(self.strides.first().copied().unwrap_or(1) as u32)
                            .push(self.dilations.first().copied().unwrap_or(1) as u32)
                            .push(pb.first().copied().unwrap_or(0) as u32);

                        let total = (src_dims[0] as u32) * (src_dims[1] as u32) * output_len;
                        (
                            &MAXPOOL_1D_SHADER,
                            gpu.workgroup_size_1d(),
                            pc,
                            [total, 1, 1],
                        )
                    }
                    2 => {
                        let pc = PushConstants::new()
                            .buffer(self.src)
                            .buffer(self.dst)
                            .push(src_dims[0] as u32)
                            .push(src_dims[1] as u32)
                            .push(src_dims[2] as u32)
                            .push(src_dims[3] as u32)
                            .push(dst_dims[2] as u32)
                            .push(dst_dims[3] as u32)
                            .push(self.kernel_shape.first().copied().unwrap_or(1) as u32)
                            .push(self.kernel_shape.get(1).copied().unwrap_or(1) as u32)
                            .push(self.strides.first().copied().unwrap_or(1) as u32)
                            .push(self.strides.get(1).copied().unwrap_or(1) as u32)
                            .push(self.dilations.first().copied().unwrap_or(1) as u32)
                            .push(self.dilations.get(1).copied().unwrap_or(1) as u32)
                            .push(pb.first().copied().unwrap_or(0) as u32)
                            .push(pb.get(1).copied().unwrap_or(0) as u32);

                        let out_w = dst_dims[3] as u32;
                        let out_h = dst_dims[2] as u32;
                        let batch_nc = (dst_dims[0] as u32) * (dst_dims[1] as u32);

                        (
                            &MAXPOOL_2D_SHADER,
                            gpu.workgroup_size_2d(),
                            pc,
                            [out_w, out_h, batch_nc],
                        )
                    }
                    3 => {
                        let pc = PushConstants::new()
                            .buffer(self.src)
                            .buffer(self.dst)
                            .push(src_dims[0] as u32)
                            .push(src_dims[1] as u32)
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
                            .push(pb.get(2).copied().unwrap_or(0) as u32);

                        let out_w = dst_dims[4] as u32;
                        let out_h = dst_dims[3] as u32;
                        let out_d = dst_dims[2] as u32;
                        let total_z = out_d * (dst_dims[0] as u32) * (dst_dims[1] as u32);

                        (
                            &MAXPOOL_3D_SHADER,
                            gpu.workgroup_size_3d(),
                            pc,
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
