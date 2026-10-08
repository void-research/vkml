mod f32_cpu;
pub mod push_constants;

use crate::VKMLError;
use crate::instruction::reducemean::f32_cpu::f32_cpu;
use crate::instruction::reducemean::push_constants::ReduceMeanPushConstants;
use crate::instruction::{Dispatch, Instruction, PushConstants, Shader, VkOperation, slang};
use crate::{
    ComputeManager,
    tensor::{ComputeTarget, TensorDesc},
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};

pub static SHADER: Shader = slang!("reducemean.slang", 2);

pub struct ReduceMeanInstruction {
    pub src: TensorId,
    pub axes: Option<Vec<i64>>,
    pub keepdims: i64,
    pub noop_with_empty_axes: i64,
    pub dst: TensorId,
}

impl Debug for ReduceMeanInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "ReduceMean(src={}, axes={:?}, keepdims={}, dst={})",
            self.src, self.axes, self.keepdims, self.dst
        )
    }
}

impl Instruction for ReduceMeanInstruction {
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
        let dst_dtype = dst_desc.data_type();

        match target {
            ComputeTarget::Gpu(gpu) => {
                if src_desc.data_type() != dst_dtype
                    || !gpu.supports_dtype(dst_dtype)
                    || !SHADER.supported_types.contains(&dst_dtype)
                {
                    return Ok(None);
                }

                let rank = src_desc.ndim() as i64;
                let axes_vec: Vec<i64> = if let Some(a) = &self.axes {
                    a.clone()
                } else if self.noop_with_empty_axes != 0 {
                    Vec::new()
                } else {
                    (0..rank).collect()
                };

                if axes_vec.is_empty() && self.noop_with_empty_axes != 0 {
                    return Ok(Some(Dispatch::Gpu(VkOperation::Copy {
                        src: self.src,
                        dst: self.dst,
                    })));
                }

                let mut reduction_size: u64 = 1;
                for &a in &axes_vec {
                    reduction_size *= src_desc.dims()[a as usize] as u64;
                }

                let out_elements = dst_desc.num_elements() as u32;

                let mean_pc = ReduceMeanPushConstants {
                    total: out_elements,
                    reduction_size: reduction_size as u32,
                };

                let local_size = gpu.workgroup_size_1d();

                Ok(Some(Dispatch::Gpu(VkOperation::Compute {
                    shader: &SHADER,
                    dtype: dst_dtype,
                    local_size,
                    work_size: [out_elements, 1, 1],
                    push_constants: PushConstants::from_struct(&mean_pc),
                    storage_buffers: vec![Some(self.src), Some(self.dst)],
                })))
            }
            ComputeTarget::Cpu => {
                let compatible =
                    src_desc.data_type() == DataType::Float && dst_dtype == DataType::Float;
                if compatible {
                    Ok(Some(Dispatch::Cpu))
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        // Basic CPU implementation: compute mean over axes
        let src_t = cm.tensor_read(self.src);
        let src_desc = src_t.desc();
        let src_dims = src_desc.dims().to_vec();
        let rank = src_dims.len() as i64;

        // Determine axes to reduce
        let axes_vec: Vec<i64> = if let Some(a) = &self.axes {
            a.clone()
        } else {
            // No axes => reduce over all axes unless noop_with_empty_axes == 1
            if self.noop_with_empty_axes != 0 {
                Vec::new()
            } else {
                (0..rank).collect()
            }
        };

        // If noop_with_empty_axes==1 and axes empty => copy input to output
        if axes_vec.is_empty() && self.noop_with_empty_axes != 0 {
            let src_bytes = src_t.get_cpu_memory_slice_or_panic();
            let dst_t = cm.tensor_write(self.dst);
            *dst_t.desc_mut() = src_desc.clone();
            dst_t
                .get_cpu_memory_mut_slice_or_panic()
                .copy_from_slice(src_bytes);
            return;
        }

        // Compute output shape
        let keep = self.keepdims != 0;
        let mut out_dims: Vec<i64> = Vec::new();
        for (i, &d) in src_dims.iter().enumerate() {
            if axes_vec.contains(&(i as i64)) {
                if keep {
                    out_dims.push(1);
                }
            } else {
                out_dims.push(d);
            }
        }
        if !keep && out_dims.is_empty() {
            out_dims.push(1); // scalar -> 1-element tensor representation
        }

        // Update dst descriptor
        let dst_t = cm.tensor_write(self.dst);
        *dst_t.desc_mut() = TensorDesc::new(out_dims.clone(), src_desc.data_type());

        let src_bytes = src_t.get_cpu_memory_slice_or_panic();
        let out_bytes = dst_t.get_cpu_memory_mut_slice_or_panic();

        // For simplicity, only implement numeric float32 path
        match src_desc.data_type() {
            DataType::Float => {
                f32_cpu(src_bytes, &src_dims, &axes_vec, keep, out_bytes);
            }
            _ => unimplemented!(
                "ReduceMean CPU: DataType {:?} not implemented",
                src_desc.data_type()
            ),
        }
    }
}
