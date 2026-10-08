mod f32_f32_cpu;
mod push_constants;

use crate::ComputeManager;
use crate::VKMLError;
use crate::instruction::expand::f32_f32_cpu::f32_f32_cpu;
use crate::instruction::expand::push_constants::ExpandPushConstants;
use crate::utils::math::broadcast_strides;
use crate::{
    instruction::{Dispatch, Instruction, PushConstants, Shader, VkOperation, slang},
    tensor::ComputeTarget,
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};

pub static SHADER: Shader = slang!("expand.slang", 2);

pub struct ExpandInstruction {
    pub src: TensorId,
    pub dst: TensorId,
    pub shape_values: Vec<i64>,
}

impl Debug for ExpandInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Expand(src={}, dst={}, shape={:?})",
            self.src, self.dst, self.shape_values
        )
    }
}

impl Instruction for ExpandInstruction {
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
        let dst_dims = dst_desc.dims();
        let dst_dtype = dst_desc.data_type();

        if dst_dims.len() > 8 {
            return Err(VKMLError::Instruction(format!(
                "Expand instruction {:?}: tensor rank exceeds max supported 8 (got {})",
                self,
                dst_dims.len()
            )));
        }

        match target {
            ComputeTarget::Gpu(gpu) => {
                if src_desc.data_type() != dst_dtype
                    || !gpu.supports_dtype(dst_dtype)
                    || !SHADER.supported_types.contains(&dst_dtype)
                {
                    return Ok(None);
                }

                let rank = dst_dims.len() as u32;

                let mut dims_arr = [0u32; 8];
                for (i, &d) in dst_dims.iter().enumerate().take(8) {
                    dims_arr[i] = d as u32;
                }

                let strides_src_usize = broadcast_strides(src_desc.dims(), dst_dims);

                let mut strides_src_arr = [0u32; 8];
                for (i, &s) in strides_src_usize.iter().enumerate().take(8) {
                    strides_src_arr[i] = s as u32;
                }

                let num_elements = dst_desc.num_elements() as u32;

                let push_const_values = ExpandPushConstants {
                    rank,
                    pad: 0,
                    total: num_elements,
                    dims: dims_arr,
                    strides_src: strides_src_arr,
                };

                let local_size = gpu.workgroup_size_1d();

                Ok(Some(Dispatch::Gpu(VkOperation::Compute {
                    shader: &SHADER,
                    dtype: dst_dtype,
                    local_size,
                    work_size: [num_elements, 1, 1],
                    push_constants: PushConstants::from_struct(&push_const_values),
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
        let src_tensor = cm.tensor_read(self.src);
        let dst_tensor = cm.tensor_write(self.dst);

        let src_dims = src_tensor.desc().dims();
        let dst_dims = dst_tensor.desc().dims().to_vec();

        // Verify that the expand is valid
        // According to ONNX spec, dimensions are right-aligned
        // Two corresponding dimensions must have the same value, or one of them is equal to 1
        let src_rank = src_dims.len();
        let dst_rank = dst_dims.len();

        // Pad src_dims on the left to match dst_rank
        let mut padded_src_dims = vec![1; dst_rank];
        let offset = dst_rank.saturating_sub(src_rank);
        for (i, &dim) in src_dims.iter().enumerate() {
            padded_src_dims[offset + i] = dim;
        }

        // Verify broadcast compatibility
        for i in 0..dst_rank {
            let src_dim = padded_src_dims[i];
            let dst_dim = dst_dims[i];
            if src_dim != dst_dim && src_dim != 1 {
                panic!(
                    "Expand: incompatible shapes src={:?} (padded={:?}), dst={:?}",
                    src_dims, padded_src_dims, dst_dims
                );
            }
        }

        // Calculate broadcast strides
        let strides_src = broadcast_strides(src_dims, &dst_dims);

        let src_dtype = src_tensor.desc().data_type();
        let dst_dtype = dst_tensor.desc().data_type();

        let src_bytes = src_tensor.get_cpu_memory_slice_or_panic();
        let dst_ptr = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        match (src_dtype, dst_dtype) {
            (DataType::Float, DataType::Float) => {
                f32_f32_cpu(strides_src, dst_dims, src_bytes, dst_ptr)
            }
            _ => unimplemented!(
                "expand.rs unimplemented cpu instruction for DataType src:{:?}, dst:{:?}",
                src_dtype,
                dst_dtype
            ),
        }
    }
}
