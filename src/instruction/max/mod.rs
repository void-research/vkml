mod f32_f32_f32_cpu;
mod push_constants;

use crate::ComputeManager;
use crate::VKMLError;
use crate::instruction::max::push_constants::MaxPushConstants;
use crate::utils::math::{broadcast_shape, broadcast_strides};
use crate::{
    instruction::{
        Dispatch, Instruction, PushConstants, Shader, VkOperation,
        max::f32_f32_f32_cpu::f32_f32_f32_cpu, slang,
    },
    tensor::ComputeTarget,
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};

pub static SHADER: Shader = slang!("max.slang", 3);

pub struct MaxInstruction {
    pub src1: TensorId,
    pub src2: TensorId,
    pub dst: TensorId,
}

impl Debug for MaxInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Max(src1={}, src2={}, dst={})",
            self.src1, self.src2, self.dst
        )
    }
}

impl Instruction for MaxInstruction {
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
        let dst_dims = dst_desc.dims();
        let dst_dtype = dst_desc.data_type();

        if broadcast_shape(src1_desc.dims(), src2_desc.dims()).is_none() {
            return Err(VKMLError::Instruction(format!(
                "Max instruction {:?}: cannot broadcast shapes {:?} and {:?}",
                self,
                src1_desc.dims(),
                src2_desc.dims()
            )));
        }

        if dst_dims.len() > 8 {
            return Err(VKMLError::Instruction(format!(
                "Max instruction {:?}: tensor rank exceeds max supported 8 (got {})",
                self,
                dst_dims.len()
            )));
        }

        match target {
            ComputeTarget::Gpu(gpu) => {
                if src1_desc.data_type() != dst_dtype
                    || src2_desc.data_type() != dst_dtype
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

                let strides_a_usize = broadcast_strides(src1_desc.dims(), dst_dims);
                let strides_b_usize = broadcast_strides(src2_desc.dims(), dst_dims);

                let mut strides_a_arr = [0u32; 8];
                for (i, &s) in strides_a_usize.iter().enumerate().take(8) {
                    strides_a_arr[i] = s as u32;
                }

                let mut strides_b_arr = [0u32; 8];
                for (i, &s) in strides_b_usize.iter().enumerate().take(8) {
                    strides_b_arr[i] = s as u32;
                }

                let num_elements = dst_desc.num_elements() as u32;

                let push_const_values = MaxPushConstants {
                    rank,
                    pad: 0,
                    total: num_elements,
                    dims: dims_arr,
                    strides_a: strides_a_arr,
                    strides_b: strides_b_arr,
                };

                let local_size = gpu.workgroup_size_1d();

                Ok(Some(Dispatch::Gpu(VkOperation::Compute {
                    shader: &SHADER,
                    dtype: dst_dtype,
                    local_size,
                    work_size: [num_elements, 1, 1],
                    push_constants: PushConstants::from_struct(&push_const_values),
                    storage_buffers: vec![Some(self.src1), Some(self.src2), Some(self.dst)],
                })))
            }
            ComputeTarget::Cpu => {
                let compatible = src1_desc.data_type() == DataType::Float
                    && src2_desc.data_type() == DataType::Float
                    && dst_dtype == DataType::Float;
                if compatible {
                    Ok(Some(Dispatch::Cpu))
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        assert!(
            self.src1 != self.dst && self.src2 != self.dst,
            "Cannot use Max for in-place operation. Use MaxInplace instead."
        );
        let src1_tensor = cm.tensor_read(self.src1);
        let src2_tensor = cm.tensor_read(self.src2);
        let dst_tensor = cm.tensor_write(self.dst);

        let a = src1_tensor.desc().dims();
        let b = src2_tensor.desc().dims();
        let c = dst_tensor.desc().dims().to_vec();

        let bc =
            broadcast_shape(a, b).unwrap_or_else(|| panic!("Can't broadcast {:?} vs {:?}", a, b));
        assert_eq!(bc, c, "Broadcast {:?} != dst {:?}", bc, c);

        let sa = broadcast_strides(a, &c);
        let sb = broadcast_strides(b, &c);

        let src1_dtype = src1_tensor.desc().data_type();
        let src2_dtype = src2_tensor.desc().data_type();
        let dst_dtype = dst_tensor.desc().data_type();

        let src1_bytes = src1_tensor.get_cpu_memory_slice_or_panic();
        let src2_bytes = src2_tensor.get_cpu_memory_slice_or_panic();
        let dst_ptr = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        match (src1_dtype, src2_dtype, dst_dtype) {
            (DataType::Float, DataType::Float, DataType::Float) => {
                f32_f32_f32_cpu(sa, sb, c, src1_bytes, src2_bytes, dst_ptr)
            }
            _ => unimplemented!(
                "max.rs unimplemented cpu instruction for DataType src1:{:?}, src2:{:?}, dst:{:?}",
                src1_dtype,
                src2_dtype,
                dst_dtype
            ),
        }
    }
}
