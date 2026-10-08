mod f32_f32_cpu;

use crate::VKMLError;
use crate::utils::math::{broadcast_shape, broadcast_strides};
use crate::{
    ComputeManager,
    instruction::{
        Dispatch, FLOAT_TYPES, Instruction, PushConstants, Shader, VkOperation,
        sigmoid::f32_f32_cpu::f32_f32_cpu, slang,
    },
    tensor::ComputeTarget,
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};

pub static SHADER: Shader = slang!("sigmoid.slang", 2, FLOAT_TYPES);

pub struct SigmoidInstruction {
    pub src: TensorId,
    pub dst: TensorId,
}

impl Debug for SigmoidInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "Sigmoid(src={}, dst={})", self.src, self.dst)
    }
}

impl Instruction for SigmoidInstruction {
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

                let num_elements = dst_desc.num_elements() as u32;
                let local_size = gpu.workgroup_size_1d();

                Ok(Some(Dispatch::Gpu(VkOperation::Compute {
                    shader: &SHADER,
                    dtype: dst_dtype,
                    local_size,
                    work_size: [num_elements, 1, 1],
                    push_constants: PushConstants::from_struct(&num_elements),
                    storage_buffers: vec![Some(self.src), Some(self.dst)],
                })))
            }
            ComputeTarget::Cpu => {
                if src_desc.data_type() == DataType::Float && dst_dtype == DataType::Float {
                    Ok(Some(Dispatch::Cpu))
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        assert!(
            self.src != self.dst,
            "Cannot use Sigmoid for in-place operation"
        );

        let src_tensor = cm.tensor_read(self.src);
        let dst_tensor = cm.tensor_write(self.dst);

        let a = src_tensor.desc().dims();
        let c = dst_tensor.desc().dims().to_vec();

        let bc =
            broadcast_shape(a, &c).unwrap_or_else(|| panic!("Can't broadcast {:?} vs {:?}", a, c));
        assert_eq!(bc.as_slice(), c, "Broadcast {:?} != dst {:?}", bc, c);

        let sa = broadcast_strides(a, &c);

        let src_dtype = src_tensor.desc().data_type();
        let dst_dtype = dst_tensor.desc().data_type();

        let src_bytes = src_tensor.get_cpu_memory_slice_or_panic();
        let dst_ptr = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        match (src_dtype, dst_dtype) {
            (DataType::Float, DataType::Float) => {
                f32_f32_cpu(sa, vec![], c, src_bytes, &[], dst_ptr)
            }
            _ => unimplemented!(
                "sigmoid.rs unimplemented cpu instruction for DataType src:{:?}, dst:{:?}",
                src_dtype,
                dst_dtype
            ),
        }
    }
}
