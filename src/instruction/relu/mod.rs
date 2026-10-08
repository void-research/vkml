mod f32_f32_cpu;

use crate::ComputeManager;
use crate::VKMLError;
use crate::instruction::{
    Dispatch, Instruction, PushConstants, Shader, VkOperation, relu::f32_f32_cpu::f32_f32_cpu,
    slang,
};
use crate::tensor::ComputeTarget;
use crate::tensor_graph::TensorId;
use crate::utils::math::{broadcast_shape, broadcast_strides};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};

pub static SHADER: Shader = slang!("relu.slang", 2);

pub struct ReLUInstruction {
    pub src: TensorId,
    pub dst: TensorId,
}

impl Debug for ReLUInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "ReLU(src={}, dst={})", self.src, self.dst)
    }
}

impl Instruction for ReLUInstruction {
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
        // Follow add.rs style: compute broadcast shapes/strides and dispatch to typed helpers
        assert!(
            self.src != self.dst,
            "Cannot use ReLU for in-place operation"
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
                f32_f32_cpu(sa, c, src_bytes, dst_ptr);
            }
            _ => unimplemented!(
                "relu.rs unimplemented cpu instruction for DataType src:{:?}, dst:{:?}",
                src_dtype,
                dst_dtype
            ),
        }
    }
}
