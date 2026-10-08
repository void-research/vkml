mod f32_f32_cpu;
mod push_constants;

use crate::ComputeManager;
use crate::VKMLError;
use crate::instruction::softmax::push_constants::SoftmaxPushConstants;

use crate::{
    instruction::{
        Dispatch, FLOAT_TYPES, Instruction, PushConstants, Shader, VkOperation, slang,
        softmax::f32_f32_cpu::f32_f32_cpu,
    },
    tensor::ComputeTarget,
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};

pub static SHADER: Shader = slang!("softmax.slang", 2, FLOAT_TYPES);

pub struct SoftmaxInstruction {
    pub src: TensorId,
    pub dst: TensorId,
    pub axis: Option<i64>,
}

impl Debug for SoftmaxInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Softmax(src={}, dst={}, axis={:?})",
            self.src, self.dst, self.axis
        )
    }
}

impl SoftmaxInstruction {
    fn resolve_axis(&self, rank: usize) -> usize {
        let axis = self.axis.unwrap_or(-1);
        if axis < 0 {
            (rank as i64 + axis) as usize
        } else {
            axis as usize
        }
    }
}

impl Instruction for SoftmaxInstruction {
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

                let dims = src_desc.dims();
                let dim = self.resolve_axis(dims.len());

                let feature_size = dims[dim] as usize;
                let batch_size = src_desc.num_elements() / feature_size;

                let push_constants = SoftmaxPushConstants {
                    batch_size: batch_size as u32,
                    feature_size: feature_size as u32,
                };

                let local_size = [256, 1, 1];

                Ok(Some(Dispatch::Gpu(VkOperation::Compute {
                    shader: &SHADER,
                    dtype: dst_dtype,
                    local_size,
                    work_size: [(batch_size * local_size[0] as usize) as u32, 1, 1],
                    push_constants: PushConstants::from_struct(&push_constants),
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
        assert!(
            self.src != self.dst,
            "Cannot use Softmax for in-place operation"
        );

        let src_tensor = cm.tensor_read(self.src);
        let dst_tensor = cm.tensor_write(self.dst);

        let dims = src_tensor.desc().dims();
        let dim = self.resolve_axis(dims.len());

        assert_eq!(
            dim,
            dims.len() - 1,
            "CPU Softmax currently only supports the last dimension"
        );

        let src_dtype = src_tensor.desc().data_type();
        let dst_dtype = dst_tensor.desc().data_type();

        let src_bytes = src_tensor.get_cpu_memory_slice_or_panic();
        let dst_ptr = dst_tensor.get_cpu_memory_mut_slice_or_panic();

        match (src_dtype, dst_dtype) {
            (DataType::Float, DataType::Float) => {
                f32_f32_cpu(dims, dim, src_bytes, dst_ptr);
            }
            _ => unimplemented!(
                "softmax.rs unimplemented cpu instruction for DataType src:{:?}, dst:{:?}",
                src_dtype,
                dst_dtype
            ),
        }
    }
}
