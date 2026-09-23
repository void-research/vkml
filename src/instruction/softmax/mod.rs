mod f32_f32_cpu;
mod push_constants;

use crate::ComputeManager;
use crate::VKMLError;
use crate::instruction::softmax::push_constants::SoftmaxPushConstants;
use crate::utils::as_bytes;

use crate::{
    gpu::Gpu,
    instruction::{GpuShader, Instruction, softmax::f32_f32_cpu::f32_f32_cpu},
    tensor::ComputeTarget,
    tensor_graph::TensorId,
};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

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

    fn can_run_on(&self, target: &ComputeTarget, cm: &ComputeManager) -> Result<bool, VKMLError> {
        let src_desc = cm.tensor_desc(self.src);
        let dst_desc = cm.tensor_desc(self.dst);
        let dst_dtype = dst_desc.data_type();

        match target {
            ComputeTarget::Gpu(gpu) => {
                let compatible = src_desc.data_type() == dst_dtype
                    && GpuShader::Softmax.info().can_run_on(gpu, dst_dtype);
                Ok(compatible)
            }
            ComputeTarget::Cpu => {
                let compatible =
                    src_desc.data_type() == DataType::Float && dst_dtype == DataType::Float;
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
        let op_name = GpuShader::Softmax;

        let src_tensor = cm.tensor_read(self.src);
        let src_mem = src_tensor.get_gpu_memory_or_panic();
        let dst_tensor = cm.tensor_read(self.dst);
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        let dims = src_tensor.desc().dims();
        let dim = self.resolve_axis(dims.len());

        let feature_size = dims[dim] as usize;
        let batch_size = src_tensor.desc().num_elements() / feature_size;

        // Create push constants struct (compute before GPU ops)
        let push_constants = SoftmaxPushConstants {
            batch_size: batch_size as u32,
            feature_size: feature_size as u32,
        };

        let pc_bytes = as_bytes(&push_constants);
        let dst_dtype = dst_tensor.desc().data_type();

        // Standard path
        let gpu_op = op_name;
        let local_size = [256, 1, 1];

        gpu.bind_slang_compute_pipeline(command_buffer, gpu_op, dst_dtype, local_size);
        gpu.bind_storage_buffers(command_buffer, &[src_mem, dst_mem]);
        gpu.bind_push_constants(command_buffer, gpu_op, pc_bytes);

        gpu.dispatch(
            command_buffer,
            local_size,
            [batch_size as u64 * local_size[0] as u64, 1, 1],
        );

        Ok(())
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
