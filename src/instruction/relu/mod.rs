mod f32_f32_cpu;

use crate::VKMLError;
use crate::utils::math::{broadcast_shape, broadcast_strides};
use crate::{
    ComputeManager,
    gpu::Gpu,
    instruction::{GpuShader, Instruction, relu::f32_f32_cpu::f32_f32_cpu},
    tensor::ComputeTarget,
    tensor_graph::TensorId,
};

use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use vulkanalia::vk;

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

    fn can_run_on(&self, target: &ComputeTarget, cm: &ComputeManager) -> Result<bool, VKMLError> {
        let src_desc = cm.tensor_desc(self.src);
        let dst_desc = cm.tensor_desc(self.dst);
        let dst_dtype = dst_desc.data_type();

        match target {
            ComputeTarget::Gpu(gpu) => {
                let compatible = src_desc.data_type() == dst_dtype
                    && GpuShader::ReLU.info().can_run_on(gpu, dst_dtype);
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
        let op_name = GpuShader::ReLU;

        let src_tensor = cm.tensor_read(self.src);
        let src_mem = src_tensor.get_gpu_memory_or_panic();
        let dst_tensor = cm.tensor_read(self.dst);
        let dst_mem = dst_tensor.get_gpu_memory_or_panic();

        let num_elements = dst_tensor.desc().num_elements();

        let dst_dtype = dst_tensor.desc().data_type();

        let local_size = gpu.workgroup_size_1d();

        gpu.bind_slang_compute_pipeline(command_buffer, op_name, dst_dtype, local_size);

        gpu.bind_storage_buffers(command_buffer, &[src_mem, dst_mem]);

        let pc_data = (num_elements as u32).to_ne_bytes();
        gpu.bind_push_constants(command_buffer, op_name, &pc_data);

        gpu.dispatch(command_buffer, local_size, [num_elements as u32, 1, 1]);

        Ok(())
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
