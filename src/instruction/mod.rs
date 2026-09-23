mod add;
mod concat;
mod conv;
mod div;
mod expand;
mod gemm;
mod gpu_operations;
mod identity;
mod matmul;
mod max;
mod maxpool;
mod min;
mod mul;
mod reducemean;
mod relu;
mod reshape;
mod shape;
mod sigmoid;
mod softmax;
mod sub;
mod transfer;

pub use add::AddInstruction;
pub use concat::ConcatInstruction;
pub use conv::ConvInstruction;
pub use div::DivInstruction;
pub use expand::ExpandInstruction;
pub use gemm::GemmInstruction;
pub use gpu_operations::GpuShader;
pub use identity::IdentityInstruction;
pub use matmul::MatMulInstruction;
pub use max::MaxInstruction;
pub use maxpool::MaxPoolInstruction;
pub use min::MinInstruction;
pub use mul::MulInstruction;
pub use reducemean::ReduceMeanInstruction;
pub use relu::ReLUInstruction;
pub use reshape::ReshapeInstruction;
pub use shape::ShapeInstruction;
pub use sigmoid::SigmoidInstruction;
pub use softmax::SoftmaxInstruction;
pub use sub::SubInstruction;
pub use transfer::TransferToDeviceInstruction;

use crate::{
    ComputeManager, gpu::Gpu, tensor::ComputeTarget, tensor_graph::TensorId,
    utils::error::VKMLError,
};
use std::fmt::Debug;
use vulkanalia::vk;

pub trait Instruction: Debug {
    // Get all input tensor IDs used by this instruction
    fn get_input_tensor_ids(&self) -> Vec<TensorId>;

    // Get all output tensor IDs for this instruction
    fn get_output_tensor_ids(&self) -> Vec<TensorId>;

    // Remap tensor IDs (used during graph construction)
    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]);

    // Check if this instruction can execute on the specified device with its tensor shapes, attributes, and dtypes
    fn can_run_on(&self, target: &ComputeTarget, cm: &ComputeManager) -> Result<bool, VKMLError>;

    // Record this instruction into an already begun command buffer
    fn record_into_command_buffer(
        &self,
        _gpu: &Gpu,
        _command_buffer: vk::CommandBuffer,
        _cm: &ComputeManager,
    ) -> Result<(), VKMLError> {
        Err(VKMLError::Instruction(format!(
            "GPU execution not implemented for {:?}",
            self
        )))
    }

    // Execute on CPU (default implementation returns error)
    fn execute_cpu(&self, _cm: &ComputeManager) {
        panic!("CPU execution not implemented for {:?}", self)
    }
}
