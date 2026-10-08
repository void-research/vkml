pub mod add;
pub mod concat;
pub mod conv;
pub mod dispatch;
pub mod div;
pub mod expand;
pub mod gemm;
pub mod identity;
pub mod matmul;
pub mod max;
pub mod maxpool;
pub mod min;
pub mod mul;
pub mod reducemean;
pub mod relu;
pub mod reshape;
pub mod shape;
pub mod sigmoid;
pub mod softmax;
pub mod sub;
pub mod transfer;

pub use add::AddInstruction;
pub use concat::ConcatInstruction;
pub use conv::ConvInstruction;
pub use div::DivInstruction;
pub use expand::ExpandInstruction;
pub use gemm::GemmInstruction;
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

pub use crate::gpu::PushConstants;
pub use crate::slang::{Shader, slang};
pub use crate::utils::dtype::{ARITHMETIC_TYPES, FLOAT_TYPES};
pub use dispatch::{Dispatch, VkOperation};

use crate::{
    ComputeManager, tensor::ComputeTarget, tensor_graph::TensorId, utils::error::VKMLError,
};
use std::fmt::Debug;

pub trait Instruction: Debug {
    // Get all input tensor IDs used by this instruction
    fn get_input_tensor_ids(&self) -> Vec<TensorId>;

    // Get all output tensor IDs for this instruction
    fn get_output_tensor_ids(&self) -> Vec<TensorId>;

    // Remap tensor IDs (used during graph construction)
    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]);

    // Select execution dispatch for a given compute target
    fn select_operation(
        &self,
        target: &ComputeTarget,
        cm: &ComputeManager,
    ) -> Result<Option<Dispatch>, VKMLError>;

    // Execute on CPU (default implementation returns error)
    fn execute_cpu(&self, _cm: &ComputeManager) {
        panic!("CPU execution not implemented for {:?}", self)
    }
}
