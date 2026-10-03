pub mod add;
pub mod concat;
pub mod conv;
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

pub use crate::utils::dtype::{ARITHMETIC_TYPES, FLOAT_TYPES};

use crate::{
    ComputeManager, gpu::Gpu, tensor::ComputeTarget, tensor_graph::TensorId,
    utils::error::VKMLError,
};
use onnx_extractor::DataType;
use std::ffi::CStr;
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

macro_rules! slang {
    ($path:literal, $bindings:expr) => {
        slang!($path, $bindings, $crate::instruction::ARITHMETIC_TYPES)
    };
    ($path:literal, $bindings:expr, $types:expr) => {
        $crate::instruction::Shader {
            path: match std::ffi::CStr::from_bytes_with_nul(concat!($path, "\0").as_bytes()) {
                Ok(c) => c,
                Err(_) => panic!("shader path contains internal null byte"),
            },
            source: match std::ffi::CStr::from_bytes_with_nul(
                concat!(include_str!($path), "\0").as_bytes(),
            ) {
                Ok(c) => c,
                Err(_) => panic!("shader file contains internal null byte"),
            },
            binding_count: $bindings,
            supported_types: $types,
        }
    };
}
pub(crate) use slang;

pub struct Shader {
    pub path: &'static CStr,
    pub source: &'static CStr,
    pub binding_count: usize,
    pub supported_types: &'static [DataType],
}

impl Shader {
    pub fn can_run_on(&self, gpu: &Gpu, dtype: DataType) -> bool {
        if !self.supported_types.contains(&dtype) {
            return false;
        }
        if dtype == DataType::Float16 && !gpu.extensions().supports_fp16() {
            return false;
        }
        true
    }
}

impl PartialEq for Shader {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other) || self.path == other.path
    }
}

impl Eq for Shader {}

impl std::hash::Hash for Shader {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.path.hash(state);
    }
}
