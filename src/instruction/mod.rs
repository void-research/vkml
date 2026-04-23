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
pub use gpu_operations::GPUOperation;
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
    ComputeManager,
    gpu::vk_gpu::Gpu,
    tensor::DeviceId,
    tensor_graph::TensorId,
    utils::{OnnxAutoPad, error::VKMLError},
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

    // Record this instruction into an already begun command buffer
    fn record_into_command_buffer(
        &self,
        _gpu: &Gpu,
        _command_buffer: vk::CommandBuffer,
        _cm: &ComputeManager,
    ) -> Result<(), VKMLError> {
        Err(VKMLError::Vulkan(format!(
            "GPU execution not implemented for {:?}",
            self
        )))
    }

    // Execute on CPU (default implementation returns error)
    fn execute_cpu(&self, _cm: &ComputeManager) {
        panic!("CPU execution not implemented for {:?}", self)
    }

    // Return true if this instruction must be executed on the CPU (eg transfers)
    fn must_execute_on_cpu(&self) -> bool {
        false
    }
}

pub fn add(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(AddInstruction { src1, src2, dst })
}

pub fn concat(sources: Vec<TensorId>, dst: TensorId, dim: usize) -> Box<dyn Instruction> {
    Box::new(ConcatInstruction { sources, dst, dim })
}

pub fn conv(
    src: TensorId,
    weights: TensorId,
    bias: Option<TensorId>,
    dst: TensorId,
    auto_pad: OnnxAutoPad,
    dilations: Vec<i64>,
    group: i64,
    kernel_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
) -> Box<dyn Instruction> {
    Box::new(ConvInstruction {
        src,
        weights,
        bias,
        dst,
        auto_pad,
        dilations,
        group,
        kernel_shape,
        pads,
        strides,
    })
}

pub fn div(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(DivInstruction { src1, src2, dst })
}

pub fn expand(src: TensorId, dst: TensorId, shape: Vec<i64>) -> Box<dyn Instruction> {
    Box::new(ExpandInstruction {
        src,
        dst,
        shape_values: shape,
    })
}

pub fn identity(src: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(IdentityInstruction { src, dst })
}

pub fn matmul(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(MatMulInstruction { src1, src2, dst })
}

pub fn max(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(MaxInstruction { src1, src2, dst })
}

pub fn maxpool(
    src: TensorId,
    dst: TensorId,
    auto_pad: OnnxAutoPad,
    dilations: Vec<i64>,
    kernel_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
    ceil_mode: bool,
) -> Box<dyn Instruction> {
    Box::new(MaxPoolInstruction {
        src,
        dst,
        auto_pad,
        dilations,
        kernel_shape,
        pads,
        strides,
        ceil_mode,
    })
}

pub fn min(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(MinInstruction { src1, src2, dst })
}

pub fn mul(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(MulInstruction { src1, src2, dst })
}

pub fn relu(src: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(ReLUInstruction { src, dst })
}

pub fn reducemean(
    src: TensorId,
    axes: Option<Vec<i64>>,
    keepdims: i64,
    noop_with_empty_axes: i64,
    dst: TensorId,
) -> Box<dyn Instruction> {
    Box::new(ReduceMeanInstruction {
        src,
        axes,
        keepdims,
        noop_with_empty_axes,
        dst,
    })
}

pub fn reshape(
    src: TensorId,
    dst: TensorId,
    new_shape: Vec<i64>,
    allowzero: Option<i64>,
) -> Box<dyn Instruction> {
    Box::new(ReshapeInstruction {
        src,
        dst,
        shape_values: new_shape,
        allowzero,
    })
}

pub fn shape(
    src: TensorId,
    dst: TensorId,
    start: Option<i64>,
    end: Option<i64>,
) -> Box<dyn Instruction> {
    Box::new(ShapeInstruction {
        src,
        dst,
        start,
        end,
    })
}

pub fn sigmoid(src: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(SigmoidInstruction { src, dst })
}

pub fn softmax(src: TensorId, dst: TensorId, axis: Option<i64>) -> Box<dyn Instruction> {
    Box::new(SoftmaxInstruction { src, dst, axis })
}

pub fn sub(src1: TensorId, src2: TensorId, dst: TensorId) -> Box<dyn Instruction> {
    Box::new(SubInstruction { src1, src2, dst })
}

pub fn transfer(
    src: TensorId,
    dst: TensorId,
    source_device: DeviceId,
    target_device: DeviceId,
) -> Box<dyn Instruction> {
    Box::new(TransferToDeviceInstruction {
        src,
        dst,
        source_device,
        target_device,
    })
}

pub fn gemm(
    a: TensorId,
    b: TensorId,
    c: Option<TensorId>,
    y: TensorId,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
) -> Box<dyn Instruction> {
    Box::new(GemmInstruction {
        a,
        b,
        c,
        y,
        alpha,
        beta,
        trans_a,
        trans_b,
    })
}
