//! VKML - High-level abstractions for ML model development using Vulkan compute
//!
//! This library provides universal compute utilisation across different hardware vendors
//! with a focus on performance and ease of use.

mod utils;

mod slang;

pub mod gpu;

mod compute;
mod scheduler;

mod tensor;

mod instruction;

mod tensor_graph;

mod importers;

mod weight_initialiser;

pub use compute::compute_manager::ComputeManager;
pub use compute::optimisations::Optimisations;
pub use importers::onnx_parser;
pub use onnx_extractor::DataType;
pub use tensor::Tensor;
pub use tensor::TensorDesc;
pub use tensor_graph::TensorGraph;
pub use utils::error::VKMLError;
