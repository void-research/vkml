use thiserror::Error;

#[derive(Error, Debug)]
pub enum VKMLError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vulkanalia::vk::ErrorCode),

    #[error("Compute Manager error: {0}")]
    ComputeManager(String),

    #[error("GPU error: {0}")]
    Gpu(String),

    #[error("GPU Pool error: {0}")]
    GpuPool(String),

    #[error("Graph Scheduler error: {0}")]
    GraphScheduler(String),

    #[error("Tensor Graph error: {0}")]
    TensorGraph(String),

    #[error("Graph Model error: {0}")]
    GraphModel(String),

    #[error("Layer error: {0}")]
    Layer(String),

    #[error("Instruction error: {0}")]
    Instruction(String),

    #[error("Onnx Importer error: {0}")]
    OnnxImporter(String),

    #[error("Slang error: {0}")]
    Slang(String),
}
