#[allow(non_camel_case_types)]
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum GPUOperation {
    // Slang fully dynamic generic variants
    Addition,
    Addition_NoStride,
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
    ReLU,
    Sigmoid,
    Expand,
    ReduceMean,
    Shape_Write,
    MaxPool_1D,
    MaxPool_2D,
    MaxPool_3D,
    Softmax,
    Softmax_SGA,
    Conv_1D,
    Conv_2D,
    Conv_3D,
    MatMul_1D2D,
    MatMul_2D1D,
    MatMul_2D2D,
    MatMul_2D3D,
    MatMul_3D2D,
    MatMul_3D3D,
    MatMul_3D1D,
    MatMul_1D3D,
    MatMul_2D2D_Tiled,
    Gemm,
    Gemm_2D2D_Tiled,
}

impl GPUOperation {
    pub fn as_str(&self) -> &'static str {
        match self {
            GPUOperation::Addition => "addition",
            GPUOperation::Addition_NoStride => "addition_nostride",
            GPUOperation::Subtract => "subtract",
            GPUOperation::Multiply => "multiply",
            GPUOperation::Divide => "divide",
            GPUOperation::Maximum => "maximum",
            GPUOperation::Minimum => "minimum",
            GPUOperation::ReLU => "relu",
            GPUOperation::Sigmoid => "sigmoid",
            GPUOperation::Expand => "expand",
            GPUOperation::ReduceMean => "reducemean",
            GPUOperation::Shape_Write => "shape_write",
            GPUOperation::MaxPool_1D => "maxpool_1d",
            GPUOperation::MaxPool_2D => "maxpool_2d",
            GPUOperation::MaxPool_3D => "maxpool_3d",
            GPUOperation::Softmax => "softmax",
            GPUOperation::Softmax_SGA => "softmax_sga",
            GPUOperation::Conv_1D => "conv_1d",
            GPUOperation::Conv_2D => "conv_2d",
            GPUOperation::Conv_3D => "conv_3d",
            GPUOperation::MatMul_1D2D => "matmul_1d2d",
            GPUOperation::MatMul_2D1D => "matmul_2d1d",
            GPUOperation::MatMul_2D2D => "matmul_2d2d",
            GPUOperation::MatMul_2D3D => "matmul_2d3d",
            GPUOperation::MatMul_3D2D => "matmul_3d2d",
            GPUOperation::MatMul_3D3D => "matmul_3d3d",
            GPUOperation::MatMul_3D1D => "matmul_3d1d",
            GPUOperation::MatMul_1D3D => "matmul_1d3d",
            GPUOperation::MatMul_2D2D_Tiled => "matmul_2d2d_tiled",
            GPUOperation::Gemm => "gemm",
            GPUOperation::Gemm_2D2D_Tiled => "gemm_2d2d_tiled",
        }
    }

    pub fn binding_count(&self) -> usize {
        match self {
            GPUOperation::Addition => 3,
            GPUOperation::Addition_NoStride => 3,
            GPUOperation::Subtract => 3,
            GPUOperation::Multiply => 3,
            GPUOperation::Divide => 3,
            GPUOperation::Maximum => 3,
            GPUOperation::Minimum => 3,
            GPUOperation::ReLU => 2,
            GPUOperation::Sigmoid => 2,
            GPUOperation::Expand => 2,
            GPUOperation::ReduceMean => 2,
            GPUOperation::Shape_Write => 1,
            GPUOperation::MaxPool_1D => 2,
            GPUOperation::MaxPool_2D => 2,
            GPUOperation::MaxPool_3D => 2,
            GPUOperation::Softmax => 2,
            GPUOperation::Softmax_SGA => 2,
            GPUOperation::Conv_1D => 4,
            GPUOperation::Conv_2D => 4,
            GPUOperation::Conv_3D => 4,
            GPUOperation::MatMul_1D2D
            | GPUOperation::MatMul_2D1D
            | GPUOperation::MatMul_2D2D
            | GPUOperation::MatMul_2D3D
            | GPUOperation::MatMul_3D2D
            | GPUOperation::MatMul_3D3D
            | GPUOperation::MatMul_3D1D
            | GPUOperation::MatMul_1D3D
            | GPUOperation::MatMul_2D2D_Tiled => 3,
            GPUOperation::Gemm | GPUOperation::Gemm_2D2D_Tiled => 4,
        }
    }

    pub fn to_slang_shader(&self) -> Result<&'static [u8], crate::utils::error::VKMLError> {
        match self {
            GPUOperation::Addition => Ok(include_bytes!("add/add.slang")),
            GPUOperation::Addition_NoStride => Ok(include_bytes!("add/add_nostride.slang")),
            GPUOperation::Subtract => Ok(include_bytes!("sub/sub.slang")),
            GPUOperation::Multiply => Ok(include_bytes!("mul/mul.slang")),
            GPUOperation::Divide => Ok(include_bytes!("div/div.slang")),
            GPUOperation::Maximum => Ok(include_bytes!("max/max.slang")),
            GPUOperation::Minimum => Ok(include_bytes!("min/min.slang")),
            GPUOperation::ReLU => Ok(include_bytes!("relu/relu.slang")),
            GPUOperation::Sigmoid => Ok(include_bytes!("sigmoid/sigmoid.slang")),
            GPUOperation::Expand => Ok(include_bytes!("expand/expand.slang")),
            GPUOperation::ReduceMean => Ok(include_bytes!("reducemean/reducemean.slang")),
            GPUOperation::Shape_Write => Ok(include_bytes!("shape/shape.slang")),
            GPUOperation::MaxPool_1D => Ok(include_bytes!("maxpool/maxpool_1d.slang")),
            GPUOperation::MaxPool_2D => Ok(include_bytes!("maxpool/maxpool_2d.slang")),
            GPUOperation::MaxPool_3D => Ok(include_bytes!("maxpool/maxpool_3d.slang")),
            GPUOperation::Softmax => Ok(include_bytes!("softmax/softmax.slang")),
            GPUOperation::Softmax_SGA => Ok(include_bytes!("softmax/softmax_sga.slang")),
            GPUOperation::Conv_1D => Ok(include_bytes!("conv/conv_1d.slang")),
            GPUOperation::Conv_2D => Ok(include_bytes!("conv/conv_2d.slang")),
            GPUOperation::Conv_3D => Ok(include_bytes!("conv/conv_3d.slang")),
            GPUOperation::MatMul_1D2D => Ok(include_bytes!("matmul/matmul_1d2d.slang")),
            GPUOperation::MatMul_2D1D => Ok(include_bytes!("matmul/matmul_2d1d.slang")),
            GPUOperation::MatMul_2D2D => Ok(include_bytes!("matmul/matmul_2d2d.slang")),
            GPUOperation::MatMul_2D3D => Ok(include_bytes!("matmul/matmul_2d3d.slang")),
            GPUOperation::MatMul_3D2D => Ok(include_bytes!("matmul/matmul_3d2d.slang")),
            GPUOperation::MatMul_3D3D => Ok(include_bytes!("matmul/matmul_3d3d.slang")),
            GPUOperation::MatMul_3D1D => Ok(include_bytes!("matmul/matmul_3d1d.slang")),
            GPUOperation::MatMul_1D3D => Ok(include_bytes!("matmul/matmul_1d3d.slang")),
            GPUOperation::MatMul_2D2D_Tiled => Ok(include_bytes!("matmul/matmul_tiled.slang")),
            GPUOperation::Gemm => Ok(include_bytes!("gemm/gemm.slang")),
            GPUOperation::Gemm_2D2D_Tiled => Ok(include_bytes!("gemm/gemm_tiled.slang")),
        }
    }
}
