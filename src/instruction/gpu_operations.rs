use std::ffi::CStr;

use crate::gpu::Gpu;
use onnx_extractor::DataType;

pub use crate::utils::dtype::{ARITHMETIC_TYPES, FLOAT_TYPES};

macro_rules! slang {
    ($path:literal, $bindings:expr) => {
        slang!($path, $bindings, ARITHMETIC_TYPES)
    };
    ($path:literal, $bindings:expr, $types:expr) => {
        ShaderInfo::new(
            match CStr::from_bytes_with_nul(concat!($path, "\0").as_bytes()) {
                Ok(c) => c,
                Err(_) => panic!("shader path contains internal null byte"),
            },
            match CStr::from_bytes_with_nul(concat!(include_str!($path), "\0").as_bytes()) {
                Ok(c) => c,
                Err(_) => panic!("shader file contains internal null byte"),
            },
            $bindings,
            $types,
        )
    };
}

#[allow(non_camel_case_types)]
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub enum GpuShader {
    Addition,
    Addition_NoStride,
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
    ReLU,
    Sigmoid,
    Softmax,
    Expand,
    ReduceMean,
    Shape_Write,
    MaxPool_1D,
    MaxPool_2D,
    MaxPool_3D,
    Conv_1D,
    Conv_2D,
    Conv_3D,
    MatMul_1D2D,
    MatMul_2D1D,
    MatMul_2D2D,
    MatMul_3D1D,
    MatMul_1D3D,
    MatMul_Tiled,
    Gemm,
    Gemm_Tiled,
}

#[derive(Clone, Copy, Debug)]
pub struct ShaderInfo {
    pub path: &'static CStr,
    pub source: &'static CStr,
    pub binding_count: usize,
    pub supported_types: &'static [DataType],
}

impl ShaderInfo {
    pub const fn new(
        path: &'static CStr,
        source: &'static CStr,
        binding_count: usize,
        supported_types: &'static [DataType],
    ) -> Self {
        Self {
            path,
            source,
            binding_count,
            supported_types,
        }
    }

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

impl GpuShader {
    pub const fn info(&self) -> ShaderInfo {
        match self {
            Self::Addition => slang!("add/add.slang", 3),
            Self::Addition_NoStride => slang!("add/add_nostride.slang", 3),
            Self::Subtract => slang!("sub/sub.slang", 3),
            Self::Multiply => slang!("mul/mul.slang", 3),
            Self::Divide => slang!("div/div.slang", 3),
            Self::Maximum => slang!("max/max.slang", 3),
            Self::Minimum => slang!("min/min.slang", 3),
            Self::ReLU => slang!("relu/relu.slang", 2),
            Self::Sigmoid => slang!("sigmoid/sigmoid.slang", 2, FLOAT_TYPES),
            Self::Softmax => slang!("softmax/softmax.slang", 2, FLOAT_TYPES),
            Self::Expand => slang!("expand/expand.slang", 2),
            Self::ReduceMean => slang!("reducemean/reducemean.slang", 2),
            Self::Shape_Write => slang!("shape/shape.slang", 1, &[DataType::Int64]),
            Self::MaxPool_1D => slang!("maxpool/maxpool_1d.slang", 2),
            Self::MaxPool_2D => slang!("maxpool/maxpool_2d.slang", 2),
            Self::MaxPool_3D => slang!("maxpool/maxpool_3d.slang", 2),
            Self::Conv_1D => slang!("conv/conv_1d.slang", 4),
            Self::Conv_2D => slang!("conv/conv_2d.slang", 4),
            Self::Conv_3D => slang!("conv/conv_3d.slang", 4),
            Self::MatMul_1D2D => slang!("matmul/matmul_1d2d.slang", 3),
            Self::MatMul_2D1D => slang!("matmul/matmul_2d1d.slang", 3),
            Self::MatMul_2D2D => slang!("matmul/matmul_2d2d.slang", 3),
            Self::MatMul_3D1D => slang!("matmul/matmul_3d1d.slang", 3),
            Self::MatMul_1D3D => slang!("matmul/matmul_1d3d.slang", 3),
            Self::MatMul_Tiled => slang!("matmul/matmul_tiled.slang", 3),
            Self::Gemm => slang!("gemm/gemm.slang", 4),
            Self::Gemm_Tiled => slang!("gemm/gemm_tiled.slang", 4),
        }
    }
}
