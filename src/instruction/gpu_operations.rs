use crate::{VKMLError, gpu::Gpu};
use onnx_extractor::DataType;

pub use crate::utils::dtype::{ARITHMETIC_TYPES, FLOAT_TYPES};

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
    pub name: &'static str,
    pub source: &'static str,
    pub bindings: usize,
    pub supported_types: &'static [DataType],
}

impl ShaderInfo {
    pub const fn arithmetic(name: &'static str, source: &'static str, bindings: usize) -> Self {
        Self {
            name,
            source,
            bindings,
            supported_types: ARITHMETIC_TYPES,
        }
    }

    pub const fn float(name: &'static str, source: &'static str, bindings: usize) -> Self {
        Self {
            name,
            source,
            bindings,
            supported_types: FLOAT_TYPES,
        }
    }

    pub const fn new(
        name: &'static str,
        source: &'static str,
        bindings: usize,
        supported_types: &'static [DataType],
    ) -> Self {
        Self {
            name,
            source,
            bindings,
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
    pub fn info(&self) -> ShaderInfo {
        match self {
            GpuShader::Addition => {
                ShaderInfo::arithmetic("addition", include_str!("add/add.slang"), 3)
            }
            GpuShader::Addition_NoStride => ShaderInfo::arithmetic(
                "addition_nostride",
                include_str!("add/add_nostride.slang"),
                3,
            ),
            GpuShader::Subtract => {
                ShaderInfo::arithmetic("subtract", include_str!("sub/sub.slang"), 3)
            }
            GpuShader::Multiply => {
                ShaderInfo::arithmetic("multiply", include_str!("mul/mul.slang"), 3)
            }
            GpuShader::Divide => ShaderInfo::arithmetic("divide", include_str!("div/div.slang"), 3),
            GpuShader::Maximum => {
                ShaderInfo::arithmetic("maximum", include_str!("max/max.slang"), 3)
            }
            GpuShader::Minimum => {
                ShaderInfo::arithmetic("minimum", include_str!("min/min.slang"), 3)
            }
            GpuShader::ReLU => ShaderInfo::arithmetic("relu", include_str!("relu/relu.slang"), 2),
            GpuShader::Sigmoid => {
                ShaderInfo::float("sigmoid", include_str!("sigmoid/sigmoid.slang"), 2)
            }
            GpuShader::Softmax => {
                ShaderInfo::float("softmax", include_str!("softmax/softmax.slang"), 2)
            }
            GpuShader::Expand => {
                ShaderInfo::arithmetic("expand", include_str!("expand/expand.slang"), 2)
            }
            GpuShader::ReduceMean => {
                ShaderInfo::arithmetic("reducemean", include_str!("reducemean/reducemean.slang"), 2)
            }
            GpuShader::Shape_Write => ShaderInfo::new(
                "shape_write",
                include_str!("shape/shape.slang"),
                1,
                &[DataType::Int64],
            ),
            GpuShader::MaxPool_1D => {
                ShaderInfo::arithmetic("maxpool_1d", include_str!("maxpool/maxpool_1d.slang"), 2)
            }
            GpuShader::MaxPool_2D => {
                ShaderInfo::arithmetic("maxpool_2d", include_str!("maxpool/maxpool_2d.slang"), 2)
            }
            GpuShader::MaxPool_3D => {
                ShaderInfo::arithmetic("maxpool_3d", include_str!("maxpool/maxpool_3d.slang"), 2)
            }
            GpuShader::Conv_1D => {
                ShaderInfo::arithmetic("conv_1d", include_str!("conv/conv_1d.slang"), 4)
            }
            GpuShader::Conv_2D => {
                ShaderInfo::arithmetic("conv_2d", include_str!("conv/conv_2d.slang"), 4)
            }
            GpuShader::Conv_3D => {
                ShaderInfo::arithmetic("conv_3d", include_str!("conv/conv_3d.slang"), 4)
            }
            GpuShader::MatMul_1D2D => {
                ShaderInfo::arithmetic("matmul_1d2d", include_str!("matmul/matmul_1d2d.slang"), 3)
            }
            GpuShader::MatMul_2D1D => {
                ShaderInfo::arithmetic("matmul_2d1d", include_str!("matmul/matmul_2d1d.slang"), 3)
            }
            GpuShader::MatMul_2D2D => {
                ShaderInfo::arithmetic("matmul_2d2d", include_str!("matmul/matmul_2d2d.slang"), 3)
            }
            GpuShader::MatMul_3D1D => {
                ShaderInfo::arithmetic("matmul_3d1d", include_str!("matmul/matmul_3d1d.slang"), 3)
            }
            GpuShader::MatMul_1D3D => {
                ShaderInfo::arithmetic("matmul_1d3d", include_str!("matmul/matmul_1d3d.slang"), 3)
            }
            GpuShader::MatMul_Tiled => {
                ShaderInfo::arithmetic("matmul_tiled", include_str!("matmul/matmul_tiled.slang"), 3)
            }
            GpuShader::Gemm => ShaderInfo::arithmetic("gemm", include_str!("gemm/gemm.slang"), 4),
            GpuShader::Gemm_Tiled => {
                ShaderInfo::arithmetic("gemm_tiled", include_str!("gemm/gemm_tiled.slang"), 4)
            }
        }
    }

    pub fn as_str(&self) -> &'static str {
        self.info().name
    }

    pub fn binding_count(&self) -> usize {
        self.info().bindings
    }

    pub fn to_slang_shader(self) -> Result<&'static str, VKMLError> {
        Ok(self.info().source)
    }

    pub fn is_generic(&self) -> bool {
        !matches!(self, GpuShader::Shape_Write)
    }

    pub fn min_shared_memory(&self) -> u32 {
        match self {
            GpuShader::MatMul_Tiled | GpuShader::Gemm_Tiled => 512,
            _ => 0,
        }
    }
}
