use onnx_extractor::DataType;

pub struct ShaderVariant {
    pub dtype: DataType,
    pub spv: &'static [u8],
}

pub struct Shader {
    pub variants: &'static [ShaderVariant],
}

#[macro_export]
macro_rules! shader {
    // Internal helper: embeds the compiled SPIR-V bytes
    (@var $name:literal, $dtype:ident) => {
        $crate::gpu::shader::ShaderVariant {
            dtype: onnx_extractor::DataType::$dtype,
            spv: include_bytes!(concat!(
                env!("OUT_DIR"),
                "/",
                $name,
                "_",
                stringify!($dtype),
                ".spv"
            )),
        }
    };

    // Standard IArithmetic types
    ($name:literal) => {
        $crate::shader!($name, [
            Float, Float16, Double,
            Int8, Uint8, Int16, Uint16,
            Int32, Uint32, Int64, Uint64
        ])
    };

    // Floating-point types
    ($name:literal, Float) => {
        $crate::shader!($name, [Float, Float16, Double])
    };

    // Explicit type list for custom subsets
    ($name:literal, [$($dtype:ident),+ $(,)?]) => {
        $crate::gpu::shader::Shader {
            variants: &[
                $(
                    $crate::shader!(@var $name, $dtype),
                )+
            ],
        }
    };
}
pub use shader;
