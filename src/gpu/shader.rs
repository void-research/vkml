use onnx_extractor::DataType;

pub struct ShaderVariant {
    pub dtype: DataType,
    pub spv: &'static [u8],
}

pub struct Shader {
    pub variants: &'static [ShaderVariant],
}

#[macro_export]
macro_rules! shader_variant {
    ($name:literal, $dtype:ident) => {
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
}

#[macro_export]
macro_rules! shader {
    // standard IArithmetic types
    ($name:literal, IArithmetic) => {
        $crate::gpu::shader::Shader {
            variants: &[
                $crate::shader_variant!($name, Float),
                $crate::shader_variant!($name, Float16),
                $crate::shader_variant!($name, Double),
                $crate::shader_variant!($name, Int8),
                $crate::shader_variant!($name, Uint8),
                $crate::shader_variant!($name, Int16),
                $crate::shader_variant!($name, Uint16),
                $crate::shader_variant!($name, Int32),
                $crate::shader_variant!($name, Uint32),
                $crate::shader_variant!($name, Int64),
                $crate::shader_variant!($name, Uint64),
            ],
        }
    };
    // default shorthand, delegates to IArithmetic
    ($name:literal) => {
        $crate::shader!($name, IArithmetic)
    };
    // explicit type list; [Float, Float16] or [Int64]
    ($name:literal, [$($dtype:ident),+ $(,)?]) => {
        $crate::gpu::shader::Shader {
            variants: &[
                $(
                    $crate::shader_variant!($name, $dtype),
                )+
            ],
        }
    };
}
pub use shader;
