pub mod compiler;
pub mod wrapper;

use onnx_extractor::DataType;
use std::ffi::CStr;

pub use compiler::SlangCompiler;

pub struct Shader {
    pub path: &'static CStr,
    pub source: &'static CStr,
    pub binding_count: usize,
    pub supported_types: &'static [DataType],
}

#[macro_export]
macro_rules! slang {
    ($path:literal, $bindings:expr) => {
        $crate::slang!($path, $bindings, $crate::instruction::ARITHMETIC_TYPES)
    };
    ($path:literal, $bindings:expr, $types:expr) => {
        $crate::slang::Shader {
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
pub use slang;
