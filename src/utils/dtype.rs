use std::ffi::CStr;

use onnx_extractor::DataType;
use vk::ComponentTypeKHR as Ct;
use vulkanalia::vk;

/// Returns None for unknown / unsupported mappings
pub fn vk_to_onnx_dtype(t: vk::ComponentTypeKHR) -> Option<DataType> {
    match t {
        Ct::SINT8 => Some(DataType::Int8),
        Ct::UINT8 => Some(DataType::Uint8),
        Ct::SINT16 => Some(DataType::Int16),
        Ct::UINT16 => Some(DataType::Uint16),
        Ct::SINT32 => Some(DataType::Int32),
        Ct::UINT32 => Some(DataType::Uint32),
        Ct::SINT64 => Some(DataType::Int64),
        Ct::UINT64 => Some(DataType::Uint64),
        Ct::FLOAT16 => Some(DataType::Float16),
        Ct::FLOAT32 => Some(DataType::Float),
        Ct::FLOAT64 => Some(DataType::Double),
        Ct::BFLOAT16 => Some(DataType::Bfloat16),
        Ct::FLOAT8_E4M3_EXT => Some(DataType::Float8e4m3fn),
        Ct::FLOAT8_E5M2_EXT => Some(DataType::Float8e5m2),
        _ => None,
    }
}

pub fn vk_bool32_to_bool(value: vk::Bool32) -> bool {
    value == vk::TRUE
}

pub fn bool_to_vk_bool32(value: bool) -> vk::Bool32 {
    if value { vk::TRUE } else { vk::FALSE }
}

/// Returns the Slang source code type string for a given ONNX DataType
pub fn onnx_dtype_to_slang_type(dtype: DataType) -> &'static CStr {
    match dtype {
        DataType::Float => c"float",
        DataType::Float16 => c"half",
        DataType::Double => c"double",
        DataType::Int8 => c"int8_t",
        DataType::Uint8 => c"uint8_t",
        DataType::Int16 => c"int16_t",
        DataType::Uint16 => c"uint16_t",
        DataType::Int32 => c"int",
        DataType::Uint32 => c"uint",
        DataType::Int64 => c"int64_t",
        DataType::Uint64 => c"uint64_t",
        DataType::Bool => c"bool",
        DataType::Bfloat16 => c"bfloat16_t",
        _ => unimplemented!(
            "Slang string mapping not implemented for ONNX datatype {:?}",
            dtype
        ),
    }
}

/// All DataTypes supported by Slang's `IArithmetic` interface.
/// (Excludes Bool, Bfloat16, Float8e4m3fn, Float8e5m2).
pub const ARITHMETIC_TYPES: &[DataType] = &[
    DataType::Float,
    DataType::Float16,
    DataType::Double,
    DataType::Int8,
    DataType::Uint8,
    DataType::Int16,
    DataType::Uint16,
    DataType::Int32,
    DataType::Uint32,
    DataType::Int64,
    DataType::Uint64,
];

/// All DataTypes supported by Slang's `__BuiltinFloatingPointType` interface.
pub const FLOAT_TYPES: &[DataType] = &[DataType::Float, DataType::Float16, DataType::Double];
