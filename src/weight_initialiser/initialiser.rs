use bytemuck;
use onnx_extractor::Bytes;

#[derive(Default)]
pub enum Initialiser {
    #[default]
    None,
    Bytes(Bytes),
    VecBytes(Vec<Bytes>),
    BoxU8(Box<[u8]>),
    VecU8(Vec<u8>),
    VecF32(Vec<f32>),
    VecF64(Vec<f64>),
    VecI32(Vec<i32>),
    VecI64(Vec<i64>),
    VecU64(Vec<u64>),
    Constant(Vec<u8>),
    Xavier,
    Uniform(f32, f32),
    He,
}

impl Initialiser {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Initialiser::Bytes(bytes) => bytes.as_ref(),
            Initialiser::BoxU8(boxed) => boxed.as_ref(),
            Initialiser::VecU8(vec) => vec.as_ref(),
            Initialiser::VecF32(v) => bytemuck::cast_slice(v),
            Initialiser::VecF64(v) => bytemuck::cast_slice(v),
            Initialiser::VecI32(v) => bytemuck::cast_slice(v),
            Initialiser::VecI64(v) => bytemuck::cast_slice(v),
            Initialiser::VecU64(v) => bytemuck::cast_slice(v),
            Initialiser::Constant(vec) => vec.as_ref(),

            Initialiser::None => unimplemented!("None"),
            Initialiser::VecBytes(_) => unimplemented!("BytesVec"),
            Initialiser::Xavier => unimplemented!("Xavier"),
            Initialiser::Uniform(_, _) => unimplemented!("Uniform"),
            Initialiser::He => unimplemented!("He"),
        }
    }

    // consumes self
    pub fn into_cpu_buffer(self) -> Box<[u8]> {
        match self {
            Initialiser::Bytes(bytes) => bytes.to_vec().into(),
            Initialiser::VecBytes(parts) => {
                let total_len: usize = parts.iter().map(|b| b.len()).sum();
                let mut vec = Vec::with_capacity(total_len);
                for bytes in parts {
                    vec.extend_from_slice(&bytes);
                }
                vec.into()
            }
            Initialiser::BoxU8(boxed) => boxed,
            Initialiser::VecU8(vec) => vec.into(),
            Initialiser::VecF32(v) => bytemuck::cast_vec(v).into(),
            Initialiser::VecF64(v) => bytemuck::cast_vec(v).into(),
            Initialiser::VecI32(v) => bytemuck::cast_vec(v).into(),
            Initialiser::VecI64(v) => bytemuck::cast_vec(v).into(),
            Initialiser::VecU64(v) => bytemuck::cast_vec(v).into(),
            Initialiser::Constant(vec) => vec.into(),

            Initialiser::None => unimplemented!("None"),
            Initialiser::Xavier => unimplemented!("Xavier"),
            Initialiser::Uniform(_, _) => unimplemented!("Uniform"),
            Initialiser::He => unimplemented!("He"),
        }
    }
}
