#[derive(Debug)]
pub struct Optimisations {
    pub tf32_matmul: bool,
    pub conv_gemm: bool,
}

impl Default for Optimisations {
    fn default() -> Self {
        Self {
            tf32_matmul: true,
            conv_gemm: true,
        }
    }
}

impl Optimisations {
    pub fn all_off() -> Self {
        Self {
            tf32_matmul: false,
            conv_gemm: false,
        }
    }
}
