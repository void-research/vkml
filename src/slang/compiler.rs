use crate::instruction::GpuShader;
use crate::slang::wrapper::{
    Blob, CompileTarget, CompilerOptions, ComponentType, FloatingPointMode, GlobalSession,
    OptimizationLevel, Session, SessionDesc, TargetDesc,
};
use crate::utils::dtype::onnx_dtype_to_slang_type;
use crate::utils::error::VKMLError;
use onnx_extractor::DataType;
use std::collections::HashMap;
use std::sync::RwLock;

struct SlangInner {
    session: Session,
    module_cache: HashMap<GpuShader, ComponentType>,
    blob_cache: HashMap<(GpuShader, DataType), Blob>,
}

impl SlangInner {
    fn compile(&mut self, op: GpuShader, dtype: DataType) -> Result<Blob, VKMLError> {
        let key = (op, dtype);

        if let Some(blob) = self.blob_cache.get(&key) {
            return Ok(blob.clone());
        }

        let program = match self.module_cache.get(&op) {
            Some(program) => program.clone(),
            None => {
                let module_name = op.as_str();
                let source_string = op.to_slang_shader()?;

                let virtual_path = format!("{module_name}.slang");
                let module = self.session.load_module_from_source(
                    module_name,
                    &virtual_path,
                    source_string,
                )?;

                let entry_point = module.find_entry_point_by_name(c"main").ok_or_else(|| {
                    VKMLError::Slang(format!(
                        "Entry point 'main' not found in module {module_name}"
                    ))
                })?;

                let program = self
                    .session
                    .create_composite_component_type(&[module.as_component_type(), &entry_point])?;

                self.module_cache.insert(op, program.clone());
                program
            }
        };

        let specialized_program = if op.is_generic() {
            let dtype_cstr = onnx_dtype_to_slang_type(dtype);
            program.specialize_with_type_name(0, dtype_cstr)?
        } else {
            program
        };

        let linked_program = specialized_program.link()?;
        let compiled_blob = linked_program.entry_point_code(0, 0)?;

        self.blob_cache.insert(key, compiled_blob.clone());
        Ok(compiled_blob)
    }
}

/// Thread-safe Slang compiler session and caches.
/// Internal synchronization protects Slang's ISession dictionaries and reflection caches.
pub struct SlangCompiler {
    inner: RwLock<SlangInner>,
}

impl SlangCompiler {
    pub fn new() -> Result<Self, VKMLError> {
        let global = GlobalSession::new()
            .ok_or_else(|| VKMLError::Slang("Failed to initialise Slang GlobalSession".into()))?;
        let profile = global.find_profile(c"spirv_1_6");

        let options = CompilerOptions::default()
            .matrix_layout_row(true)
            .optimization(OptimizationLevel::Maximal)
            .floating_point_mode(FloatingPointMode::Fast)
            .emit_spirv_directly(true)
            .skip_spirv_validation(true)
            .glsl_force_scalar_layout(true);

        let targets = [TargetDesc::default()
            .format(CompileTarget::Spirv)
            .profile(profile)
            .options(&options)];

        let session_desc = SessionDesc::default().targets(&targets).options(&options);

        let session = global
            .create_session(&session_desc)
            .ok_or_else(|| VKMLError::Slang("Failed to create persistent Slang Session".into()))?;

        Ok(Self {
            inner: RwLock::new(SlangInner {
                session,
                module_cache: HashMap::new(),
                blob_cache: HashMap::new(),
            }),
        })
    }

    /// Compiles a GpuShader and DataType to SPIR-V blob.
    /// Thread-safe, checks read lock before upgrading to write lock on cache miss.
    pub fn compile(&self, op: GpuShader, dtype: DataType) -> Result<Blob, VKMLError> {
        let key = (op, dtype);

        // 1. Fast read lock check
        {
            let guard = self.inner.read().unwrap();
            if let Some(blob) = guard.blob_cache.get(&key) {
                return Ok(blob.clone());
            }
        }

        let mut guard = self.inner.write().unwrap();
        guard.compile(op, dtype)
    }
}
