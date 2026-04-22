use crate::instruction::GPUOperation;
use crate::slang::wrapper::{
    Blob, CompileTarget, CompilerOptions, ComponentType, FloatingPointMode, GlobalSession, Module,
    OptimizationLevel, Session, SessionDesc, TargetDesc,
};
use crate::utils::dtype::onnx_dtype_to_slang_type;
use crate::utils::error::VKMLError;
use onnx_extractor::DataType;
use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

/// the session must be protected by a mutex because slangs ISession
/// internal dictionaries and reflection caches are not thread-safe
pub struct SlangContext {
    pub session: Session,
    pub module_cache: HashMap<GPUOperation, (Module, ComponentType)>,
    pub blob_cache: HashMap<(GPUOperation, DataType), Blob>,
}

pub static SLANG_CONTEXT: LazyLock<RwLock<SlangContext>> = LazyLock::new(|| {
    let global = GlobalSession::new().expect("Failed to initialise Slang GlobalSession");
    let profile = global.find_profile("spirv_1_6");

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
        .expect("Failed to create persistent Slang Session");

    RwLock::new(SlangContext {
        session,
        module_cache: HashMap::new(),
        blob_cache: HashMap::new(),
    })
});

pub fn compile(op: GPUOperation, dtype: DataType) -> Result<Blob, VKMLError> {
    let key = (op, dtype);

    // 1. Check if already compiled (Read Lock)
    {
        let ctx = SLANG_CONTEXT.read().unwrap();
        if let Some(blob) = ctx.blob_cache.get(&key) {
            return Ok(blob.clone());
        }
    }

    // 2. Not found, need to compile (Write Lock + Double-checked)
    let mut ctx = SLANG_CONTEXT.write().unwrap();
    if let Some(blob) = ctx.blob_cache.get(&key) {
        return Ok(blob.clone());
    }

    // 3. Get or compile module and its entry point
    let (module, entry_point) = if let Some(pair) = ctx.module_cache.get(&op) {
        pair.clone()
    } else {
        let module_name = op.as_str();
        let source_bytes = op.to_slang_shader()?;
        let source_string = std::str::from_utf8(source_bytes)
            .map_err(|e| VKMLError::Slang(format!("Shader source is not UTF-8: {e}")))?;

        let virtual_path = format!("{module_name}.slang");
        let module =
            ctx.session
                .load_module_from_source(module_name, &virtual_path, source_string)?;

        let entry_point = module.find_entry_point_by_name("main").ok_or_else(|| {
            VKMLError::Slang(format!(
                "Entry point 'main' not found in module {module_name}"
            ))
        })?;

        ctx.module_cache
            .insert(op, (module.clone(), entry_point.clone()));
        (module, entry_point)
    };

    // 4. Create composite program, specialize (if needed), and link
    let components = [module.as_component_type(), &entry_point];
    let program = ctx.session.create_composite_component_type(&components)?;

    let specialized_program = if op.is_fp_specialized() {
        program
    } else {
        let dtype_str = onnx_dtype_to_slang_type(dtype);
        program.specialize_with_type_name(0, dtype_str)?
    };

    let linked = specialized_program.link()?;
    let blob = linked.entry_point_code(0, 0)?;

    ctx.blob_cache.insert(key, blob.clone());
    Ok(blob)
}
