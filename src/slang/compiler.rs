use crate::instruction::GPUOperation;
use crate::slang::wrapper::{
    Blob, CompileTarget, CompilerOptions, FloatingPointMode, GlobalSession, OptimizationLevel,
    Session, SessionDesc, TargetDesc,
};
use crate::utils::dtype::onnx_dtype_to_slang_type;
use crate::utils::error::VKMLError;
use onnx_extractor::DataType;
use std::fmt::Write;
use std::mem::variant_count;
use std::sync::{LazyLock, Mutex, OnceLock};

const NUM_DTYPES: usize = variant_count::<DataType>();
const NUM_OPS: usize = variant_count::<GPUOperation>();

/// the session must be protected by a mutex because slangs ISession
/// internal dictionaries and reflection caches are not thread-safe
pub struct SlangContext {
    pub session: Mutex<Session>,
    pub blob_cache: [[OnceLock<Blob>; NUM_DTYPES]; NUM_OPS],
}

unsafe impl Send for SlangContext {}
unsafe impl Sync for SlangContext {}

pub static SLANG_CONTEXT: LazyLock<SlangContext> = LazyLock::new(|| {
    let global = GlobalSession::new().expect("Failed to initialise Slang GlobalSession");
    let profile = global.find_profile("spirv_1_6");

    // if maximal breaks something, can go back to high
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

    SlangContext {
        session: Mutex::new(session),
        blob_cache: [const { [const { OnceLock::new() }; NUM_DTYPES] }; NUM_OPS],
    }
});

pub fn compile(op: GPUOperation, dtype: DataType) -> Result<Blob, VKMLError> {
    let once_lock = &SLANG_CONTEXT.blob_cache[op as usize][dtype as usize];

    let blob = once_lock.get_or_try_init(|| -> Result<Blob, VKMLError> {
        let source_bytes = op.to_slang_shader()?;
        let source_string = std::str::from_utf8(source_bytes)
            .map_err(|e| VKMLError::Slang(format!("Shader source is not UTF-8: {}", e)))?;

        let dtype_str = onnx_dtype_to_slang_type(dtype);
        let module_name = format!("{}_{}", op.as_str(), dtype_str);

        let mut source = String::with_capacity(source_string.len() + 256);
        writeln!(source, "#define DTYPE {}", dtype_str).unwrap();
        source.push_str(source_string);

        let virtual_path = format!("{}.slang", module_name);

        let session = SLANG_CONTEXT.session.lock().unwrap();

        let module = session.load_module_from_source(&module_name, &virtual_path, &source)?;

        let entry_point = module.find_entry_point_by_name("main").ok_or_else(|| {
            VKMLError::Slang(format!(
                "Entry point 'void main()' not found in module {}",
                module_name
            ))
        })?;

        let components = [module.as_component_type(), entry_point.as_component_type()];

        let program = session.create_composite_component_type(&components)?;
        let linked = program.link()?;
        linked.entry_point_code(0, 0)
    })?;

    Ok(blob.clone())
}
