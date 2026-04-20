use crate::instruction::GPUOperation;
use crate::utils::dtype::onnx_dtype_to_slang_type;
use crate::utils::error::VKMLError;
use onnx_extractor::DataType;
use shader_slang::{
    Blob, CompileTarget, CompilerOptions, Downcast, FloatingPointMode, GlobalSession, IUnknown,
    Module, OptimizationLevel, Session, SessionDesc, TargetDesc,
};
use shader_slang_sys::{
    self, IBlobVtable, ISessionVtable, ISlangBlob, ISlangUnknown__bindgen_vtable,
};
use std::ffi::{CString, c_void};
use std::fmt::Write;
use std::mem::variant_count;

use std::ptr::NonNull;
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
        .floating_point_mode(FloatingPointMode::Fast);

    let targets = [TargetDesc::default()
        .format(CompileTarget::Spirv)
        .profile(profile)
        .options(&options)];

    // unsure if need to pass &options twice
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
        let session = SLANG_CONTEXT.session.lock().unwrap();

        let source_bytes = op.to_slang_shader()?;
        let source_string = std::str::from_utf8(source_bytes)
            .map_err(|e| VKMLError::Slang(format!("Shader source is not UTF-8: {}", e)))?;

        let dtype_str = onnx_dtype_to_slang_type(dtype);
        let module_name = format!("{}_{}", op.as_str(), dtype_str);

        let mut source = String::with_capacity(source_string.len() + 256);
        writeln!(source, "#define DTYPE {}", dtype_str).unwrap();
        source.push_str(source_string);

        let virtual_path = format!("{}.slang", module_name);

        let module = load_module_from_source(&session, &module_name, &virtual_path, &source)
            .map_err(VKMLError::Slang)?;

        let entry_point = module.find_entry_point_by_name("main").ok_or_else(|| {
            VKMLError::Slang(format!(
                "Entry point 'void main()' not found in module {}",
                module_name
            ))
        })?;

        let components = [module.downcast().clone(), entry_point.downcast().clone()];

        let program = session
            .create_composite_component_type(&components)
            .map_err(|e| VKMLError::Slang(e.to_string()))?;

        let linked = program
            .link()
            .map_err(|e| VKMLError::Slang(e.to_string()))?;

        linked
            .entry_point_code(0, 0)
            .map_err(|e| VKMLError::Slang(e.to_string()))
    })?;

    Ok(blob.clone())
}

/// Helper to perform a VTable call on a Slang object.
/// This mimics the internal behavior of the shader-slang crate.
unsafe fn vcall_release(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let vtable_ptr = *(ptr as *mut *mut ISlangUnknown__bindgen_vtable);
        ((*vtable_ptr).ISlangUnknown_release)(ptr as *mut _);
    }
}

pub fn load_module_from_source(
    session: &Session,
    module_name: &str,
    path: &str,
    source: &str,
) -> Result<Module, String> {
    unsafe {
        let vtable_ptr = std::mem::transmute_copy::<Session, *mut *const ISessionVtable>(session);
        let vtable = &**vtable_ptr;

        let module_name_cs = CString::new(module_name).unwrap();
        let path_cs = CString::new(path).unwrap();
        let source_cs = CString::new(source).unwrap();

        let mut diagnostics: *mut ISlangBlob = std::ptr::null_mut();
        let isession: *mut c_void = std::mem::transmute_copy(session);

        let module_ptr = (vtable.loadModuleFromSourceString)(
            isession,
            module_name_cs.as_ptr(),
            path_cs.as_ptr(),
            source_cs.as_ptr(),
            &mut diagnostics,
        );

        if module_ptr.is_null() {
            let mut err_msg = format!("Failed to compile Slang module: {}", module_name);
            if !diagnostics.is_null() {
                let diag_vtable = *(diagnostics as *mut *mut IBlobVtable);
                let buffer_ptr = ((*diag_vtable).getBufferPointer)(diagnostics as *mut _);
                let buffer_size = ((*diag_vtable).getBufferSize)(diagnostics as *mut _);

                if buffer_size > 0 && !buffer_ptr.is_null() {
                    let slice = std::slice::from_raw_parts(buffer_ptr as *const u8, buffer_size);
                    if let Ok(diag_str) = std::str::from_utf8(slice) {
                        err_msg = diag_str.to_string();
                    }
                }
                vcall_release(diagnostics as *mut _);
            }
            return Err(err_msg);
        }

        // diagnostics might contain warnings even on success
        vcall_release(diagnostics as *mut _);

        let module_unknown: IUnknown =
            std::mem::transmute(NonNull::new_unchecked(module_ptr as *mut c_void));
        Ok(std::mem::transmute::<IUnknown, Module>(module_unknown))
    }
}
