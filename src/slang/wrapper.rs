use crate::utils::error::VKMLError;
use shader_slang_sys::{
    IBlobVtable, IComponentTypeVtable, IGlobalSessionVtable, IModuleVtable, ISessionVtable,
    ISlangBlob, ISlangUnknown, ISlangUnknown__bindgen_vtable, SlangCompileTarget,
    SlangFloatingPointMode, SlangInt, SlangOptimizationLevel, SlangProfileID, SlangResult,
    slang_CompilerOptionEntry, slang_CompilerOptionName, slang_CompilerOptionValue,
    slang_CompilerOptionValueKind, slang_IComponentType, slang_SessionDesc,
    slang_SpecializationArg, slang_SpecializationArg__bindgen_ty_1, slang_SpecializationArg_Kind,
    slang_TargetDesc,
};
use std::ffi::{CStr, c_void};
use std::ptr::{NonNull, null_mut};

/// COM ref-counted pointer. Clone calls addRef, Drop calls release.
struct ComPtr(NonNull<c_void>);

impl ComPtr {
    unsafe fn new(ptr: *mut c_void) -> Self {
        Self(NonNull::new(ptr).expect("Slang returned null pointer"))
    }

    unsafe fn from_borrowed(ptr: *mut c_void) -> Self {
        let this = unsafe { Self::new(ptr) };
        unsafe {
            let vt = this.vt::<ISlangUnknown__bindgen_vtable>();
            (vt.ISlangUnknown_addRef)(this.as_ptr() as *mut ISlangUnknown);
        }
        this
    }

    fn as_ptr(&self) -> *mut c_void {
        self.0.as_ptr()
    }

    unsafe fn vt<V>(&self) -> &V {
        unsafe { &**(self.as_ptr() as *mut *mut V) }
    }
}

impl Clone for ComPtr {
    fn clone(&self) -> Self {
        unsafe {
            let vt = self.vt::<ISlangUnknown__bindgen_vtable>();
            (vt.ISlangUnknown_addRef)(self.as_ptr() as *mut ISlangUnknown);
        }
        Self(self.0)
    }
}

impl Drop for ComPtr {
    fn drop(&mut self) {
        unsafe {
            let vt = self.vt::<ISlangUnknown__bindgen_vtable>();
            (vt.ISlangUnknown_release)(self.as_ptr() as *mut ISlangUnknown);
        }
    }
}

unsafe impl Send for ComPtr {}
unsafe impl Sync for ComPtr {}

/// Extracts diagnostic text from a Slang blob, automatically releasing the blob.
unsafe fn extract_diag(diag: *mut ISlangBlob) -> Option<String> {
    let ptr = NonNull::new(diag as *mut c_void)?;
    let blob = ComPtr(ptr); // auto-releases on drop
    let vt = unsafe { blob.vt::<IBlobVtable>() };
    let (buf, len) = unsafe {
        (
            (vt.getBufferPointer)(blob.as_ptr()),
            (vt.getBufferSize)(blob.as_ptr()),
        )
    };
    if len > 0 && !buf.is_null() {
        let slice = unsafe { std::slice::from_raw_parts(buf as *const u8, len) };
        let s = String::from_utf8_lossy(slice);
        let trimmed = s.trim_end_matches('\0').trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    } else {
        None
    }
}

unsafe fn check(hr: SlangResult, diag: *mut ISlangBlob) -> Result<(), VKMLError> {
    let msg = unsafe { extract_diag(diag) };
    if hr >= 0 {
        Ok(())
    } else {
        Err(VKMLError::Slang(
            msg.unwrap_or_else(|| format!("Slang error code: {hr}")),
        ))
    }
}

/// Helper for Slang COM calls that return a SlangResult, an output pointer, and a diagnostic blob.
unsafe fn call_diag<T>(
    f: impl FnOnce(*mut *mut T, *mut *mut ISlangBlob) -> SlangResult,
) -> Result<ComPtr, VKMLError> {
    let (mut out, mut diag) = (null_mut(), null_mut());
    unsafe {
        check(f(&mut out, &mut diag), diag)?;
        Ok(ComPtr::new(out as *mut c_void))
    }
}

#[derive(Clone)]
pub struct Blob(ComPtr);

impl Blob {
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            let vt = self.0.vt::<IBlobVtable>();
            let ptr = (vt.getBufferPointer)(self.0.as_ptr());
            let len = (vt.getBufferSize)(self.0.as_ptr());
            std::slice::from_raw_parts(ptr as *const u8, len)
        }
    }
}

impl std::ops::Deref for Blob {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

pub struct GlobalSession(ComPtr);

impl GlobalSession {
    pub fn new() -> Option<Self> {
        let mut ptr = null_mut();
        unsafe {
            shader_slang_sys::slang_createGlobalSession(
                shader_slang_sys::SLANG_API_VERSION as _,
                &mut ptr,
            );
        }
        NonNull::new(ptr as *mut c_void).map(|p| Self(ComPtr(p)))
    }

    pub fn find_profile(&self, name: &CStr) -> SlangProfileID {
        let vt = unsafe { self.0.vt::<IGlobalSessionVtable>() };
        unsafe { (vt.findProfile)(self.0.as_ptr(), name.as_ptr()) }
    }

    pub fn create_session(
        &self,
        targets: &[TargetDesc],
        options: &CompilerOptions,
    ) -> Option<Session> {
        let desc = slang_SessionDesc {
            structureSize: std::mem::size_of::<slang_SessionDesc>(),
            targets: targets.as_ptr() as *const slang_TargetDesc,
            targetCount: targets.len() as SlangInt,
            compilerOptionEntries: options.as_ptr(),
            compilerOptionEntryCount: options.len(),
            ..unsafe { std::mem::zeroed() }
        };
        let mut ptr = null_mut();
        let vt = unsafe { self.0.vt::<IGlobalSessionVtable>() };
        let hr = unsafe { (vt.createSession)(self.0.as_ptr(), &desc, &mut ptr) };
        (hr >= 0 && !ptr.is_null()).then(|| unsafe { Session(ComPtr::new(ptr as *mut c_void)) })
    }
}

pub struct Session(ComPtr);

impl Session {
    pub fn load_module_from_source(&self, path: &CStr, source: &CStr) -> Result<Module, VKMLError> {
        let vt = unsafe { self.0.vt::<ISessionVtable>() };
        let mut diag = null_mut();
        let ptr = unsafe {
            (vt.loadModuleFromSourceString)(
                self.0.as_ptr(),
                path.as_ptr(),
                path.as_ptr(),
                source.as_ptr(),
                &mut diag,
            )
        };
        let diag_msg = unsafe { extract_diag(diag) };
        if ptr.is_null() {
            return Err(VKMLError::Slang(
                diag_msg.unwrap_or_else(|| format!("Failed to compile module: {path:?}")),
            ));
        }
        unsafe { Ok(Module(ComPtr::from_borrowed(ptr as *mut c_void))) }
    }

    pub fn create_composite_component_type(
        &self,
        components: &[&ComponentType],
    ) -> Result<ComponentType, VKMLError> {
        let ptrs: Vec<*const slang_IComponentType> = components
            .iter()
            .map(|c| c.0.as_ptr() as *const slang_IComponentType)
            .collect();
        let vt = unsafe { self.0.vt::<ISessionVtable>() };
        unsafe {
            call_diag(|out, diag| {
                (vt.createCompositeComponentType)(
                    self.0.as_ptr(),
                    ptrs.as_ptr(),
                    ptrs.len() as SlangInt,
                    out,
                    diag,
                )
            })
            .map(ComponentType)
        }
    }
}

#[derive(Clone)]
pub struct Module(ComPtr);

impl Module {
    pub fn find_entry_point_by_name(&self, name: &CStr) -> Option<ComponentType> {
        let vt = unsafe { self.0.vt::<IModuleVtable>() };
        let mut ptr = null_mut();
        let hr = unsafe { (vt.findEntryPointByName)(self.0.as_ptr(), name.as_ptr(), &mut ptr) };
        (hr >= 0 && !ptr.is_null())
            .then(|| unsafe { ComponentType(ComPtr::new(ptr as *mut c_void)) })
    }
}

impl std::ops::Deref for Module {
    type Target = ComponentType;
    fn deref(&self) -> &Self::Target {
        unsafe { std::mem::transmute(self) }
    }
}

#[derive(Clone)]
pub struct ComponentType(ComPtr);

impl ComponentType {
    pub fn specialize_with_type_name(
        &self,
        target_index: i64,
        type_name: &CStr,
    ) -> Result<ComponentType, VKMLError> {
        let vt = unsafe { self.0.vt::<IComponentTypeVtable>() };
        let mut layout_diag = null_mut();
        let layout = unsafe { (vt.getLayout)(self.0.as_ptr(), target_index, &mut layout_diag) };
        let diag_msg = unsafe { extract_diag(layout_diag) };
        if layout.is_null() {
            return Err(VKMLError::Slang(diag_msg.unwrap_or_else(|| {
                format!("Failed to get Slang layout for type '{type_name:?}'")
            })));
        }

        let type_reflection = unsafe {
            shader_slang_sys::spReflection_FindTypeByName(
                layout as *mut shader_slang_sys::SlangReflection,
                type_name.as_ptr(),
            )
        };
        if type_reflection.is_null() {
            return Err(VKMLError::Slang(format!(
                "Type '{type_name:?}' not found in Slang reflection layout"
            )));
        }

        let arg = slang_SpecializationArg {
            kind: slang_SpecializationArg_Kind::Type,
            __bindgen_anon_1: slang_SpecializationArg__bindgen_ty_1 {
                type_: type_reflection as *mut shader_slang_sys::slang_TypeReflection,
            },
        };

        unsafe {
            call_diag(|out, diag| (vt.specialize)(self.0.as_ptr(), &arg, 1, out, diag))
                .map(ComponentType)
        }
    }

    pub fn link(&self) -> Result<ComponentType, VKMLError> {
        let vt = unsafe { self.0.vt::<IComponentTypeVtable>() };
        unsafe { call_diag(|out, diag| (vt.link)(self.0.as_ptr(), out, diag)).map(ComponentType) }
    }

    pub fn entry_point_code(&self, entry_index: i64, target_index: i64) -> Result<Blob, VKMLError> {
        let vt = unsafe { self.0.vt::<IComponentTypeVtable>() };
        unsafe {
            call_diag(|code, diag| {
                (vt.getEntryPointCode)(self.0.as_ptr(), entry_index, target_index, code, diag)
            })
            .map(Blob)
        }
    }
}

#[derive(Default)]
pub struct CompilerOptions {
    entries: Vec<slang_CompilerOptionEntry>,
}

macro_rules! int_option {
    ($name:ident, $func:ident, $param_ty:ty) => {
        pub fn $func(self, value: $param_ty) -> Self {
            self.push_int(slang_CompilerOptionName::$name, value as i32)
        }
    };
}

impl CompilerOptions {
    fn push_int(mut self, name: slang_CompilerOptionName, int_value0: i32) -> Self {
        self.entries.push(slang_CompilerOptionEntry {
            name,
            value: slang_CompilerOptionValue {
                kind: slang_CompilerOptionValueKind::Int,
                intValue0: int_value0,
                ..unsafe { std::mem::zeroed() }
            },
        });
        self
    }

    pub fn as_ptr(&self) -> *const slang_CompilerOptionEntry {
        self.entries.as_ptr()
    }

    pub fn len(&self) -> u32 {
        self.entries.len() as u32
    }

    int_option!(MatrixLayoutRow, matrix_layout_row, bool);
    int_option!(Optimization, optimization, SlangOptimizationLevel);
    int_option!(
        FloatingPointMode,
        floating_point_mode,
        SlangFloatingPointMode
    );
    int_option!(EmitSpirvDirectly, emit_spirv_directly, bool);
    int_option!(SkipSPIRVValidation, skip_spirv_validation, bool);
    int_option!(GLSLForceScalarLayout, glsl_force_scalar_layout, bool);
}

#[repr(transparent)]
pub struct TargetDesc {
    pub(crate) inner: slang_TargetDesc,
}

impl Default for TargetDesc {
    fn default() -> Self {
        Self {
            inner: slang_TargetDesc {
                structureSize: std::mem::size_of::<slang_TargetDesc>(),
                ..unsafe { std::mem::zeroed() }
            },
        }
    }
}

impl TargetDesc {
    pub fn format(mut self, format: SlangCompileTarget) -> Self {
        self.inner.format = format;
        self
    }

    pub fn profile(mut self, profile: SlangProfileID) -> Self {
        self.inner.profile = profile;
        self
    }

    pub fn options(mut self, options: &CompilerOptions) -> Self {
        self.inner.compilerOptionEntries = options.as_ptr();
        self.inner.compilerOptionEntryCount = options.len();
        self
    }
}
