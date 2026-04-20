use crate::utils::error::VKMLError;
use shader_slang_sys::{
    IBlobVtable, IComponentTypeVtable, IGlobalSessionVtable, IModuleVtable, ISessionVtable,
    ISlangBlob, ISlangUnknown, ISlangUnknown__bindgen_vtable, SlangInt, SlangResult,
    slang_CompilerOptionEntry, slang_CompilerOptionName, slang_CompilerOptionValue,
    slang_CompilerOptionValueKind, slang_IComponentType, slang_IEntryPoint, slang_SessionDesc,
    slang_TargetDesc,
};
use std::ffi::{CString, c_void};
use std::ptr::{NonNull, null, null_mut};

/// COM ref-counted pointer. Clone calls addRef, Drop calls release.
struct ComPtr(NonNull<c_void>);

impl ComPtr {
    unsafe fn from_owned(ptr: *mut c_void) -> Self {
        Self(NonNull::new(ptr).expect("Slang returned null COM pointer"))
    }

    /// Calls addRef on a session-owned (borrowed) pointer to take our own reference
    unsafe fn from_borrowed(ptr: *mut c_void) -> Self {
        unsafe {
            let this = Self::from_owned(ptr);
            this.add_ref();
            this
        }
    }

    fn as_ptr(&self) -> *mut c_void {
        self.0.as_ptr()
    }

    unsafe fn vtable<V>(&self) -> &V {
        unsafe { &**(self.as_ptr() as *mut *mut V) }
    }

    fn add_ref(&self) {
        unsafe {
            let vt = self.vtable::<ISlangUnknown__bindgen_vtable>();
            (vt.ISlangUnknown_addRef)(self.as_ptr() as *mut ISlangUnknown);
        }
    }
}

impl Clone for ComPtr {
    fn clone(&self) -> Self {
        self.add_ref();
        Self(self.0)
    }
}

impl Drop for ComPtr {
    fn drop(&mut self) {
        unsafe {
            let vt = self.vtable::<ISlangUnknown__bindgen_vtable>();
            (vt.ISlangUnknown_release)(self.as_ptr() as *mut ISlangUnknown);
        }
    }
}

unsafe impl Send for ComPtr {}
unsafe impl Sync for ComPtr {}

unsafe fn extract_diagnostics(diag: *mut ISlangBlob) -> Option<String> {
    if diag.is_null() {
        return None;
    }
    unsafe {
        let vt = &**(diag as *mut *mut IBlobVtable);
        let ptr = (vt.getBufferPointer)(diag as *mut _);
        let len = (vt.getBufferSize)(diag as *mut _);
        let msg = if len > 0 && !ptr.is_null() {
            let slice = std::slice::from_raw_parts(ptr as *const u8, len);
            std::str::from_utf8(slice).ok().map(|s| s.to_owned())
        } else {
            None
        };
        let unknown_vt = &**(diag as *mut *mut ISlangUnknown__bindgen_vtable);
        (unknown_vt.ISlangUnknown_release)(diag as *mut _);
        msg
    }
}

unsafe fn check(hr: SlangResult, diag: *mut ISlangBlob) -> Result<(), VKMLError> {
    unsafe {
        if hr >= 0 {
            // success — still release diagnostics (may contain warnings)
            if !diag.is_null() {
                let unknown_vt = &**(diag as *mut *mut ISlangUnknown__bindgen_vtable);
                (unknown_vt.ISlangUnknown_release)(diag as *mut _);
            }
            Ok(())
        } else {
            let msg =
                extract_diagnostics(diag).unwrap_or_else(|| format!("Slang error code: {hr}"));
            Err(VKMLError::Slang(msg))
        }
    }
}

#[derive(Clone)]
pub struct Blob(ComPtr);

impl Blob {
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            let vt = self.0.vtable::<IBlobVtable>();
            let ptr = (vt.getBufferPointer)(self.0.as_ptr());
            let len = (vt.getBufferSize)(self.0.as_ptr());
            std::slice::from_raw_parts(ptr as *const u8, len)
        }
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
        NonNull::new(ptr as *mut c_void).map(|nn| Self(ComPtr(nn)))
    }

    pub fn find_profile(&self, name: &str) -> ProfileID {
        let name_cs = CString::new(name).unwrap();
        unsafe {
            let vt = self.0.vtable::<IGlobalSessionVtable>();
            ProfileID((vt.findProfile)(self.0.as_ptr(), name_cs.as_ptr()))
        }
    }

    pub fn create_session(&self, desc: &SessionDesc) -> Option<Session> {
        let mut ptr = null_mut();
        unsafe {
            let vt = self.0.vtable::<IGlobalSessionVtable>();
            (vt.createSession)(self.0.as_ptr(), &desc.inner, &mut ptr);
        }
        NonNull::new(ptr as *mut c_void).map(|nn| Session(ComPtr(nn)))
    }
}

#[derive(Clone, Copy)]
pub struct ProfileID(pub shader_slang_sys::SlangProfileID);

pub struct Session(ComPtr);

impl Session {
    pub fn load_module_from_source(
        &self,
        module_name: &str,
        path: &str,
        source: &str,
    ) -> Result<Module, VKMLError> {
        let name_cs = CString::new(module_name).unwrap();
        let path_cs = CString::new(path).unwrap();
        let source_cs = CString::new(source).unwrap();

        unsafe {
            let vt = self.0.vtable::<ISessionVtable>();
            let mut diag: *mut ISlangBlob = null_mut();

            let module_ptr = (vt.loadModuleFromSourceString)(
                self.0.as_ptr(),
                name_cs.as_ptr(),
                path_cs.as_ptr(),
                source_cs.as_ptr(),
                &mut diag,
            );

            if module_ptr.is_null() {
                let msg = extract_diagnostics(diag)
                    .unwrap_or_else(|| format!("Failed to compile Slang module: {module_name}"));
                return Err(VKMLError::Slang(msg));
            }

            if !diag.is_null() {
                let unknown_vt = &**(diag as *mut *mut ISlangUnknown__bindgen_vtable);
                (unknown_vt.ISlangUnknown_release)(diag as *mut _);
            }

            // loadModuleFromSourceString returns a session-owned (borrowed) reference
            Ok(Module(ComPtr::from_borrowed(module_ptr as *mut c_void)))
        }
    }

    pub fn create_composite_component_type(
        &self,
        components: &[&ComponentType],
    ) -> Result<ComponentType, VKMLError> {
        let ptrs: Vec<*const slang_IComponentType> = components
            .iter()
            .map(|c| c.0.as_ptr() as *const _)
            .collect();

        unsafe {
            let vt = self.0.vtable::<ISessionVtable>();
            let mut out = null_mut();
            let mut diag = null_mut();

            let hr = (vt.createCompositeComponentType)(
                self.0.as_ptr(),
                ptrs.as_ptr(),
                ptrs.len() as SlangInt,
                &mut out,
                &mut diag,
            );
            check(hr, diag)?;
            Ok(ComponentType(ComPtr::from_owned(out as *mut c_void)))
        }
    }
}

pub struct Module(ComPtr);

impl Module {
    pub fn find_entry_point_by_name(&self, name: &str) -> Option<EntryPoint> {
        let name_cs = CString::new(name).unwrap();
        unsafe {
            let vt = self.0.vtable::<IModuleVtable>();
            let mut ep: *mut slang_IEntryPoint = null_mut();
            (vt.findEntryPointByName)(self.0.as_ptr(), name_cs.as_ptr(), &mut ep);
            NonNull::new(ep as *mut c_void).map(|nn| EntryPoint(ComPtr(nn)))
        }
    }

    /// Module inherits IComponentType, same COM pointer
    pub fn as_component_type(&self) -> &ComponentType {
        unsafe { &*(self as *const Module as *const ComponentType) }
    }
}

#[allow(dead_code)] // field used structurally for Drop ref-count release
pub struct EntryPoint(ComPtr);

impl EntryPoint {
    /// EntryPoint inherits IComponentType, same COM pointer
    pub fn as_component_type(&self) -> &ComponentType {
        unsafe { &*(self as *const EntryPoint as *const ComponentType) }
    }
}

pub struct ComponentType(ComPtr);

impl ComponentType {
    pub fn link(&self) -> Result<ComponentType, VKMLError> {
        unsafe {
            let vt = self.0.vtable::<IComponentTypeVtable>();
            let mut out = null_mut();
            let mut diag = null_mut();
            let hr = (vt.link)(self.0.as_ptr(), &mut out, &mut diag);
            check(hr, diag)?;
            Ok(ComponentType(ComPtr::from_owned(out as *mut c_void)))
        }
    }

    pub fn entry_point_code(&self, entry_index: i64, target_index: i64) -> Result<Blob, VKMLError> {
        unsafe {
            let vt = self.0.vtable::<IComponentTypeVtable>();
            let mut code = null_mut();
            let mut diag = null_mut();
            let hr = (vt.getEntryPointCode)(
                self.0.as_ptr(),
                entry_index,
                target_index,
                &mut code,
                &mut diag,
            );
            check(hr, diag)?;
            Ok(Blob(ComPtr::from_owned(code as *mut c_void)))
        }
    }
}

pub use shader_slang_sys::SlangCompileTarget as CompileTarget;
pub use shader_slang_sys::SlangFloatingPointMode as FloatingPointMode;
pub use shader_slang_sys::SlangOptimizationLevel as OptimizationLevel;

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
    fn push_int(mut self, name: slang_CompilerOptionName, v0: i32) -> Self {
        self.entries.push(slang_CompilerOptionEntry {
            name,
            value: slang_CompilerOptionValue {
                kind: slang_CompilerOptionValueKind::Int,
                intValue0: v0,
                intValue1: 0,
                stringValue0: null(),
                stringValue1: null(),
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
    int_option!(Optimization, optimization, OptimizationLevel);
    int_option!(FloatingPointMode, floating_point_mode, FloatingPointMode);
    int_option!(EmitSpirvDirectly, emit_spirv_directly, bool);
    int_option!(SkipSPIRVValidation, skip_spirv_validation, bool);
    int_option!(GLSLForceScalarLayout, glsl_force_scalar_layout, bool);
}

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
    pub fn format(mut self, format: CompileTarget) -> Self {
        self.inner.format = format;
        self
    }

    pub fn profile(mut self, profile: ProfileID) -> Self {
        self.inner.profile = profile.0;
        self
    }

    pub fn options(mut self, opts: &CompilerOptions) -> Self {
        self.inner.compilerOptionEntries = opts.as_ptr();
        self.inner.compilerOptionEntryCount = opts.len();
        self
    }
}

pub struct SessionDesc {
    pub(crate) inner: slang_SessionDesc,
}

impl Default for SessionDesc {
    fn default() -> Self {
        Self {
            inner: slang_SessionDesc {
                structureSize: std::mem::size_of::<slang_SessionDesc>(),
                ..unsafe { std::mem::zeroed() }
            },
        }
    }
}

impl SessionDesc {
    pub fn targets(mut self, targets: &[TargetDesc]) -> Self {
        self.inner.targets = targets.as_ptr() as *const slang_TargetDesc;
        self.inner.targetCount = targets.len() as SlangInt;
        self
    }

    pub fn options(mut self, opts: &CompilerOptions) -> Self {
        self.inner.compilerOptionEntries = opts.as_ptr();
        self.inner.compilerOptionEntryCount = opts.len();
        self
    }
}
