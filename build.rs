use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const TYPES: &[(&str, &str)] = &[
    ("Float", "float"),
    ("Float16", "half"),
    ("Double", "double"),
    ("Int8", "int8_t"),
    ("Uint8", "uint8_t"),
    ("Int16", "int16_t"),
    ("Uint16", "uint16_t"),
    ("Int32", "int"),
    ("Uint32", "uint"),
    ("Int64", "int64_t"),
    ("Uint64", "uint64_t"),
];

fn find_slang_files(dir: &Path, files: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                find_slang_files(&path, files);
            } else if path.extension().is_some_and(|ext| ext == "slang") {
                files.push(path);
            }
        }
    }
}

fn slangc_path() -> String {
    if let Ok(dir) = env::var("SLANG_DIR") {
        let p = Path::new(&dir).join("bin/slangc");
        if p.exists() {
            return p.to_string_lossy().into_owned();
        }
    }
    let opt = Path::new("/opt/shader-slang-bin/bin/slangc");
    if opt.exists() {
        return opt.to_string_lossy().into_owned();
    }
    "slangc".to_string()
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SLANG_DIR");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let slangc = slangc_path();

    let mut slang_files = Vec::new();
    find_slang_files(Path::new("src/instruction"), &mut slang_files);

    for path in &slang_files {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    for path in slang_files {
        let name = path.file_stem().unwrap().to_str().unwrap();
        let stamp_file = out_dir.join(format!("{name}.stamp"));

        // Skip compilation if the shader has already been compiled and is newer than the source
        if stamp_file.exists()
            && let (Ok(stamp_meta), Ok(src_meta)) = (stamp_file.metadata(), path.metadata())
            && let (Ok(stamp_mtime), Ok(src_mtime)) = (stamp_meta.modified(), src_meta.modified())
            && stamp_mtime >= src_mtime
        {
            continue;
        }

        let mut compiled_any = false;
        let mut last_error = String::new();

        for (dtype_name, slang_type) in TYPES {
            let out_file = out_dir.join(format!("{name}_{dtype_name}.spv"));

            let output = Command::new(&slangc)
                .arg(&path)
                .args([
                    "-target",
                    "spirv",
                    "-entry",
                    "main",
                    "-profile",
                    "spirv_1_6",
                ])
                .args(["-specialize", slang_type])
                .args(["-matrix-layout-row-major", "-O3", "-fp-mode", "fast"])
                .args(["-skip-spirv-validation", "-force-glsl-scalar-layout"])
                .arg("-o")
                .arg(&out_file)
                .output()
                .unwrap_or_else(|e| panic!("Failed to execute slangc: {e}"));

            if output.status.success() {
                compiled_any = true;
            } else {
                last_error = String::from_utf8_lossy(&output.stderr).into_owned();
            }
        }

        if !compiled_any {
            panic!("Shader '{name}' failed to compile for any type:\n{last_error}");
        }

        // Record stamp to mark this shader as successfully compiled
        let _ = fs::write(&stamp_file, b"ok");
    }
}
