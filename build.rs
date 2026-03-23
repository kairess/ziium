use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "linux" {
        return;
    }

    let python = if PathBuf::from(".venv/bin/python").exists() {
        ".venv/bin/python"
    } else {
        "python3"
    };

    let output = Command::new(python)
        .arg("-c")
        .arg(
            "import pathlib, torch\n\
root = pathlib.Path(torch.__file__).resolve().parent\n\
print(root / 'lib')",
        )
        .output();
    let Ok(output) = output else {
        return;
    };
    if !output.status.success() {
        return;
    }
    let Ok(libtorch_lib) = String::from_utf8(output.stdout) else {
        return;
    };
    let libtorch_lib = PathBuf::from(libtorch_lib.trim());
    if !libtorch_lib.exists() {
        return;
    }

    println!("cargo:rustc-link-search=native={}", libtorch_lib.display());
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");

    for lib in ["torch_cuda", "torch_cuda_linalg", "c10_cuda"] {
        let filename = format!("lib{lib}.so");
        let full_path = libtorch_lib.join(filename);
        if full_path.exists() {
            println!("cargo:rustc-link-arg={}", full_path.display());
        }
    }

    // PyTorch wheel layout: torch/lib next to nvidia/*/lib directories.
    if let Some(site_packages) = libtorch_lib.parent().and_then(|p| p.parent()) {
        let nvidia_dir = site_packages.join("nvidia");
        if let Ok(entries) = std::fs::read_dir(nvidia_dir) {
            for entry in entries.flatten() {
                let lib_dir = entry.path().join("lib");
                if lib_dir.exists() {
                    println!("cargo:rustc-link-search=native={}", lib_dir.display());
                }
            }
        }
    }
}
