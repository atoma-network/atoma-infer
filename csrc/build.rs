// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the ATOMA_FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.
use anyhow::{Context, Result};
use std::path::PathBuf;

const KERNEL_FILES: [&str; 2] = [
    "kernels/flash_fwd_hdim32_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim32_bf16_sm80.cu",
];

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo:rerun-if-changed={kernel_file}");
    }
    println!("cargo:rerun-if-changed=kernels/flash_fwd_kernel.h");
    println!("cargo:rerun-if-changed=kernels/flash_fwd_launch_template.h");
    println!("cargo:rerun-if-changed=kernels/flash.h");
    println!("cargo:rerun-if-changed=kernels/philox.cuh");
    println!("cargo:rerun-if-changed=kernels/softmax.h");
    println!("cargo:rerun-if-changed=kernels/utils.h");
    println!("cargo:rerun-if-changed=kernels/kernel_traits.h");
    println!("cargo:rerun-if-changed=kernels/block_info.h");
    println!("cargo:rerun-if-changed=kernels/static_switch.h");
    println!("cargo:rerun-if-changed=kernels/rotary.h");
    println!("cargo:rerun-if-changed=kernels/alibi.h");
    println!("cargo:rerun-if-changed=kernels/flash_api.cpp");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let build_dir = match std::env::var("ATOMA_FLASH_ATTN_BUILD_DIR") {
        Err(_) =>
        {
            #[allow(clippy::redundant_clone)]
            out_dir.clone()
        }
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            path.canonicalize().expect(&format!(
                "Directory doesn't exists: {} (the current directory is {})",
                &path.display(),
                std::env::current_dir()?.display()
            ))
        }
    };
    println!("cargo:warning={:?}", build_dir.display());
    println!("cargo:warning={:?}", std::env::current_dir()?.display());

    let kernels = KERNEL_FILES.iter().collect();
    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(kernels)
        .out_dir(build_dir.clone())
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("-Icsrc/cutlass/include")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("--verbose");

    println!("cargo:info={builder:?}");

    let out_file = build_dir.join("libflashattention.a");
    builder.build_lib(out_file);

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=flashattention");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}
