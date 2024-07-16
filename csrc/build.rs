// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the ATOMA_FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;

const KERNEL_FILES: [&str; 64] = [
    // "kernels/flash_api.cu",
    "kernels/flash_fwd_hdim32_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim32_bf16_sm80.cu",
    "kernels/flash_fwd_hdim32_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim32_fp16_sm80.cu",
    "kernels/flash_fwd_hdim64_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim64_bf16_sm80.cu",
    "kernels/flash_fwd_hdim64_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim64_fp16_sm80.cu",
    "kernels/flash_fwd_hdim96_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim96_bf16_sm80.cu",
    "kernels/flash_fwd_hdim96_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim96_fp16_sm80.cu",
    "kernels/flash_fwd_hdim128_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim128_bf16_sm80.cu",
    "kernels/flash_fwd_hdim128_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim128_fp16_sm80.cu",
    "kernels/flash_fwd_hdim160_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim160_bf16_sm80.cu",
    "kernels/flash_fwd_hdim160_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim160_fp16_sm80.cu",
    "kernels/flash_fwd_hdim192_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim192_bf16_sm80.cu",
    "kernels/flash_fwd_hdim192_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim192_fp16_sm80.cu",
    "kernels/flash_fwd_hdim224_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim224_bf16_sm80.cu",
    "kernels/flash_fwd_hdim224_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim224_fp16_sm80.cu",
    "kernels/flash_fwd_hdim256_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim256_bf16_sm80.cu",
    "kernels/flash_fwd_hdim256_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim256_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim32_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim32_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim32_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim32_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim64_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim64_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim64_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim64_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim96_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim96_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim96_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim96_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim128_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim128_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim128_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim128_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim160_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim160_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim160_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim160_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim192_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim192_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim192_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim192_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim224_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim224_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim224_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim224_fp16_sm80.cu",
    "kernels/flash_fwd_split_hdim256_bf16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim256_bf16_sm80.cu",
    "kernels/flash_fwd_split_hdim256_fp16_causal_sm80.cu",
    "kernels/flash_fwd_split_hdim256_fp16_sm80.cu",
];

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter().chain(std::iter::once(&"kernels/flash_api.cu")) {
        println!("cargo:rerun-if-changed={kernel_file}");
    }
    // Your existing rerun-if-changed statements for header files

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let build_dir = match std::env::var("ATOMA_FLASH_ATTN_BUILD_DIR") {
        Err(_) => out_dir.clone(),
        Ok(build_dir) => PathBuf::from(build_dir).canonicalize().context("Failed to canonicalize build directory")?
    };
    println!("cargo:warning=Build directory: {:?}", build_dir.display());

    let current_dir = std::env::current_dir()?;
    let cutlass_include_dir = current_dir.join("cutlass/include");
    let cutlass_include_arg = format!("-I{}", cutlass_include_dir.display());

    // Step 1: Compile flash_api.cu first
    compile_flash_api(&build_dir, &cutlass_include_arg)?;

    // Step 2: Compile all other CUDA files
    compile_cuda_files(&build_dir, &cutlass_include_arg)?;

    // Step 3: Link libraries
    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=flash_api");
    println!("cargo:rustc-link-lib=static=flashattention");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}

fn compile_flash_api(build_dir: &PathBuf, cutlass_include_arg: &str) -> Result<()> {
    let flash_api_o = build_dir.join("flash_api.o");
    let status = Command::new("nvcc")
        .args(&[
            "-c",
            "kernels/flash_api.cu",
            "-o", flash_api_o.to_str().unwrap(),
            "--gpu-architecture=sm_80", // Adjust as needed
            "-O2",
            cutlass_include_arg,
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--verbose"
        ])
        .status()
        .context("Failed to compile flash_api.cu")?;

    if !status.success() {
        return Err(anyhow::anyhow!("nvcc command for flash_api.cu failed"));
    }

    // Create libflash_api.a
    let status = Command::new("ar")
        .args(&["rcs", "libflash_api.a", "flash_api.o"])
        .current_dir(build_dir)
        .status()
        .context("Failed to create libflash_api.a")?;

    if !status.success() {
        return Err(anyhow::anyhow!("ar command for libflash_api.a failed"));
    }

    Ok(())
}

fn compile_cuda_files(build_dir: &PathBuf, cutlass_include_arg: &str) -> Result<()> {
    let mut object_files = Vec::new();

    for kernel_file in &KERNEL_FILES {
        let file_stem = std::path::Path::new(kernel_file)
            .file_stem()
            .and_then(std::ffi::OsStr::to_str)
            .context("Failed to get file stem")?;
        let output_file = build_dir.join(format!("{}.o", file_stem));

        let status = Command::new("nvcc")
            .args(&[
                "-c",
                kernel_file,
                "-o", output_file.to_str().unwrap(),
                "--gpu-architecture=sm_80", // Adjust as needed
                "-O2",
                cutlass_include_arg,
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "--verbose"
            ])
            .status()
            .with_context(|| format!("Failed to compile {}", kernel_file))?;

        if !status.success() {
            return Err(anyhow::anyhow!("nvcc command failed for {}", kernel_file));
        }

        object_files.push(output_file);
    }

    // Create libflashattention.a
    let mut ar_command = Command::new("ar");
    ar_command.arg("rcs").arg("libflashattention.a");
    for obj_file in &object_files {
        ar_command.arg(obj_file);
    }
    let status = ar_command
        .current_dir(build_dir)
        .status()
        .context("Failed to create libflashattention.a")?;

    if !status.success() {
        return Err(anyhow::anyhow!("ar command for libflashattention.a failed"));
    }

    Ok(())
}
