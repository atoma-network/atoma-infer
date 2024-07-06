// // Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// // The cuda build time is very long so one can set the CANDLE_FLASH_ATTN_BUILD_DIR environment
// // variable in order to cache the compiled artifacts and avoid recompiling too often.
// use anyhow::{Context, Result};
// use std::path::PathBuf;

// const KERNEL_FILES: [&str; 2] = [
//     "kernels/attention/attention_kernels.cu",
//     "kernels/cache_kernels.cu",
// ];

// fn main() -> Result<()> {
//     println!("cargo:rerun-if-changed=build.rs");
//     for kernel_file in KERNEL_FILES.iter() {
//         println!("cargo:rerun-if-changed={kernel_file}");
//     }
//     let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
//     let build_dir = match std::env::var("ATOMA_PAGED_ATTENTION_BUILD_DIR") {
//         Err(_) =>
//         {
//             #[allow(clippy::redundant_clone)]
//             out_dir.clone()
//         }
//         Ok(build_dir) => {
//             let path = PathBuf::from(build_dir);
//             path.canonicalize().expect(&format!(
//                 "Directory doesn't exists: {} (the current directory is {})",
//                 &path.display(),
//                 std::env::current_dir()?.display()
//             ))
//         }
//     };

//     let kernels = KERNEL_FILES.iter().collect();
//     let builder = bindgen_cuda::Builder::default()
//         .kernel_paths(kernels)
//         .out_dir(build_dir.clone())
//         .arg("-gencode=arch=compute_89,code=sm_89")
//         .arg("--verbose");

//     let out_file = build_dir.join("libpagedattention.a");
//     builder.build_lib(out_file);

//     println!("cargo:rustc-link-search={}", build_dir.display());
//     println!("cargo:rustc-link-lib=pagedattention");
//     println!("cargo:rustc-link-lib=dylib=cudart");
//     println!("cargo:rustc-link-lib=dylib=stdc++");

//     Ok(())
// }

use anyhow::Result;
use std::fs::read_to_string;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::path::PathBuf;

fn read_lines(filename: &str) -> Vec<String> {
    let mut result = Vec::new();

    for line in read_to_string(filename).unwrap().lines() {
        result.push(line.to_string())
    }

    result
}

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/pagedattention.cu");
    println!("cargo:rerun-if-changed=src/copy_blocks_kernel.cu");
    println!("cargo:rerun-if-changed=src/reshape_and_cache_kernel.cu");
    let builder = bindgen_cuda::Builder::default();
    println!("cargo:info={builder:?}");
    builder.build_lib("libpagedattention.a");

    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/lib.rs").unwrap();

    let kernel_dir = PathBuf::from("./kernels/");
    let absolute_kernel_dir = std::fs::canonicalize(&kernel_dir).unwrap();

    println!(
        "cargo:rustc-link-search=native={}",
        absolute_kernel_dir.display()
    );
    println!("cargo:rustc-link-lib=pagedattention");
    println!("cargo:rustc-link-lib=dylib=cudart");

    let contents = read_lines("src/lib.rs");
    for line in contents {
        if line == "pub mod ffi;" {
            return Ok(());
        }
    }
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open("src/lib.rs")
        .unwrap();
    //Expose paged attention interface to Rust
    if let Err(e) = writeln!(file, "pub mod ffi;") {
        anyhow::bail!("error while building dependencies: {:?}\n", e,)
    } else {
        Ok(())
    }
}

