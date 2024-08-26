use std::fmt::Debug;

use candle_core::Tensor;

pub fn proper_tensor_print<T: Debug + candle_core::WithDType>(tensor: &Tensor, nested: usize) {
    print!("{}", " ".repeat(nested));
    if tensor.dims().len() == 1 {
        println!("{:?}", tensor.to_vec1::<T>().unwrap());
    } else {
        println!("[");
        for i in 0..tensor.dims()[0] {
            proper_tensor_print::<T>(&tensor.get(i).unwrap(), nested + 1);
        }
        print!("{}", " ".repeat(nested));
        println!("]");
    }
}

#[macro_export]
macro_rules! print_tensor_with_type {
    ($x:expr, $t:ty, $d:expr) => {{
        let (storage, layout) = $x.storage_and_layout();
        let ptr = match &*storage {
            candle_core::Storage::Cuda(c) => {
                let cuda_slice = c.as_cuda_slice::<$t>()?;
                let data = cuda_slice.slice(layout.start_offset()..);
                let stride = layout.stride();
                let rank = stride.len();
                if stride[rank - 1] != 1 {
                    candle_core::bail!("block_table must be contiguous")
                }
                *data.device_ptr() as *const i64
            }
            _ => candle_core::bail!("block_table must be a cuda tensor"),
        };
        println!(
            "{}:{} - {} dtype {:?} dims {:?} contiguous {} ptr {:?}",
            file!(),
            line!(),
            stringify!($x),
            $x.dtype(),
            $x.dims(),
            $x.is_contiguous(),
            ptr
        );
        if $d {
            proper_tensor_print::<$t>(&$x, 0);
        }
    }};
}

#[macro_export]
macro_rules! print_tensor {
    ($x:expr, $y:expr) => {
        if $y {
            use candle_core::cuda::cudarc::driver::DevicePtr;
            use help::print_tensor_with_type;
            use help::proper_tensor_print;
            match $x.dtype() {
                DType::U8 => print_tensor_with_type!($x, u8, true),
                DType::U32 => print_tensor_with_type!($x, u32, true),
                DType::I64 => print_tensor_with_type!($x, i64, true),
                DType::BF16 => print_tensor_with_type!($x, half::bf16, true),
                DType::F16 => print_tensor_with_type!($x, half::f16, true),
                DType::F32 => print_tensor_with_type!($x, f32, true),
                DType::F64 => print_tensor_with_type!($x, f64, true),
            }
        }
    };
}

#[macro_export]
macro_rules! print_tensor_no_data {
    ($x:expr, $y:expr) => {
        if $y {
            use help::print_tensor_with_type;
            use help::proper_tensor_print;
            match $x.dtype() {
                DType::U8 => print_tensor_with_type!($x, u8, false),
                DType::U32 => print_tensor_with_type!($x, u32, false),
                DType::I64 => print_tensor_with_type!($x, i64, false),
                DType::BF16 => print_tensor_with_type!($x, half::bf16, false),
                DType::F16 => print_tensor_with_type!($x, half::f16, false),
                DType::F32 => print_tensor_with_type!($x, f32, false),
                DType::F64 => print_tensor_with_type!($x, f64, false),
            }
        }
    };
}
