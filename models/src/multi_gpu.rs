use candle_core::{CpuStorage, CustomOp1, DType, Layout, Result, Shape, Tensor};
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use candle_nn::{Linear, Module};
use cudarc::nccl::Comm;
use half::{bf16, f16};
use std::rc::Rc;

pub struct TensorParallelColumnLinear {
    pub linear: Linear,
}

impl TensorParallelColumnLinear {
    pub fn new(linear: Linear) -> Self {
        Self { linear }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
    pub fn load(vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weight = vb.get_with_hints((), "weight", shard(0, rank, size))?;
        Ok(Self::new(Linear::new(weight, None)))
    }

    pub fn load_multi(vb: VarBuilder, prefixes: &[&str], comm: Rc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weights: Vec<_> = prefixes
            .iter()
            .map(|p| vb.pp(p).get_with_hints((), "weight", shard(0, rank, size)))
            .collect::<Result<Vec<_>>>()?;
        let weight = Tensor::cat(&weights, 0)?;
        Ok(Self::new(Linear::new(weight, None)))
    }
}

pub struct TensorParallelRowLinear {
    linear: Linear,
    all_reduce: AllReduce,
}

impl TensorParallelRowLinear {
    pub fn new(linear: Linear, comm: Rc<Comm>) -> Self {
        let all_reduce = AllReduce { comm };
        Self { linear, all_reduce }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)?.apply_op1_no_bwd(&self.all_reduce)
    }
    pub fn load(vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weight = vb.get_with_hints((), "weight", shard(1, rank, size))?;
        Ok(Self::new(Linear::new(weight, None), comm))
    }
}

pub struct AllGather {
    pub comm: Rc<Comm>,
}

unsafe impl Sync for AllGather {}
unsafe impl Send for AllGather {}

impl CustomOp1 for AllGather {
    fn name(&self) -> &'static str {
        "all_gather"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("AllGather is never used on cpu")
    }

    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::{backend::BackendStorage, cuda_backend::WrapErr};
        use cudarc::driver::DeviceSlice;

        let elem_count = l.shape().elem_count();
        let num_devices = self.comm.world_size();
        let dev = s.device().clone();
        let mut new_shape: Vec<usize> = l.shape().dims().to_vec();
        if let Some(last) = new_shape.last_mut() {
            *last *= num_devices;
        }
        // new_shape[new_shape.len() - 1] *= num_devices;
        let new_shape = Shape::from(new_shape);
        let dst = match s.dtype() {
            DType::BF16 => {
                let s = s.as_cuda_slice::<bf16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<bf16>(elem_count * num_devices) }.w()?;
                self.comm
                    .all_gather(s, &mut dst)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F16 => {
                let s = s.as_cuda_slice::<f16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<f16>(elem_count * num_devices) }.w()?;
                self.comm
                    .all_gather(s, &mut dst)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
        };
        Ok((dst, new_shape))
    }
}

pub struct AllReduce {
    comm: Rc<Comm>,
}

/// This is actually not safe: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
/// But for this example purposes, this will work
unsafe impl Sync for AllReduce {}
unsafe impl Send for AllReduce {}

impl CustomOp1 for AllReduce {
    fn name(&self) -> &'static str {
        "allreduce"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("AllReduce is never used on cpu")
    }

    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::{backend::BackendStorage, cuda_backend::WrapErr};
        use cudarc::{driver::DeviceSlice, nccl::ReduceOp};

        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let dst = match s.dtype() {
            DType::BF16 => {
                let s = s.as_cuda_slice::<bf16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F16 => {
                let s = s.as_cuda_slice::<f16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
        };
        Ok((dst, l.shape().clone()))
    }
}

pub fn shard(dim: usize, rank: usize, world_size: usize) -> candle_nn::var_builder::Shard {
    candle_nn::var_builder::Shard {
        dim,
        rank,
        world_size,
    }
}
