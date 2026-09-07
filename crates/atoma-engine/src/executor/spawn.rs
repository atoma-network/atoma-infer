//! Spawning one pinned executor thread per rank, each driving its capture session from
//! Allocation through Capture to Replay before it serves a step.
//!
//! Every rank's thread is launched before any is waited on, and the ranks are waited on together:
//! under NCCL, opening a communicator is a rendezvous that returns only once every rank has
//! joined, so waiting for rank zero alone would never return when a follower fails before it
//! joins.

use atoma_core::engine::{EngineConfig, ExecutorRings};
use atoma_core::types::{RequestCount, TokenCount};
use atoma_runtime::context::RuntimeContext;
use atoma_runtime::session::Allocation;
use candle_core::DType;
use thiserror::Error;
use tracing::info;

use crate::config::{DeviceOrdinal, ExecutorConfig, ModelConfig, Rank, RankConfig};
#[cfg(not(feature = "nccl"))]
use crate::device::decode::{DecodeStep, DecodeStepPlan};
use crate::device::forward::{Allocated, CudaForward};
use crate::device::sampler::DeviceSampler;
use crate::device::{Checkpoint, KvCache, KvGeometry, RankDevice, Weights};
use crate::executor::{
    feed, launch, wait_all, Cause, Executor, ExecutorThread, Follower, FollowerRings, Launched,
    SpawnError,
};
use crate::model::{llama_config, ModelFiles};

#[cfg(feature = "nccl")]
use cudarc::nccl::sys::ncclResult_t;
#[cfg(feature = "nccl")]
use cudarc::nccl::Id;

#[cfg(feature = "nccl")]
use crate::device::Communicator;

/// Why the ranks could not be started.
#[derive(Debug, Error)]
pub enum StartupError {
    #[error(transparent)]
    Spawn(#[from] SpawnError),
    #[error("executor.ranks is empty; name at least one rank")]
    NoRanks,
    #[error(
        "{ranks} ranks are configured but this build has no NCCL; build with the nccl feature \
         or configure one rank"
    )]
    RanksNeedNccl { ranks: usize },
    #[cfg(feature = "nccl")]
    #[error("the NCCL collective could not be opened: {status:?}")]
    Collective { status: ncclResult_t },
}

/// What every rank is built from.
#[derive(Clone)]
struct RankPlan {
    world_size: usize,
    block_count: usize,
    block_size: TokenCount,
    max_batch: RequestCount,
    /// The request slots whatever is indexed by slot is sized for.
    slot_count: usize,
    dtype: DType,
    files: ModelFiles,
    #[cfg(not(feature = "nccl"))]
    decode_step: DecodeStepPlan,
    #[cfg(feature = "nccl")]
    collective: Id,
}

/// Spawns rank zero's executor thread over `rings` and one follower thread per further rank,
/// each pinned to its core and holding its device, and returns once every rank is serving.
///
/// # Errors
///
/// Returns [`StartupError`] when no rank is configured, more than one is without NCCL, the
/// collective cannot be opened, or any rank's thread cannot be spawned, pinned or built.
///
/// # Panics
///
/// Panics on a target whose pointer is narrower than the configured block count.
pub fn spawn_ranks(
    engine: &EngineConfig,
    executor: &ExecutorConfig,
    model: &ModelConfig,
    files: &ModelFiles,
    rings: ExecutorRings,
) -> Result<Vec<ExecutorThread>, StartupError> {
    let Some(leader) = executor.ranks.first().copied() else {
        return Err(StartupError::NoRanks);
    };
    let world_size = executor.ranks.len();
    #[cfg(not(feature = "nccl"))]
    if world_size > 1 {
        return Err(StartupError::RanksNeedNccl { ranks: world_size });
    }
    let plan = RankPlan {
        world_size,
        block_count: usize::try_from(engine.block_count).expect("a u32 block count fits usize"),
        block_size: engine.scheduler.block_size,
        max_batch: engine.scheduler.max_batch,
        slot_count: engine.scheduler.slot_count(),
        dtype: model.dtype.into(),
        files: files.clone(),
        #[cfg(not(feature = "nccl"))]
        decode_step: DecodeStepPlan {
            dispatch: engine.dispatch.clone(),
            max_model_len: engine.scheduler.max_model_len,
            block_size: engine.scheduler.block_size,
            dtype: model.dtype,
            staging_depth: executor.staging_depth,
        },
        #[cfg(feature = "nccl")]
        collective: Id::new().map_err(|error| StartupError::Collective { status: error.0 })?,
    };
    let mut feeds = Vec::with_capacity(world_size - 1);
    let mut followers: Vec<(Rank, RankConfig, FollowerRings)> = Vec::with_capacity(world_size - 1);
    for (index, &config) in executor.ranks.iter().enumerate().skip(1) {
        let rank = Rank::new(index);
        let (leader_end, follower_end) = feed(rank, rings.unparker());
        feeds.push(leader_end);
        followers.push((rank, config, follower_end));
    }

    let mut launched: Vec<Launched> = Vec::with_capacity(world_size);
    let leader_plan = plan.clone();
    launched.push(launch(Rank::ZERO, leader.core, move || {
        let forward = open_forward(Rank::ZERO, leader.device, &leader_plan)?;
        let mut executor = Executor::new(rings, forward, leader_plan.block_size);
        for leader_end in feeds {
            executor.follow(leader_end);
        }
        Ok(executor)
    })?);
    for (rank, config, follower_end) in followers {
        let plan = plan.clone();
        launched.push(launch(rank, config.core, move || {
            let forward = open_forward(rank, config.device, &plan)?;
            Ok(Follower::new(follower_end, forward, plan.block_size))
        })?);
    }
    let threads = wait_all(launched)?;
    info!(ranks = threads.len(), "every rank is serving");
    Ok(threads)
}

/// Opens the forward of rank `rank` on the current thread: drives the rank's session from
/// Allocation to Replay, allocating everything the rank holds in between, the step over runtime
/// tensors included, and hands back the forward that holds it all.
fn open_forward(rank: Rank, ordinal: DeviceOrdinal, plan: &RankPlan) -> Result<CudaForward, Cause> {
    let context = RuntimeContext::new(ordinal.get())?;
    let allocation = Allocation::new(&context)?;
    let device = RankDevice::open(&allocation, ordinal)?;
    let config = llama_config(&plan.files.config)?;
    #[cfg(feature = "nccl")]
    let communicator =
        Communicator::open(&allocation, &device, rank, plan.world_size, plan.collective)?;
    let checkpoint = Checkpoint {
        files: &plan.files,
        config: &config,
        dtype: plan.dtype,
    };
    #[cfg(not(feature = "nccl"))]
    let weights = Weights::load(&allocation, &device, checkpoint)?;
    #[cfg(feature = "nccl")]
    let weights = Weights::load(&allocation, &device, checkpoint, &communicator)?;
    let geometry = KvGeometry::new(&config, plan.block_count, plan.block_size, plan.world_size)?;
    let kv_cache = KvCache::allocate(&allocation, &device, &config, geometry, plan.dtype)?;
    // Rank zero alone samples: a follower runs the forward for its share of the model and
    // produces nothing, so it holds no sampler.
    let sampler = if rank == Rank::ZERO {
        Some(DeviceSampler::new(
            &allocation,
            device.stream(),
            plan.slot_count,
            plan.max_batch,
            config.vocab_size,
        )?)
    } else {
        None
    };
    #[cfg(not(feature = "nccl"))]
    let decode_step =
        DecodeStep::build(&allocation, &device, &weights, &kv_cache, &plan.decode_step)?;
    let session = allocation.into_capture().into_replay();
    info!(%rank, "session in its Replay phase; serving through it");
    Ok(CudaForward::new(
        Allocated {
            device,
            weights,
            kv_cache,
            sampler,
            vocab: config.vocab_size,
        },
        #[cfg(not(feature = "nccl"))]
        decode_step,
        session,
    )?)
}
