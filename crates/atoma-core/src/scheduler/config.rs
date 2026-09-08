//! What the scheduler is built from, and the configurations it refuses to start under.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::{Validate, ValidationError};

use crate::kv::{HashAlgorithm, PaddingReservation};
use crate::scheduler::AdmissionPolicy;
use crate::types::{RequestCount, TokenCount};

/// What the scheduler is built from, fixed for the process lifetime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
#[validate(schema(function = "a_batch_is_drawn_from_the_slab"))]
pub struct SchedulerConfig {
    /// Query tokens one step may compute, summed over entries.
    pub token_budget: TokenCount,
    /// Entries one step may hold. It bounds one step, and with it the buckets a step can fill:
    /// a uniform decode gives every entry one token, so a bucket above it never fills.
    pub max_batch: RequestCount,
    /// The longest sequence the model serves.
    pub max_model_len: TokenCount,
    /// Tokens per KV block.
    pub block_size: TokenCount,
    /// Admission candidates one pass examines.
    pub window: RequestCount,
    /// How admission orders the window.
    pub admission: AdmissionPolicy,
    /// Requests the slab holds at once — running, waiting and preempted together — before
    /// ingress is refused. Where `max_batch` bounds one step, this bounds the whole population
    /// a step is drawn from.
    pub max_requests: RequestCount,
    /// Generated tokens a client may leave unread on its egress channel before its request is
    /// retired. A client that keeps up leaves none, so this is what one stalled client costs the
    /// host: at most this many events, plus the one pass that runs before the next sweep sees it.
    pub max_client_backlog: TokenCount,
    /// The model's end-of-sequence token ids; sampling one finishes a request that does not
    /// ignore it.
    pub eos_token_ids: Vec<u32>,
    /// The chain-hash algorithm block identity is minted under.
    pub hash_algorithm: HashAlgorithm,
}

impl SchedulerConfig {
    /// The request slots the slab hands out: one per live request, plus the padding dummies,
    /// which take theirs at startup and hold them for the process lifetime. Whatever is indexed
    /// by request slot is sized by this.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.max_requests.get() + PaddingReservation::dummies_for(self.max_batch)
    }
}

/// One step's batch is drawn from the requests the slab holds, so a `max_batch` above
/// `max_requests` names a batch that could never be filled.
fn a_batch_is_drawn_from_the_slab(config: &SchedulerConfig) -> Result<(), ValidationError> {
    if config.max_batch <= config.max_requests {
        return Ok(());
    }
    let mut error = ValidationError::new("max_batch_over_max_requests");
    error.message = Some(
        format!(
            "max_batch is {} but the slab holds only {} requests, so a full batch could never be \
             drawn from it; raise max_requests or lower max_batch",
            config.max_batch.get(),
            config.max_requests.get()
        )
        .into(),
    );
    Err(error)
}

/// A configuration the scheduler refuses to start under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SchedulerError {
    /// The pool cannot hold one request of the maximum model length, so such a request could
    /// never finish and would bounce through preemption forever.
    #[error(
        "the pool has {free} free blocks but one request of the maximum model length needs \
         {needed}"
    )]
    PoolTooSmallForMaxModelLength { needed: usize, free: usize },
}
