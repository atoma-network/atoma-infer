//! What the executor is built from: where the model comes from and how it is loaded, and the
//! device and core each rank owns.

use std::collections::HashSet;
use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

use crate::decode::ring::StagingDepth;

/// Where the model comes from and how its weights are loaded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// The Hugging Face Hub repository the model is fetched from.
    #[validate(custom(function = "names_a_repository"))]
    pub id: ModelId,
    /// The revision of the repository to fetch.
    #[serde(default = "main_revision")]
    #[validate(length(min = 1, message = "model.revision is empty"))]
    pub revision: String,
    /// Where fetched files are kept; the Hub's default cache when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_dir: Option<PathBuf>,
    /// The dtype the weights are loaded in and the forward computes in.
    pub dtype: Dtype,
    /// The chat template a conversation is rendered through before it is tokenized.
    pub prompt_template: PromptTemplate,
}

fn main_revision() -> String {
    "main".to_owned()
}

/// A Hugging Face Hub repository the engine is served from.
///
/// Its own type because the strings it sits beside mean other things — the revision in this
/// configuration, the completion's own id in a response — and a swap between two `String` fields
/// is silent.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(String);

impl ModelId {
    /// The repository `id` names.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The repository as the Hub and the API spell it.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

fn names_a_repository(id: &ModelId) -> Result<(), ValidationError> {
    if id.as_str().is_empty() {
        return Err(validation_error(
            "model_names_no_repository",
            "model.id names no repository".to_owned(),
        ));
    }
    Ok(())
}

/// The chat template a conversation is rendered through.
///
/// A checkpoint's template is not derivable from its repository name: an ungated mirror carries
/// the same weights and the same template under a different owner, and two checkpoints from one
/// owner can differ. The template is therefore declared beside the repository rather than
/// guessed from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptTemplate {
    Llama2,
    Llama3,
    Hermes3,
}

/// The dtype the model computes in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    F16,
    Bf16,
    F32,
}

/// The executor: one thread per rank, each owning a device and pinned to a core.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
#[validate(schema(function = "ranks_own_distinct_devices_and_cores"))]
pub struct ExecutorConfig {
    /// One entry per rank, in rank order: the first is rank zero, which owns the engine's rings.
    #[validate(length(min = 1, message = "executor.ranks is empty; name at least one rank"))]
    pub ranks: Vec<RankConfig>,
    /// How many staging entries the staging ring holds: two unless set, so the host writes one
    /// step's inputs while the device is still reading the last step's. Between one and
    /// [`StagingDepth::MAX`]; a depth outside that refuses the configuration.
    #[serde(default)]
    pub staging_depth: StagingDepth,
}

/// What one rank owns: its device, and the core its thread is pinned to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankConfig {
    pub device: DeviceOrdinal,
    pub core: CoreId,
}

/// A CUDA device, by the ordinal the driver enumerates it under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceOrdinal(usize);

impl DeviceOrdinal {
    #[must_use]
    pub const fn new(ordinal: usize) -> Self {
        Self(ordinal)
    }

    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// A CPU core, by the id the operating system enumerates it under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CoreId(usize);

impl CoreId {
    #[must_use]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl fmt::Display for CoreId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// One executor rank: its position in [`ExecutorConfig::ranks`]. Rank zero owns the engine's
/// rings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rank(usize);

impl Rank {
    pub const ZERO: Self = Self(0);

    #[must_use]
    pub const fn new(rank: usize) -> Self {
        Self(rank)
    }

    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Every rank owns its own device and is pinned to its own core, so a device or core named twice
/// would be shared.
fn ranks_own_distinct_devices_and_cores(config: &ExecutorConfig) -> Result<(), ValidationError> {
    let mut devices = HashSet::new();
    let mut cores = HashSet::new();
    for rank in &config.ranks {
        if !devices.insert(rank.device) {
            return Err(validation_error(
                "device_shared_by_ranks",
                format!(
                    "executor.ranks name device {} more than once; every rank owns its own \
                     device",
                    rank.device.get()
                ),
            ));
        }
        if !cores.insert(rank.core) {
            return Err(validation_error(
                "core_shared_by_ranks",
                format!(
                    "executor.ranks name core {} more than once; every rank is pinned to its own \
                     core",
                    rank.core.get()
                ),
            ));
        }
    }
    Ok(())
}

fn validation_error(code: &'static str, message: String) -> ValidationError {
    let mut error = ValidationError::new(code);
    error.message = Some(message.into());
    error
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use figment::providers::{Format, Toml};
    use figment::Figment;
    use serde::de::DeserializeOwned;
    use validator::Validate;

    use super::{
        CoreId, DeviceOrdinal, Dtype, ExecutorConfig, ModelConfig, ModelId, PromptTemplate,
        RankConfig,
    };
    use crate::decode::ring::StagingDepth;

    fn from_toml<T: DeserializeOwned>(source: &str) -> Result<T, Box<figment::Error>> {
        Figment::new()
            .merge(Toml::string(source))
            .extract()
            .map_err(Box::new)
    }

    fn rank(device: usize, core: usize) -> RankConfig {
        RankConfig {
            device: DeviceOrdinal::new(device),
            core: CoreId::new(core),
        }
    }

    #[test]
    fn the_model_is_read_from_toml_with_its_defaults_filled_in() {
        let config: ModelConfig = from_toml(
            r#"
            id = "meta-llama/Llama-3.2-1B-Instruct"
            dtype = "bf16"
            prompt_template = "llama3"
            "#,
        )
        .unwrap();
        assert_eq!(
            config,
            ModelConfig {
                id: ModelId::new("meta-llama/Llama-3.2-1B-Instruct"),
                revision: "main".to_owned(),
                cache_dir: None,
                dtype: Dtype::Bf16,
                prompt_template: PromptTemplate::Llama3,
            }
        );
        config.validate().unwrap();

        let config: ModelConfig = from_toml(
            r#"
            id = "meta-llama/Llama-3.2-1B-Instruct"
            revision = "abc123"
            cache_dir = "/var/cache/models"
            dtype = "f16"
            prompt_template = "llama3"
            "#,
        )
        .unwrap();
        assert_eq!(config.revision, "abc123");
        assert_eq!(config.cache_dir, Some(PathBuf::from("/var/cache/models")));
        assert_eq!(config.dtype, Dtype::F16);
    }

    #[test]
    fn the_ranks_are_read_from_toml_as_inline_tables_in_rank_order() {
        let config: ExecutorConfig =
            from_toml("ranks = [{ device = 0, core = 2 }, { device = 1, core = 3 }]").unwrap();
        assert_eq!(config.ranks, vec![rank(0, 2), rank(1, 3)]);
        config.validate().unwrap();
    }

    #[test]
    fn the_staging_depth_is_two_unless_configured() {
        let config: ExecutorConfig = from_toml("ranks = [{ device = 0, core = 2 }]").unwrap();
        assert_eq!(config.staging_depth.get(), 2);

        let config: ExecutorConfig =
            from_toml("ranks = [{ device = 0, core = 2 }]\nstaging_depth = 3").unwrap();
        assert_eq!(config.staging_depth.get(), 3);
    }

    #[test]
    fn a_staging_depth_of_zero_or_above_the_deepest_is_refused_by_name() {
        for depth in [0, 9] {
            let source = format!("ranks = [{{ device = 0, core = 2 }}]\nstaging_depth = {depth}");
            let error = from_toml::<ExecutorConfig>(&source)
                .unwrap_err()
                .to_string();
            assert!(error.contains("staging_depth"), "depth {depth}: {error}");
            assert!(error.contains("between 1 and 8"), "depth {depth}: {error}");
        }
    }

    #[test]
    fn both_configurations_round_trip_through_serde() {
        let model = ModelConfig {
            id: ModelId::new("org/model"),
            revision: "v1".to_owned(),
            cache_dir: Some(PathBuf::from("/cache")),
            dtype: Dtype::F32,
            prompt_template: PromptTemplate::Hermes3,
        };
        let json = serde_json::to_string(&model).unwrap();
        assert_eq!(serde_json::from_str::<ModelConfig>(&json).unwrap(), model);

        let executor = ExecutorConfig {
            ranks: vec![rank(0, 2)],
            staging_depth: StagingDepth::new(4).unwrap(),
        };
        let json = serde_json::to_string(&executor).unwrap();
        assert_eq!(
            json,
            r#"{"ranks":[{"device":0,"core":2}],"staging_depth":4}"#
        );
        assert_eq!(
            serde_json::from_str::<ExecutorConfig>(&json).unwrap(),
            executor
        );
    }

    #[test]
    fn unknown_fields_and_unknown_dtypes_are_refused() {
        assert!(
            from_toml::<ExecutorConfig>("ranks = [{ device = 0, core = 2, gpu = 1 }]").is_err()
        );
        assert!(from_toml::<ExecutorConfig>("ranks = []\nworld_size = 1").is_err());
        assert!(from_toml::<ModelConfig>(
            r#"id = "org/model"
dtype = "int8""#
        )
        .is_err());
    }

    #[test]
    fn an_executor_with_no_ranks_is_refused() {
        let config = ExecutorConfig {
            ranks: Vec::new(),
            staging_depth: StagingDepth::default(),
        };
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("at least one rank"), "{error}");
    }

    #[test]
    fn ranks_sharing_a_device_or_a_core_are_refused_by_name() {
        let shared_device = ExecutorConfig {
            ranks: vec![rank(0, 2), rank(0, 3)],
            staging_depth: StagingDepth::default(),
        };
        let error = shared_device.validate().unwrap_err().to_string();
        assert!(error.contains("device 0 more than once"), "{error}");

        let shared_core = ExecutorConfig {
            ranks: vec![rank(0, 2), rank(1, 2)],
            staging_depth: StagingDepth::default(),
        };
        let error = shared_core.validate().unwrap_err().to_string();
        assert!(error.contains("core 2 more than once"), "{error}");
    }

    #[test]
    fn a_model_with_no_repository_or_no_revision_is_refused() {
        let no_id = ModelConfig {
            id: ModelId::new(""),
            revision: "main".to_owned(),
            cache_dir: None,
            dtype: Dtype::Bf16,
            prompt_template: PromptTemplate::Llama3,
        };
        let error = no_id.validate().unwrap_err().to_string();
        assert!(error.contains("names no repository"), "{error}");

        let no_revision = ModelConfig {
            id: ModelId::new("org/model"),
            revision: String::new(),
            cache_dir: None,
            dtype: Dtype::Bf16,
            prompt_template: PromptTemplate::Llama3,
        };
        let error = no_revision.validate().unwrap_err().to_string();
        assert!(error.contains("revision is empty"), "{error}");
    }
}
