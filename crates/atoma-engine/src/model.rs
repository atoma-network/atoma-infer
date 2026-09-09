//! Fetching the model's files from the Hub, and reading what they declare: the end-of-sequence
//! ids, settled before any device is touched, and the Llama configuration a device is built from.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{self, BufReader};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::{ApiBuilder, ApiError, ApiRepo};
use hf_hub::{Repo, RepoType};
use serde::Deserialize;
use thiserror::Error;
use tracing::info;

use atoma_core::kv::{BlockLayout, CacheKind, KvCacheSpec, KvSource, LayerGroup};
use atoma_core::types::{LayerGroupId, TokenCount};

use crate::config::{Dtype, ModelConfig, ModelId};

#[cfg(feature = "cuda")]
use models::llama::{Config, LlamaConfig};

/// The environment variable a Hub token is read from.
const HF_TOKEN: &str = "HF_TOKEN";
const CONFIG_FILE: &str = "config.json";
const TOKENIZER_FILE: &str = "tokenizer.json";
const WEIGHTS_FILE: &str = "model.safetensors";
const WEIGHTS_INDEX_FILE: &str = "model.safetensors.index.json";

/// Where the model's files are on this machine once fetched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFiles {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    /// The safetensors shards, in name order.
    pub weights: Vec<PathBuf>,
}

/// Why the model's files could not be fetched or read.
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("{id} at revision {revision} could not be fetched from the Hub: {source}")]
    Hub {
        id: ModelId,
        revision: String,
        #[source]
        source: ApiError,
    },
    #[error(
        "{id} at revision {revision} holds neither {WEIGHTS_FILE} nor {WEIGHTS_INDEX_FILE}; \
         only safetensors weights are loaded"
    )]
    NoWeights { id: ModelId, revision: String },
    #[error("{} could not be read: {source}", path.display())]
    Unreadable {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{} is not the JSON expected of it: {source}", path.display())]
    Malformed {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    /// The configured end-of-sequence ids are not the model's. Both are named so the
    /// configuration can be corrected.
    #[error(
        "engine.scheduler.eos_token_ids is {configured:?} but the model's {CONFIG_FILE} declares \
         {model:?}; set engine.scheduler.eos_token_ids to what the model declares"
    )]
    EosMismatch {
        configured: Vec<u32>,
        model: Vec<u32>,
    },
    /// The model's config declares a cache geometry of zero somewhere a count is needed.
    #[error("{} declares {what} as {value}, which is not a count", path.display())]
    NotACount {
        path: PathBuf,
        what: &'static str,
        value: usize,
    },
}

/// Fetches `model`'s config, tokenizer and safetensors from the Hub, or finds them in the cache.
/// A token in the `HF_TOKEN` environment variable authenticates the fetch.
///
/// # Errors
///
/// Returns [`ModelError`] when the Hub cannot be reached, the repository holds no safetensors,
/// or the weight index cannot be read.
pub fn fetch(model: &ModelConfig) -> Result<ModelFiles, ModelError> {
    let hub = hub_error(model);
    let mut builder = ApiBuilder::from_env().with_progress(false);
    if let Ok(token) = env::var(HF_TOKEN) {
        builder = builder.with_token(Some(token));
    }
    if let Some(cache_dir) = &model.cache_dir {
        builder = builder.with_cache_dir(cache_dir.clone());
    }
    let api = builder.build().map_err(&hub)?;
    let repo = api.repo(Repo::with_revision(
        model.id.as_str().to_owned(),
        RepoType::Model,
        model.revision.clone(),
    ));
    info!(id = %model.id, revision = %model.revision, "fetching the model's files");
    let config = repo.get(CONFIG_FILE).map_err(&hub)?;
    let tokenizer = repo.get(TOKENIZER_FILE).map_err(&hub)?;
    let weights = fetch_weights(&repo, model)?;
    info!(
        shards = weights.len(),
        "the model's files are on this machine"
    );
    Ok(ModelFiles {
        config,
        tokenizer,
        weights,
    })
}

/// The safetensors shards the repository holds: the one file, or every file its index names.
fn fetch_weights(repo: &ApiRepo, model: &ModelConfig) -> Result<Vec<PathBuf>, ModelError> {
    let hub = hub_error(model);
    let held: HashSet<String> = repo
        .info()
        .map_err(&hub)?
        .siblings
        .into_iter()
        .map(|sibling| sibling.rfilename)
        .collect();
    if held.contains(WEIGHTS_INDEX_FILE) {
        let index = repo.get(WEIGHTS_INDEX_FILE).map_err(&hub)?;
        return weight_files_in_index(&index)?
            .iter()
            .map(|shard| repo.get(shard).map_err(&hub))
            .collect();
    }
    if held.contains(WEIGHTS_FILE) {
        return Ok(vec![repo.get(WEIGHTS_FILE).map_err(&hub)?]);
    }
    Err(ModelError::NoWeights {
        id: model.id.clone(),
        revision: model.revision.clone(),
    })
}

/// The Hub error for `model`, naming the repository and revision.
fn hub_error(model: &ModelConfig) -> impl Fn(ApiError) -> ModelError + '_ {
    |source| ModelError::Hub {
        id: model.id.clone(),
        revision: model.revision.clone(),
        source,
    }
}

/// The safetensors index: which shard holds each weight.
#[derive(Deserialize)]
struct WeightIndex {
    weight_map: HashMap<String, String>,
}

/// The shard names a safetensors index maps weights to, each once, in name order.
///
/// # Errors
///
/// Returns [`ModelError`] when the index cannot be read or is not a safetensors index.
pub fn weight_files_in_index(index: &Path) -> Result<Vec<String>, ModelError> {
    let index: WeightIndex = read_json(index)?;
    let shards: BTreeSet<String> = index.weight_map.into_values().collect();
    Ok(shards.into_iter().collect())
}

/// The end-of-sequence ids a model's `config.json` declares: one, several, or none.
#[derive(Deserialize)]
struct EosDeclaration {
    #[serde(default)]
    eos_token_id: Option<EosTokenIds>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EosTokenIds {
    Single(u32),
    Multiple(Vec<u32>),
}

/// The end-of-sequence ids `config` declares, in declaration order; none when it declares
/// none.
///
/// # Errors
///
/// Returns [`ModelError`] when `config` cannot be read or is not a model configuration.
pub fn eos_token_ids(config: &Path) -> Result<Vec<u32>, ModelError> {
    let declaration: EosDeclaration = read_json(config)?;
    Ok(match declaration.eos_token_id {
        Some(EosTokenIds::Single(id)) => vec![id],
        Some(EosTokenIds::Multiple(ids)) => ids,
        None => Vec::new(),
    })
}

/// Refuses `configured` end-of-sequence ids that are not, as a set, the model's.
///
/// # Errors
///
/// Returns [`ModelError::EosMismatch`] naming both sets when they differ.
pub fn check_eos_token_ids(configured: &[u32], model: &[u32]) -> Result<(), ModelError> {
    let configured_set: BTreeSet<u32> = configured.iter().copied().collect();
    let model_set: BTreeSet<u32> = model.iter().copied().collect();
    if configured_set == model_set {
        return Ok(());
    }
    Err(ModelError::EosMismatch {
        configured: configured.to_vec(),
        model: model.to_vec(),
    })
}

/// What a checkpoint's `config.json` declares of its cache's geometry.
#[derive(Debug, Deserialize)]
struct KvGeometryDeclaration {
    num_hidden_layers: usize,
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
}

/// The model's cache as one layer group of full attention, read from its `config.json`: every
/// layer caches keys and values for `num_key_value_heads` heads of `hidden_size /
/// num_attention_heads` elements each, in `dtype`, over blocks of `block_size` tokens. What a
/// configuration's padding costs is answerable from this before anything is allocated. The
/// bytes are the whole cache's; under tensor parallelism each rank holds its share of the heads.
///
/// # Errors
///
/// Returns [`ModelError`] when the file cannot be read or is not a model configuration, and
/// [`ModelError::NotACount`] when it declares a layer count, a head count or a head width of
/// zero.
///
/// # Panics
///
/// Never: one layer group writing its own cache passes every rule [`KvCacheSpec::new`] checks.
pub fn kv_cache_spec(
    config: &Path,
    block_size: TokenCount,
    dtype: Dtype,
) -> Result<KvCacheSpec, ModelError> {
    let declared: KvGeometryDeclaration = read_json(config)?;
    let count = |what: &'static str, value: usize| {
        NonZeroUsize::new(value).ok_or_else(|| ModelError::NotACount {
            path: config.to_path_buf(),
            what,
            value,
        })
    };
    let attention_heads = count("num_attention_heads", declared.num_attention_heads)?;
    let group = LayerGroup {
        id: LayerGroupId::new(0),
        layer_count: count("num_hidden_layers", declared.num_hidden_layers)?,
        kind: CacheKind::Full,
        layout: BlockLayout {
            block_size,
            kv_head_count: count("num_key_value_heads", declared.num_key_value_heads)?,
            head_width: count(
                "hidden_size / num_attention_heads",
                declared.hidden_size / attention_heads.get(),
            )?,
            element_bytes: dtype.element_bytes(),
        },
        kv_source: KvSource::Own,
    };
    Ok(KvCacheSpec::new(vec![group]).expect("one layer group writing its own cache is a spec"))
}

/// The model's Llama configuration as its `config.json` declares it.
///
/// # Errors
///
/// Returns [`ModelError`] when the file cannot be read or is not a Llama configuration.
#[cfg(feature = "cuda")]
pub fn llama_config(config: &Path) -> Result<Config, ModelError> {
    let declared: LlamaConfig = read_json(config)?;
    Ok(declared.into_config())
}

/// The JSON at `path`, read as `T`.
///
/// # Errors
///
/// Returns [`ModelError`] when the file cannot be read or does not hold a `T`.
pub(crate) fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, ModelError> {
    let file = File::open(path).map_err(|source| ModelError::Unreadable {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_reader(BufReader::new(file)).map_err(|source| ModelError::Malformed {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use atoma_core::kv::PaddingReservation;
    use atoma_core::types::{RequestCount, TokenCount};
    use tempfile::TempDir;

    use super::{
        check_eos_token_ids, eos_token_ids, kv_cache_spec, weight_files_in_index, ModelError,
    };
    use crate::config::Dtype;

    fn file(dir: &TempDir, name: &str, contents: &str) -> PathBuf {
        let path = dir.path().join(name);
        fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn eos_ids_are_read_whether_one_several_or_none_are_declared() {
        let dir = TempDir::new().unwrap();
        let single = file(
            &dir,
            "single.json",
            r#"{"vocab_size": 8, "eos_token_id": 2}"#,
        );
        assert_eq!(eos_token_ids(&single).unwrap(), [2]);
        let several = file(
            &dir,
            "several.json",
            r#"{"eos_token_id": [128001, 128008, 128009], "hidden_size": 16}"#,
        );
        assert_eq!(
            eos_token_ids(&several).unwrap(),
            [128_001, 128_008, 128_009]
        );
        let none = file(&dir, "none.json", r#"{"vocab_size": 8}"#);
        assert!(eos_token_ids(&none).unwrap().is_empty());
    }

    #[test]
    fn the_cache_spec_is_one_full_attention_group_of_the_configs_geometry() {
        // Llama 3.1 8B: 32 layers, 8 KV heads of 4096 / 32 = 128 elements, bf16.
        let dir = TempDir::new().unwrap();
        let config = file(
            &dir,
            "config.json",
            r#"{"num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32,
                "num_key_value_heads": 8, "vocab_size": 128256}"#,
        );
        let block_size = TokenCount::new(16).unwrap();

        let spec = kv_cache_spec(&config, block_size, Dtype::Bf16).unwrap();

        // One block: 32 layers of 16 tokens, each token K and V over 8 heads of 128 bf16.
        assert_eq!(spec.groups().len(), 1);
        assert_eq!(spec.bytes_per_block(), 32 * 16 * 2 * 8 * 128 * 2);
        // What the padding dummies of a maximum batch of 64 cost: one block each.
        assert_eq!(
            PaddingReservation::cost_bytes(&spec, RequestCount::new(64).unwrap()),
            64 * 32 * 16 * 2 * 8 * 128 * 2
        );
        assert_eq!(
            kv_cache_spec(&config, block_size, Dtype::F32)
                .unwrap()
                .bytes_per_block(),
            32 * 16 * 2 * 8 * 128 * 4,
            "f32 elements are four bytes"
        );
    }

    #[test]
    fn a_geometry_of_zero_is_refused_naming_the_field() {
        let dir = TempDir::new().unwrap();
        let block_size = TokenCount::new(16).unwrap();
        let no_heads = file(
            &dir,
            "no_heads.json",
            r#"{"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 0,
                "num_key_value_heads": 0}"#,
        );
        let error = kv_cache_spec(&no_heads, block_size, Dtype::Bf16).unwrap_err();
        assert!(
            matches!(
                &error,
                ModelError::NotACount {
                    what: "num_attention_heads",
                    value: 0,
                    ..
                }
            ),
            "{error}"
        );
        assert!(error.to_string().contains("no_heads.json"), "{error}");

        // More attention heads than the hidden size has elements leaves a head no width.
        let thin = file(
            &dir,
            "thin.json",
            r#"{"num_hidden_layers": 2, "hidden_size": 16, "num_attention_heads": 32,
                "num_key_value_heads": 8}"#,
        );
        let error = kv_cache_spec(&thin, block_size, Dtype::Bf16).unwrap_err();
        assert!(
            matches!(
                &error,
                ModelError::NotACount {
                    what: "hidden_size / num_attention_heads",
                    value: 0,
                    ..
                }
            ),
            "{error}"
        );

        let no_layers = file(
            &dir,
            "no_layers.json",
            r#"{"num_hidden_layers": 0, "hidden_size": 64, "num_attention_heads": 4,
                "num_key_value_heads": 4}"#,
        );
        let error = kv_cache_spec(&no_layers, block_size, Dtype::Bf16).unwrap_err();
        assert!(
            matches!(
                &error,
                ModelError::NotACount {
                    what: "num_hidden_layers",
                    value: 0,
                    ..
                }
            ),
            "{error}"
        );

        let missing = file(&dir, "missing.json", r#"{"num_hidden_layers": 2}"#);
        let error = kv_cache_spec(&missing, block_size, Dtype::Bf16).unwrap_err();
        assert!(matches!(error, ModelError::Malformed { .. }), "{error}");
    }

    #[test]
    fn a_config_that_cannot_be_read_names_its_path() {
        let dir = TempDir::new().unwrap();
        let missing = dir.path().join("missing.json");
        let error = eos_token_ids(&missing).unwrap_err();
        assert!(matches!(error, ModelError::Unreadable { .. }), "{error}");
        assert!(error.to_string().contains("missing.json"), "{error}");

        let malformed = file(&dir, "malformed.json", r#"{"eos_token_id": "two"}"#);
        let error = eos_token_ids(&malformed).unwrap_err();
        assert!(matches!(error, ModelError::Malformed { .. }), "{error}");
        assert!(error.to_string().contains("malformed.json"), "{error}");
    }

    #[test]
    fn configured_eos_ids_must_be_the_models_as_a_set() {
        check_eos_token_ids(&[2], &[2]).unwrap();
        check_eos_token_ids(&[128_009, 128_001], &[128_001, 128_009]).unwrap();
        check_eos_token_ids(&[], &[]).unwrap();
        let error = check_eos_token_ids(&[2], &[128_001, 128_009]).unwrap_err();
        assert!(
            matches!(&error, ModelError::EosMismatch { configured, model }
                if configured == &[2] && model == &[128_001, 128_009]),
            "{error}"
        );
        assert!(error.to_string().contains("[2]"), "{error}");
        assert!(error.to_string().contains("[128001, 128009]"), "{error}");
        assert!(
            check_eos_token_ids(&[2], &[]).is_err(),
            "a model declaring none disagrees with a configuration declaring one"
        );
    }

    #[test]
    fn the_weight_index_names_each_shard_once_in_order() {
        let dir = TempDir::new().unwrap();
        let index = file(
            &dir,
            "model.safetensors.index.json",
            r#"{"metadata": {"total_size": 3}, "weight_map": {
                "lm_head.weight": "model-00002-of-00002.safetensors",
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.norm.weight": "model-00002-of-00002.safetensors"
            }}"#,
        );
        assert_eq!(
            weight_files_in_index(&index).unwrap(),
            [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ]
        );
        let not_an_index = file(&dir, "config.json", r#"{"vocab_size": 8}"#);
        assert!(matches!(
            weight_files_in_index(&not_an_index).unwrap_err(),
            ModelError::Malformed { .. }
        ));
    }
}
