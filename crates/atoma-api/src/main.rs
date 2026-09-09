//! The OpenAI-compatible server over the engine thread and the executor threads.
//!
//! Startup fetches the model's files, checks the configured end-of-sequence ids against the
//! model's, loads the tokenizer, spawns the engine thread and one pinned executor thread per
//! rank, and serves. The server stops on Ctrl+C or when the engine thread exits, and ends with
//! the cause when any executor thread failed.

pub mod api;
pub mod completion;
pub mod config;
pub mod detokenize;
pub mod server;
pub mod stream;

#[cfg(feature = "cuda")]
mod startup {
    use std::path::PathBuf;
    use std::sync::Arc;

    use anyhow::{anyhow, Context};
    use atoma_core::attention::{CaptureContract, ModelDeclaration};
    use atoma_core::config::load;
    use atoma_core::engine::{Control, Engine};
    use atoma_core::kv::PaddingReservation;
    use atoma_engine::decode::declaration;
    use atoma_engine::executor::spawn_ranks;
    use atoma_engine::model::{check_eos_token_ids, eos_token_ids, fetch, kv_cache_spec};
    use clap::Parser;
    use tokenizers::Tokenizer;
    use tokio::net::TcpListener;
    use tracing::{error, info};

    use crate::api::chat_completions::ServedModel;
    use crate::config::Config;
    use crate::server::{run_server, AppState, EngineThreads};

    #[derive(Debug, Parser)]
    #[command(author, version, about, long_about = None)]
    pub struct Args {
        /// The TOML configuration file; `config.example.toml` at the repository root is the
        /// template.
        #[arg(short, long)]
        config_path: PathBuf,
    }

    /// The decode step over runtime tensors serves uniform single-token decodes, which the
    /// engine keys and pads; every other step runs eagerly on candle.
    fn contract(model: &str) -> CaptureContract {
        CaptureContract::resolve(&[declaration()], &ModelDeclaration::new(model))
    }

    pub async fn run() -> anyhow::Result<()> {
        let args = Args::parse();
        let config: Config =
            load(&args.config_path).context("the configuration could not be loaded")?;
        let files = fetch(&config.model)?;
        let model_eos = eos_token_ids(&files.config)?;
        check_eos_token_ids(&config.engine.scheduler.eos_token_ids, &model_eos)?;
        let tokenizer = Tokenizer::from_file(&files.tokenizer).map_err(|error| {
            anyhow!(
                "the tokenizer at {} could not be loaded: {error}",
                files.tokenizer.display()
            )
        })?;
        // The padding dummies' KV is reserved for the process lifetime; what it costs is known
        // from the model's geometry before the pool or the device holds anything.
        let scheduler = &config.engine.scheduler;
        let kv_spec = kv_cache_spec(&files.config, scheduler.block_size, config.model.dtype)?;
        info!(
            dummies = PaddingReservation::dummies_for(scheduler.max_batch),
            bytes = PaddingReservation::cost_bytes(&kv_spec, scheduler.max_batch),
            "padding dummies' KV cost"
        );

        let (handle, handoff, engine) =
            Engine::spawn(&config.engine, &contract(config.model.id.as_str()))?;
        let executors = match spawn_ranks(
            &config.engine,
            &config.executor,
            &config.model,
            &files,
            handoff,
        ) {
            Ok(executors) => executors,
            Err(error) => {
                // The engine has nothing to serve; take it down before reporting. It has had
                // no control yet and no executor to lose, so it is there to be shut down.
                if handle.control.send(Control::Shutdown).is_err() {
                    error!("the engine thread exited before it was shut down");
                }
                engine.join();
                return Err(error.into());
            }
        };
        info!(ranks = executors.len(), "engine and executors running");

        let listener = TcpListener::bind(config.server.bind)
            .await
            .with_context(|| format!("cannot listen on {}", config.server.bind))?;
        let state = AppState {
            engine: handle,
            tokenizer: Arc::new(tokenizer),
            served: ServedModel::new(config.model.id.clone(), config.model.prompt_template),
            max_model_len: config.engine.scheduler.max_model_len,
            keep_alive: config.server.keep_alive,
            heartbeat_stale_after: config.server.heartbeat_stale_after,
        };
        run_server(listener, state, EngineThreads { engine, executors }).await
    }
}

#[cfg(feature = "cuda")]
use tracing_subscriber::fmt::init as init_tracing;

#[cfg(feature = "cuda")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    startup::run().await
}

#[cfg(not(feature = "cuda"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("CUDA support is disabled; rebuild with `--features cuda`")
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;

    use serde_json::json;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::Tokenizer;

    /// The beginning-of-sequence token, as Llama 3 spells it.
    pub(crate) const BOS: &str = "<|begin_of_text|>";

    /// A byte-level BPE tokenizer over every byte, with a few merges, built from the JSON a
    /// `tokenizer.json` holds: no file, no Hub. As the Llama 3 tokenizer does, it carries the BOS
    /// token and a post-processor that prepends it whenever special tokens are asked for.
    pub(crate) fn tokenizer() -> Arc<Tokenizer> {
        let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
        alphabet.sort_unstable();
        let mut vocab: Vec<(String, u32)> = alphabet
            .iter()
            .enumerate()
            .map(|(id, &c)| (c.to_string(), u32::try_from(id).unwrap()))
            .collect();
        let merges = ["h e", "l l", "Ġ w", "o r"];
        for merge in merges {
            let merged = merge.replace(' ', "");
            let id = u32::try_from(vocab.len()).unwrap();
            vocab.push((merged, id));
        }
        let bos_id = u32::try_from(vocab.len()).unwrap();
        let vocab: serde_json::Map<String, serde_json::Value> = vocab
            .into_iter()
            .map(|(token, id)| (token, json!(id)))
            .collect();
        let bos = json!({ "SpecialToken": { "id": BOS, "type_id": 0 } });
        let post_processor = json!({
            "type": "TemplateProcessing",
            "single": [bos, { "Sequence": { "id": "A", "type_id": 0 } }],
            "pair": [
                bos,
                { "Sequence": { "id": "A", "type_id": 0 } },
                bos,
                { "Sequence": { "id": "B", "type_id": 1 } }
            ],
            "special_tokens": { BOS: { "id": BOS, "ids": [bos_id], "tokens": [BOS] } }
        });
        let byte_level = json!({
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        });
        let spec = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [{
                "id": bos_id,
                "content": BOS,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }],
            "normalizer": null,
            "pre_tokenizer": byte_level,
            "post_processor": post_processor,
            "decoder": byte_level,
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": vocab,
                "merges": merges,
            }
        });
        Arc::new(serde_json::from_value(spec).expect("a byte-level BPE tokenizer"))
    }
}
