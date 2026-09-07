//! The server's configuration, read from one TOML file with `ATOMA_` environment overrides: the
//! engine's, the executor's, the model's and the server's own, each validated on its own terms
//! and the whole refused when any part is.

use std::net::SocketAddr;
use std::time::Duration;

use atoma_core::engine::EngineConfig;
use atoma_engine::config::{ExecutorConfig, ModelConfig};
use serde::{Deserialize, Serialize};
use validator::Validate;

/// Everything the server process is built from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[validate(nested)]
    pub engine: EngineConfig,
    #[validate(nested)]
    pub executor: ExecutorConfig,
    #[validate(nested)]
    pub model: ModelConfig,
    #[validate(nested)]
    pub server: ServerConfig,
}

/// What the HTTP server itself is built from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// The address and port the server listens on.
    pub bind: SocketAddr,
    /// How often a keep-alive comment is written to a stream with nothing to say.
    #[serde(rename = "keep_alive_millis", with = "millis")]
    pub keep_alive: Duration,
    /// How old the engine thread's heartbeat may be before the node reports itself unhealthy.
    #[serde(rename = "heartbeat_stale_millis", with = "millis")]
    pub heartbeat_stale_after: Duration,
}

/// A duration in memory, milliseconds on the wire.
mod millis {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S: Serializer>(
        duration: &Duration,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        u64::try_from(duration.as_millis())
            .unwrap_or(u64::MAX)
            .serialize(serializer)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Duration, D::Error> {
        u64::deserialize(deserializer).map(Duration::from_millis)
    }
}

#[cfg(test)]
mod tests {
    // The closure `Jail::expect_with` takes returns `figment::Error`, which is large. The
    // signature is figment's, so the size is not ours to change.
    #![allow(clippy::result_large_err)]

    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use atoma_core::config::{load, ConfigError};
    use atoma_engine::config::Dtype;
    use figment::Jail;

    use super::Config;

    /// The repository's example configuration, loaded inside a jail that restores whatever
    /// environment a test sets.
    fn in_a_jail_with_the_example(test: impl FnOnce(&mut Jail, &Path)) {
        let example = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../config.example.toml")
            .canonicalize()
            .expect("the example configuration is at the repository root");
        let source = fs::read_to_string(example).expect("the example configuration reads");
        Jail::expect_with(|jail| {
            jail.create_file("config.toml", &source)?;
            test(jail, Path::new("config.toml"));
            Ok(())
        });
    }

    #[test]
    fn the_example_configuration_loads_whole() {
        in_a_jail_with_the_example(|_, path| {
            let config: Config = load(path).expect("the example configuration loads");
            assert_eq!(config.engine.scheduler.block_size.get(), 16);
            assert_eq!(config.executor.ranks.len(), 1);
            assert_eq!(config.executor.staging_depth.get(), 2);
            assert_eq!(config.model.id.as_str(), "meta-llama/Llama-3.2-1B-Instruct");
            assert_eq!(config.server.bind.port(), 8080);
            assert_eq!(config.server.keep_alive, Duration::from_millis(100));
            assert_eq!(config.server.heartbeat_stale_after, Duration::from_secs(1));
        });
    }

    #[test]
    fn an_environment_variable_overrides_a_field_of_any_part() {
        in_a_jail_with_the_example(|jail, path| {
            jail.set_env("ATOMA_SERVER__BIND", "127.0.0.1:9000");
            jail.set_env("ATOMA_MODEL__DTYPE", "f16");
            jail.set_env("ATOMA_EXECUTOR__STAGING_DEPTH", "4");
            let config: Config = load(path).expect("the overridden configuration loads");
            assert_eq!(config.server.bind.to_string(), "127.0.0.1:9000");
            assert_eq!(config.model.dtype, Dtype::F16);
            assert_eq!(config.executor.staging_depth.get(), 4);
        });
    }

    #[test]
    fn a_part_that_does_not_hold_together_refuses_the_whole() {
        in_a_jail_with_the_example(|jail, path| {
            jail.set_env("ATOMA_ENGINE__SCHEDULER__MAX_REQUESTS", "4");
            let error = load::<Config>(path).expect_err("a batch over the slab cannot load");
            assert!(matches!(error, ConfigError::Invalid { .. }), "{error}");
            assert!(error.to_string().contains("max_batch"), "{error}");
        });
    }
}
