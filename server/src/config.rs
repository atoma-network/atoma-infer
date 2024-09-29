use lazy_static::lazy_static;
use std::env;

lazy_static! {
    pub static ref CONFIG: Config = Config::new();
}

pub struct Config {
    pub server_address: String,
    pub server_port: String,
    pub chat_completions_path: String,
}

impl Config {
    fn new() -> Self {
        dotenvy::from_filename(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join(".env"),
        )
        .expect("Failed to load .env file from workspace root");

        // Check if deployment mode matches the build profile
        let deployment_mode = env_var("DEPLOYMENT_MODE");
        #[cfg(debug_assertions)]
        if deployment_mode != "development" {
            panic!("DEPLOYMENT_MODE must be set to 'development' in debug builds");
        }
        #[cfg(not(debug_assertions))]
        if deployment_mode != "production" {
            panic!("DEPLOYMENT_MODE must be set to 'production' in release builds");
        }

        Config {
            server_address: env_var("SERVER_ADDRESS"),
            server_port: env_var("SERVER_PORT"),
            chat_completions_path: env_var("CHAT_COMPLETIONS_PATH"),
        }
    }
}

fn env_var(key: &str) -> String {
    env::var(key).unwrap_or_else(|_| panic!("Environment variable '{}' not defined", key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    // Set environment variables for testing (only DEPLOYMENT_MODE is used)
    fn set_env_vars() {
        env::set_var("SERVER_ADDRESS", "127.0.0.1");
    }

    #[test]
    fn test_config_initialization() {
        set_env_vars();
        let config = Config::new();
        assert_eq!(config.server_address, "127.0.0.1");
    }

    #[test]
    #[should_panic(expected = "Environment variable 'MISSING_VAR' not defined")]
    fn test_missing_env_var() {
        env_var("MISSING_VAR");
    }
}
