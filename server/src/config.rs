use lazy_static::lazy_static;
use std::env;

lazy_static! {
    pub static ref CONFIG: Config = Config::new();
}

pub struct Config {
    pub server_address: String,
    pub server_port: String,
    pub server_api_root_path: String,
    pub server_api_base: String,
    pub chat_completions_path: String,
}

fn env_var(key: &str) -> String {
    env::var(key).unwrap_or_else(|_| panic!("Environment variable '{}' not defined", key))
}

fn load_env_vars() {
    dotenvy::from_filename(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join(".env"),
    )
    .expect("Failed to load .env file from workspace root");
}

impl Config {
    fn new() -> Self {
        load_env_vars();

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

        let server_address = env_var("SERVER_ADDRESS");
        let server_port = env_var("SERVER_PORT");
        let server_api_root_path = "/v1";
        let server_api_base = format!(
            "http://{}:{}/{}",
            server_address, server_port, server_api_root_path
        );
        let chat_completions_path = "/chat/completions";

        Config {
            server_address,
            server_port,
            server_api_root_path: server_api_root_path.to_string(),
            server_api_base,
            chat_completions_path: chat_completions_path.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_initialization() {
        load_env_vars();
        let config = Config::new();
        assert_eq!(config.server_address, env_var("SERVER_ADDRESS"));
    }
}
