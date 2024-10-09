use clap::Parser;
use std::{
    env,
    sync::{atomic::AtomicU64, Arc},
};

#[cfg(feature = "vllm")]
use atoma_backends::{LlamaModel, LlmService};
use tokio::{net::TcpListener, sync::mpsc};

use server::{run_server, AppState};

pub mod api;
pub mod server;
pub mod stream;
#[cfg(test)]
pub mod tests;

// TODO: Add version path prefix, eg. `/v1` although maybe something along the lines of `/beta` would be more fitting?
/// The URL path to POST JSON for model chat completions.
pub const DEFAULT_SERVER_ADDRESS: &str = "0.0.0.0";
pub const DEFAULT_SERVER_PORT: &str = "8080";

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(short, long)]
    config_path: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Args::parse();

    // TODO: Write a clap cli for passing arguments
    let address =
        env::var("ATOMA_NODE_INFERENCE_SERVER_ADDRESS").unwrap_or(DEFAULT_SERVER_ADDRESS.into());
    let config_path = cli.config_path;
    let port = env::var("ATOMA_NODE_INFERENCE_SERVER_PORT").unwrap_or(DEFAULT_SERVER_PORT.into());
    let listener = TcpListener::bind(format!("{address}:{port}")).await?;

    let (llm_service_sender, llm_service_receiver) = mpsc::unbounded_channel();
    let (shutdown_signal_sender, shutdown_signal_receiver) = mpsc::channel(1);
    // TODO: Add model dispatcher
    let llm_service = LlmService::start::<LlamaModel, _>(
        llm_service_receiver,
        config_path,
        shutdown_signal_receiver,
    )
    .await
    .map_err(|e| anyhow::anyhow!("Failed to start `LlmService`, with error: {e}"))?;

    let join_handle = tokio::spawn(async move {
        llm_service
            .run()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to run `LlmService`, with error: {e}"))
    });

    let app_state = AppState {
        request_counter: Arc::new(AtomicU64::new(0)),
        llm_service_sender,
        shutdown_signal_sender,
        streaming_interval_in_millis: env::var("STREAMING_INTERVAL_IN_MILLIS")
            .map(|s| s.parse::<u64>().unwrap_or(100))
            .unwrap_or(100),
    };
    run_server(listener, app_state, join_handle).await
}
