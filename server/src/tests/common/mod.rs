use crate::app_start;
use tokio::net::TcpListener;
use tokio::sync::OnceCell;

static SERVER: OnceCell<String> = OnceCell::const_new();

/// This test harness ensures only one server is spawned for all tests.
/// - It uses OnceCell to ensure the server is only started once and to store the server address.
/// - It binds to a random available port to avoid conflicts between test runs.
/// - It uses tokio::net::TcpListener for consistency with async Axum servers.
pub async fn spawn_server() -> &'static str {
    SERVER
        .get_or_init(|| async {
            let config_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("backends/vllm/src/tests/test_config.toml")
                .to_str()
                .unwrap()
                .to_string();

            // Bind to a random available port
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let server_address = format!("http://{}/v1", addr);

            // Spawn the server in the background
            tokio::spawn(async move { app_start(listener, config_path).await });

            // Wait for the server to start
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

            server_address
        })
        .await
}
