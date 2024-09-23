pub mod llama;

use std::path::PathBuf;

/// Helper function to download safetensors from the HF API
pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> candle_core::Result<Vec<PathBuf>> {
    let json_file = match repo.get(json_file) {
        Ok(path) => path,
        Err(e) => candle_core::bail!("Failed to get json file from HF API, with error: {e}"),
    };
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value = match serde_json::from_reader(&json_file) {
        Ok(json) => json,
        Err(e) => candle_core::bail!("Failed to deserialize json file, with error: {e}"),
    };
    let weight_map = match json.get("weight_map") {
        None => candle_core::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => candle_core::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(candle_core::Error::wrap))
        .collect::<candle_core::Result<Vec<_>>>()
}
