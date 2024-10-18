use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct ModelInfo {
    pub name: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}
