use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::draft::Draft;
use crate::id::MaterialId;

/// A compound clip. `draft` is the nested document, when the file inlines it.
/// `combination_type` names AI-music roles. It does not define nesting.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct MaterialDraft {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub formula_id: String,
    #[serde(default)]
    pub combination_id: String,
    #[serde(default)]
    pub combination_type: String,
    #[serde(default)]
    pub precompile_combination: bool,
    #[serde(default)]
    pub draft: Option<Box<Draft>>,
    #[serde(default)]
    pub draft_file_path: String,
    #[serde(default)]
    pub draft_cover_path: String,
    #[serde(default)]
    pub draft_config_path: String,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}
