//! Version-1 raw pose audit DTOs. Keep the example self-contained when packaged;
//! the publication tool validates this wire schema before measuring these traces.
use serde::Serialize;
use serde_json::Value;
pub const TIMES: usize = 33;
#[derive(Serialize)]
pub struct Path {
    pub positions: Vec<[f64; 3]>,
    pub forwards: Vec<[f64; 3]>,
    pub target: [f64; 3],
    pub fov_degrees: f64,
    pub length_m: f64,
    pub fisheye: Value,
    pub ceiling_mounted: bool,
}
#[derive(Serialize)]
pub struct Group {
    pub seed: u64,
    pub baseline: f64,
    pub camera_count: usize,
    pub error: Option<String>,
    pub cameras: Vec<Path>,
    pub policy: Value,
    pub geometry: Value,
    pub reference_overlap: Value,
    pub diagnostics: Value,
    pub mixed_rig_contract: Value,
    pub elapsed_seconds: f64,
}
