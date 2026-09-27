#![cfg(not(target_arch = "wasm32"))]
#![recursion_limit = "256"]
#[path = "support/optical_flow.rs"]
mod reference;

#[test]
#[ignore = "requires native GPU; analytic flow, deformation, occlusion and sequence reset"]
fn capture_pairs_track_rigid_camera_vertex_and_skeletal_motion() {
    reference::run_validation();
}
