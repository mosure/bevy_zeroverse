//! Identical probe rays and filtering with browser event-loop yield points.
//! The synchronous/native paths retain their shared pool and asset ordering.
use super::*;
use crate::scene::procedural_indoor::preparation::cooperate;

pub(crate) async fn build_async(scene: &IndoorManifest, transport: &BakeScene) -> [Image; 2] {
    stencil::convolve_async(&trace_async(scene, transport).await).await
}

pub(crate) async fn trace_async(scene: &IndoorManifest, transport: &BakeScene) -> Vec<Vec3> {
    let trace = ProbeTrace::new(scene, transport, transport.reflection_origin(scene));
    let mut pixels = Vec::with_capacity((6 * SIZE * SIZE) as usize);
    for face in 0..6 {
        for y in 0..SIZE {
            trace.row(face, y, &mut pixels);
            cooperate().await;
        }
    }
    pixels
}
