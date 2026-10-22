//! A view must be active before Bevy prepares its visibility and shadow cascades.
//! Late deactivation is safe and avoids duplicate draws during readback; late
//! activation would extract a view with missing directional-light cascades.
use super::{CaptureFailure, CaptureProgress, CaptureReadiness, SamplerState};
use crate::{app::BevyZeroverseConfig, camera::ZeroverseCamera, io::image_copy::ImageCopier};
use bevy::{camera::CameraUpdateSystems, prelude::*};

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CaptureCameraGate;

#[cfg(not(target_arch = "wasm32"))]
mod submitted;

/// Read the shared submission stamp at extraction time, not the earlier main
/// schedule. A new request or an interactive camera must keep rendering.
pub(crate) fn render_active(
    args: &BevyZeroverseConfig,
    active: bool,
    copier: Option<&ImageCopier>,
) -> bool {
    active
        && !(cfg!(not(target_arch = "wasm32"))
            && args.headless
            && !args.editor
            && copier
                .is_some_and(|c| c.requested_id() != 0 && c.submitted_id() == c.requested_id()))
}

pub(super) fn configure(app: &mut App) {
    #[cfg(not(target_arch = "wasm32"))]
    submitted::install(app);
    app.add_systems(
        PostUpdate,
        gate::<true>
            .in_set(CaptureCameraGate)
            .after(super::readiness::update)
            .before(CameraUpdateSystems)
            .before(bevy::camera::visibility::VisibilitySystems::CheckVisibility)
            .before(bevy::light::SimulationLightSystems::UpdateDirectionalLightCascades),
    );
    app.add_systems(Last, gate::<false>.in_set(CaptureCameraGate));
}

fn gate<const ALLOW_ACTIVATION: bool>(
    args: Res<BevyZeroverseConfig>,
    state: Res<SamplerState>,
    capture: Res<CaptureProgress>,
    readiness: Res<CaptureReadiness>,
    failure: Res<CaptureFailure>,
    mut cameras: Query<(&mut Camera, &ImageCopier), With<ZeroverseCamera>>,
) {
    if !args.headless || args.editor {
        return;
    }
    for (mut camera, copier) in &mut cameras {
        camera.is_active = (ALLOW_ACTIVATION || camera.is_active)
            && state.enabled
            && capture.ovoxel_wait_started.is_none()
            && readiness.scene_ready()
            && failure.0.is_none()
            && capture.pending.is_none_or(|id| copier.submitted_id() != id);
    }
}

#[cfg(test)]
mod tests;
