//! Room-camera proposals for a joint mounted/walkthrough rig. Local proposals
//! reuse navigable paths, not accepted camera identities or admission decisions.
//! A fixed slot FOV must participate in clearance and content screening. Applying
//! it only after this search would preferentially return invalid wide-lens poses
//! for narrow slots, repeatedly exhausting the outer rig's placement budget.
use super::super::viewpoint::MAX_EYE_HEIGHT_M;
use super::*;
use crate::scene::procedural_indoor::{
    cameras::{coverage::Coverage, multiview},
    layout::{IndoorManifest, CAMERA_CLEARANCE},
};
use rand::{seq::IteratorRandom, Rng};

/// When independent origins repeatedly fail in a cramped rig, spend the
/// remaining joint search on a larger variety of paths and parent views.
/// Each accepted pose still passes the same clearance and visibility tests.
pub(super) fn origin_attempts(scene: &IndoorManifest) -> usize {
    if scene.generator_version < 226 {
        return 32;
    }
    match scene.camera_diagnostics.reference_proposals {
        0..16_384 => 32,
        16_384..24_576 => 16,
        _ => 8,
    }
}

pub(super) fn propose(
    scene: &mut IndoorManifest,
    index: usize,
    rng: &mut rand_chacha::ChaCha8Rng,
    coverage: &Coverage,
    fixed_fov: Option<f32>,
    independent: bool,
    prefer_primary: bool,
) -> Option<IndoorCamera> {
    // Mounted views are physically restricted to the primary room. When room
    // cameras may also use neighboring rooms, keep a mixture of primary-room
    // and unrestricted proposals: otherwise useful primary-room wide views can
    // be drowned out by easy but disconnected views in adjacent rooms.
    let primary = scene.camera_settings.primary_room;
    if !primary && prefer_primary && rng.random_bool(0.5) {
        scene.camera_settings.primary_room = true;
    }
    let result = propose_inner(scene, index, rng, coverage, fixed_fov, independent);
    scene.camera_settings.primary_room = primary;
    result
}

fn propose_inner(
    scene: &mut IndoorManifest,
    index: usize,
    rng: &mut rand_chacha::ChaCha8Rng,
    coverage: &Coverage,
    fixed_fov: Option<f32>,
    independent: bool,
) -> Option<IndoorCamera> {
    if let Some(policy) = scene
        .camera_settings
        .multiview
        .as_ref()
        .filter(|_| !independent && rng.random_bool(0.5))
    {
        let reference = scene
            .cameras
            .iter()
            .filter(|c| !c.is_ceiling_mounted())
            .choose(rng);
        if let Some(reference) = reference {
            let support = scene.envelope.as_ref().map_or_else(
                || scene.floor_height(reference.start.xz()),
                |e| e.support_height(reference.start),
            );
            let floor = support + 0.70;
            let ceiling = (scene.ceiling_height(reference.start.xz())
                - scene.program.as_ref().map_or(0.5, |p| p.light_drop)
                - 0.04
                - CAMERA_CLEARANCE)
                .min(support + MAX_EYE_HEIGHT_M);
            let local = multiview::MultiViewSettings {
                min_reference_baseline: 0.,
                ..policy.clone()
            };
            let mut camera = multiview::propose(
                reference,
                reference.path_length().max(0.001),
                policy.max_baseline.min(2.).max(policy.min_baseline),
                Vec2::new(floor - reference.start.y, ceiling - reference.start.y),
                &local,
                &scene.camera_settings,
                rng,
            );
            if let Some(fov) = fixed_fov {
                camera.fov_degrees = fov;
            }
            scene
                .camera_settings
                .prepare_motion(scene.seed, index, &mut camera, rng);
            let require_person = super::super::sampling::requires_person(scene, index);
            return (scene
                .camera_settings
                .accepts_motion(scene.seed, index, &camera)
                && scene.camera_curve_clear(&camera)
                && coverage.suitable(&camera, require_person))
            .then_some(camera);
        }
    }
    scene
        .sample_slots_with_budget(index, 1, rng, coverage, origin_attempts(scene), fixed_fov)
        .ok()?;
    scene.cameras.pop()
}
