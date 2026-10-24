//! Room-camera proposals for a joint mounted/walkthrough rig. Local proposals
//! reuse navigable paths, not accepted camera identities or admission decisions.
use super::*;
use crate::scene::procedural_indoor::{
    cameras::{coverage::Coverage, multiview},
    layout::{IndoorManifest, CAMERA_CLEARANCE},
};
use rand::{seq::IteratorRandom, Rng};

pub(super) fn propose(
    scene: &mut IndoorManifest,
    index: usize,
    rng: &mut rand_chacha::ChaCha8Rng,
    coverage: &Coverage,
) -> Option<IndoorCamera> {
    if let Some(policy) = scene
        .camera_settings
        .multiview
        .as_ref()
        .filter(|_| rng.random_bool(0.5))
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
                .min(support + 3.25);
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
                &scene.camera_settings.intrinsics,
                rng,
            );
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
        .sample_slots_with_budget(index, 1, rng, coverage, 32)
        .ok()?;
    scene.cameras.pop()
}
