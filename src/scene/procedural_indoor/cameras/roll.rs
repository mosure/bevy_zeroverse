//! Sensor bank is independent of where the rig can find free space and a common view.
use super::{
    coverage::{Coverage, VisibilityView},
    multiview::{pair_samples_cached, CameraPairOverlap, OverlapGraph, CHECK_TIMES},
    sampling::requires_person,
    CameraMotion, CameraSettings, IndoorCamera, IndoorManifest,
};
use bevy::prelude::*;
use rand::Rng;

mod distribution;
pub use distribution::RollDistribution;
const SEARCH_BANK: f32 = 0.09; // radians at a mixed-rig search knot

impl CameraSettings {
    // Preserve the hard visibility graph while fitting gravity-relative framing.
    // Handheld increments are defined in the rolled camera frame before placement.
    fn fit_roll_after_placement(&self) -> bool {
        self.handheld.is_none() && self.multiview.is_some()
    }

    fn slot_roll(&self, seed: u64, index: usize) -> f32 {
        let mut angle = self.roll.sample_degrees(seed, index).to_radians();
        if self.handheld.is_none() && self.camera_moves(seed, index) && self.roll.max_degrees > 0. {
            angle *= 1. - self.roll.drift() / self.roll.max_degrees.to_radians();
        }
        angle
    }

    /// Bind independent and handheld poses before their path is admitted.
    pub(super) fn bind_room_roll(&self, seed: u64, index: usize, camera: &mut IndoorCamera) {
        if camera.is_ceiling_mounted() || self.fit_roll_after_placement() {
            return;
        }
        let roll = self.slot_roll(seed, index);
        if roll == camera.roll_radians {
            return;
        }
        if let Some(CameraMotion {
            orientations: Some([start, end]),
            control,
            route,
            ..
        }) = &mut camera.motion
        {
            // Reused handheld paths store world-space poses. Rotate the whole
            // path about the initial optical axis to preserve metric increments
            // in the sensor frame, including when the path plays in reverse.
            let twist =
                *start * Quat::from_rotation_z(roll - camera.roll_radians) * start.inverse();
            let rotate_point = |p| camera.start + twist * (p - camera.start);
            camera.end = rotate_point(camera.end);
            *control = control.map(rotate_point);
            for p in route {
                *p = rotate_point(*p);
            }
            *start = (twist * *start).normalize();
            *end = (twist * *end).normalize();
        }
        camera.roll_radians = roll;
    }

    /// Look-at trajectories may gently bank around the persistent mount offset.
    pub(super) fn sample_roll_drift(&self, rng: &mut impl Rng) -> [f32; 2] {
        let limit = self.roll.support_degrees().to_radians();
        if self.fit_roll_after_placement() {
            // Preserve the constructive search stream. Even at zero amplitude,
            // the next position or lens draw must not depend on the roll setting.
            let scale = limit.min(SEARCH_BANK) / SEARCH_BANK;
            return std::array::from_fn(|_| rng.random_range(-SEARCH_BANK..SEARCH_BANK) * scale);
        }
        let range = self.roll.drift();
        std::array::from_fn(|_| rng.random_range(-1.0..1.0) * range)
    }
}

/// Try the preferred bank first. A tight reference or peer edge may require a
/// smaller angle. Previously accepted camera positions and paths are never lost.
pub(super) fn fit_to_rig(scene: &mut IndoorManifest) {
    let settings = &scene.camera_settings;
    let support = settings.roll.support_degrees().to_radians();
    if !settings.fit_roll_after_placement()
        || support == 0.
        || scene.cameras.iter().all(IndoorCamera::is_ceiling_mounted)
    {
        return;
    }
    let coverage = Coverage::new(scene);
    let aspect = scene.camera_aspect_ratio;
    let observed = |c: &IndoorCamera| -> Vec<VisibilityView> {
        if c.start == c.end && c.motion.is_none() {
            return vec![coverage.view(c, aspect, 0.); CHECK_TIMES.len()];
        }
        CHECK_TIMES
            .into_iter()
            .map(|t| coverage.view(c, aspect, t))
            .collect()
    };
    let mut views: Vec<_> = scene.cameras.iter().map(observed).collect();
    let peers = scene
        .camera_coherence_policy()
        .filter(|_| scene.camera_diagnostics.peer_search_succeeded);
    let connected = settings
        .multiview
        .as_ref()
        .is_some_and(|p| p.graph == OverlapGraph::Connected);
    let drift_scale = settings.roll.drift() / support.min(SEARCH_BANK);
    for index in 0..scene.cameras.len() {
        let source = scene.cameras[index].clone();
        if source.is_ceiling_mounted() {
            continue;
        }
        let desired = settings.slot_roll(scene.seed, index);
        for weight in [1., 0.5, 0.25] {
            let mut candidate = source.clone();
            candidate.roll_radians = weight * desired;
            if let Some(m) = &mut candidate.motion {
                for r in &mut m.roll {
                    *r *= 1. + weight * (drift_scale - 1.);
                }
            }
            if !scene.camera_curve_clear(&candidate)
                || !coverage.suitable(&candidate, requires_person(scene, index))
            {
                continue;
            }
            let candidate_views = observed(&candidate);
            if let Some(policy) = peers.filter(|_| index != 0) {
                if (1..views.len())
                    .filter(|&other| other != index)
                    .any(|other| {
                        let floor =
                            policy.preferred_floor(&candidate, &scene.cameras[other], aspect);
                        !policy.supports(&coverage, &candidate_views, &views[other], floor)
                    })
                {
                    continue;
                }
            }
            scene.cameras[index] = candidate;
            let accepted = if connected {
                scene.accepts_camera_visibility()
            } else {
                // Only edges incident on this slot changed. Each test uses the
                // actual calibrated frame, including fisheye and shifted centers.
                let accepts = |peer: usize| {
                    let (a, b) = if index == 0 {
                        (&candidate_views, &views[peer])
                    } else {
                        (&views[0], &candidate_views)
                    };
                    scene.accepts_camera_pair(&CameraPairOverlap {
                        reference: 0,
                        camera: peer,
                        samples: pair_samples_cached(
                            &coverage,
                            &scene.cameras[0],
                            a,
                            &scene.cameras[peer],
                            b,
                        ),
                    })
                };
                if index == 0 {
                    (1..views.len()).all(accepts)
                } else {
                    accepts(index)
                }
            };
            if accepted {
                views[index] = candidate_views;
                break;
            }
            scene.cameras[index] = source.clone();
        }
    }
}

/// Signed angle of camera right above the world horizon (local +Z roll).
/// The horizon has no unique direction when the optical axis is vertical.
pub fn horizon_roll_radians(rotation: Quat) -> Option<f32> {
    let right_y = (rotation * Vec3::X).y;
    let up_y = (rotation * Vec3::Y).y;
    (right_y * right_y + up_y * up_y > 1e-10).then(|| right_y.atan2(up_y))
}

#[cfg(test)]
mod tests;
