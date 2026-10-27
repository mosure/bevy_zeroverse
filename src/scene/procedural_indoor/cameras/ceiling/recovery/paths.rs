//! Bounded path proposals for difficult mixed rigs. No admission threshold is
//! relaxed: every derived path uses its destination slot's calibration, motion
//! role and full clearance/content checks before joining the constraint solver.
use super::*;
use crate::scene::procedural_indoor::cameras::CameraMotion;

pub(super) struct Budget {
    poses: usize,
    paths: usize,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            poses: 512,
            paths: 16_384,
        }
    }
}
impl Budget {
    fn path(&mut self, scene: &mut IndoorManifest) -> bool {
        if self.paths == 0 {
            return false;
        }
        self.paths -= 1;
        scene.camera_diagnostics.recovery_path_proposals += 1;
        true
    }
}

// A rich midpoint often admits a short moving view where extending a path
// entirely forward loses the furniture at one endpoint. Fixed gaze retains
// framing; the requested path length, synchronized spacing and motion diversity
// are still enforced. Temporary stationary sampling never escapes this helper.
pub(super) fn centered(
    scene: &mut IndoorManifest,
    index: usize,
    fov: f32,
    coverage: &Coverage,
    rng: &mut rand_chacha::ChaCha8Rng,
    budget: &mut Budget,
) -> Option<IndoorCamera> {
    if scene.camera_settings.handheld.is_some()
        || !scene.camera_settings.camera_moves(scene.seed, index)
    {
        return None;
    }
    if budget.poses == 0 || budget.paths == 0 {
        return None;
    }
    budget.poses -= 1;
    let fraction = std::mem::replace(&mut scene.camera_settings.motion_fraction, 0.);
    let result = scene.sample_slots_with_budget(index, 1, rng, coverage, 32, Some(fov));
    scene.camera_settings.motion_fraction = fraction;
    result.ok()?;
    let pose = scene.cameras.pop()?;
    let min = scene
        .camera_settings
        .path_length_min
        .max(0.001)
        .min(scene.camera_settings.path_length_max);
    let max = scene.camera_settings.path_length_max;
    if max == 0. {
        return Some(pose);
    }
    for _ in 0..16 {
        if !budget.path(scene) {
            return None;
        }
        let length = rng.random_range(min.ln()..=max.ln()).exp();
        let angle = rng.random_range(0.0..std::f32::consts::TAU);
        let delta = Vec3::new(angle.cos(), 0., angle.sin()) * length;
        let mut camera = pose.clone();
        camera.start = pose.start - delta * 0.5;
        camera.end = pose.start + delta * 0.5;
        camera.motion = Some(CameraMotion {
            orientations: None,
            route: Vec::new(),
            control: [camera.start + delta / 3., camera.start + delta * 2. / 3.],
            target_end: camera.target,
            roll: [0.; 2],
        });
        if acceptable(scene, index, &camera, coverage) {
            return Some(camera);
        }
    }
    None
}

pub(super) fn reuse(
    scene: &mut IndoorManifest,
    coverage: &Coverage,
    prototypes: &[Prototype],
    pools: &mut [Vec<Candidate>],
    budget: &mut Budget,
) {
    let sources: Vec<_> = pools
        .iter()
        .flatten()
        .filter(|c| !c.camera.is_ceiling_mounted())
        .map(|c| c.camera.clone())
        .collect();
    for (index, prototype) in prototypes.iter().enumerate() {
        if scene.camera_settings.ceiling_mounted(scene.seed, index) {
            continue;
        }
        for source in &sources {
            if !budget.path(scene) {
                return;
            }
            if pools[index].len() >= 96 {
                break;
            }
            let mut camera = source.clone();
            camera.fisheye = prototype.0;
            camera.fov_degrees = prototype.1;
            if pools[index].iter().any(|c| c.camera == camera) {
                continue;
            }
            if !acceptable(scene, index, &camera, coverage) {
                continue;
            }
            let views = CHECK_TIMES
                .into_iter()
                .map(|t| coverage.view(&camera, scene.camera_aspect_ratio, t))
                .collect();
            pools[index].push(Candidate {
                track: Track::new(&camera),
                camera,
                views,
            });
        }
    }
}

pub(super) fn perturb(
    scene: &mut IndoorManifest,
    coverage: &Coverage,
    pools: &mut [Vec<Candidate>],
    budget: &mut Budget,
) {
    for (index, pool) in pools.iter_mut().enumerate() {
        if scene.camera_settings.handheld.is_some()
            || scene.camera_settings.ceiling_mounted(scene.seed, index)
            || pool.is_empty()
            || pool[0].camera.fisheye.is_some()
            || pool[0].camera.fov_degrees >= 70.
        {
            continue;
        }
        let sources: Vec<_> = pool.iter().map(|c| c.camera.clone()).collect();
        let primary: Vec<_> = sources
            .iter()
            .filter(|c| scene.in_primary_room(c.start, 0.))
            .collect();
        let mut rng = stream(scene.seed, 0x4c4f_4341_4c00 + index as u64);
        for _ in 0..192 {
            if !budget.path(scene) {
                return;
            }
            if pool.len() >= 64 {
                break;
            }
            let source = if !primary.is_empty() && rng.random_bool(0.75) {
                primary[rng.random_range(0..primary.len())]
            } else {
                &sources[rng.random_range(0..sources.len())]
            };
            let mut camera = source.clone();
            let center = (camera.start + camera.end) * 0.5;
            let rotation = Quat::from_rotation_y(rng.random_range(-0.35..0.35));
            let scale = rng.random_range(0.94..1.06);
            let offset = Vec3::new(
                rng.random_range(-0.65..0.65),
                rng.random_range(-0.35..0.35),
                rng.random_range(-0.65..0.65),
            );
            let point = |p| center + rotation * (p - center) * scale + offset;
            camera.start = point(camera.start);
            camera.end = point(camera.end);
            if let Some(m) = &mut camera.motion {
                m.control = m.control.map(point);
                for p in &mut m.route {
                    *p = point(*p);
                }
            }
            if !acceptable(scene, index, &camera, coverage) {
                continue;
            }
            let views = CHECK_TIMES
                .into_iter()
                .map(|t| coverage.view(&camera, scene.camera_aspect_ratio, t))
                .collect();
            pool.push(Candidate {
                track: Track::new(&camera),
                camera,
                views,
            });
        }
    }
}

pub(super) fn resample_intrinsics(
    scene: &mut IndoorManifest,
    coverage: &Coverage,
    prototypes: &[Prototype],
    pools: &mut [Vec<Candidate>],
    budget: &mut Budget,
) {
    let sources: Vec<_> = pools
        .iter()
        .flatten()
        .filter(|c| !c.camera.is_ceiling_mounted())
        .map(|c| c.camera.clone())
        .collect();
    if sources.is_empty() {
        return;
    }
    for (index, prototype) in prototypes.iter().enumerate().skip(1) {
        if scene.camera_settings.ceiling_mounted(scene.seed, index) || prototype.0.is_some() {
            continue;
        }
        let pdf = *scene.camera_settings.intrinsics_for(scene.seed, index);
        if pdf.fov_y_min_degrees == pdf.fov_y_max_degrees || pdf.spread == 0. {
            continue;
        }
        let mut rng = stream(scene.seed, 0x464f_5653_0000 + index as u64);
        for _ in 0..96 {
            if !budget.path(scene) {
                return;
            }
            let mut camera = sources[rng.random_range(0..sources.len())].clone();
            camera.fisheye = None;
            camera.fov_degrees = pdf.sample(&mut rng);
            scene.camera_diagnostics.recovery_intrinsic_draws += 1;
            if !acceptable(scene, index, &camera, coverage) {
                continue;
            }
            let views = CHECK_TIMES
                .into_iter()
                .map(|t| coverage.view(&camera, scene.camera_aspect_ratio, t))
                .collect();
            pools[index].push(Candidate {
                track: Track::new(&camera),
                camera,
                views,
            });
        }
    }
}

/// Reuse cannot turn a requested moving camera into a stationary view, even if
/// the requested minimum travel is zero. Explicit zero-length paths remain valid.
fn acceptable(
    scene: &IndoorManifest,
    index: usize,
    camera: &IndoorCamera,
    coverage: &Coverage,
) -> bool {
    !(scene.camera_settings.camera_moves(scene.seed, index)
        && scene.camera_settings.path_length_max > 0.
        && camera.motion.is_none()
        && camera.start == camera.end)
        && scene
            .camera_settings
            .accepts_motion(scene.seed, index, camera)
        && scene.camera_curve_clear(camera)
        && coverage.suitable(
            camera,
            super::super::super::sampling::requires_person(scene, index),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        cameras::intrinsics::IntrinsicsSettings, layout::IndoorLayout,
    };

    fn fixture() -> (
        IndoorManifest,
        Coverage,
        Vec<Prototype>,
        Vec<Vec<Candidate>>,
    ) {
        let scene =
            IndoorManifest::generate_with_humans(66_242_909, IndoorLayout::Mixed, 0.65, 1, 0.)
                .unwrap();
        let coverage = Coverage::new(&scene);
        let camera = scene.cameras[0].clone();
        let prototypes = vec![(None, camera.fov_degrees, 0.15, 0.); 2];
        let pools = vec![
            vec![Candidate {
                track: Track::new(&camera),
                camera,
                views: Vec::new(),
            }],
            vec![],
        ];
        (scene, coverage, prototypes, pools)
    }

    #[test]
    fn fixed_intrinsics_and_exhausted_budgets_leave_the_requested_policy_intact() {
        let (mut scene, coverage, prototypes, mut pools) = fixture();
        for intrinsics in [
            IntrinsicsSettings {
                fov_y_min_degrees: 80.,
                fov_y_max_degrees: 80.,
                ..Default::default()
            },
            IntrinsicsSettings {
                spread: 0.,
                ..Default::default()
            },
        ] {
            scene.camera_settings.intrinsics = intrinsics;
            let before = scene.clone();
            resample_intrinsics(
                &mut scene,
                &coverage,
                &prototypes,
                &mut pools,
                &mut Budget::default(),
            );
            assert_eq!(scene, before);
            assert!(pools[1].is_empty());
        }
        scene.camera_settings.intrinsics = Default::default();
        let before = scene.clone();
        let mut budget = Budget { poses: 0, paths: 0 };
        assert!(centered(
            &mut scene,
            1,
            80.,
            &coverage,
            &mut stream(42, 1),
            &mut budget
        )
        .is_none());
        resample_intrinsics(&mut scene, &coverage, &prototypes, &mut pools, &mut budget);
        reuse(&mut scene, &coverage, &prototypes, &mut pools, &mut budget);
        perturb(&mut scene, &coverage, &mut pools, &mut budget);
        assert_eq!(scene, before);
        assert!(pools[1].is_empty());
    }

    #[test]
    fn variable_intrinsics_qualify_actual_lenses_without_replacing_geometry_or_motion_roles() {
        let (mut scene, coverage, prototypes, mut pools) = fixture();
        let before = scene.clone();
        resample_intrinsics(
            &mut scene,
            &coverage,
            &prototypes,
            &mut pools,
            &mut Budget::default(),
        );
        assert!(!pools[1].is_empty());
        assert_eq!(scene.camera_diagnostics.recovery_intrinsic_draws, 96);
        for c in &pools[1] {
            assert!(scene
                .camera_settings
                .intrinsics
                .contains(c.camera.fov_degrees));
            assert_eq!(c.camera.fisheye, None);
            assert!(acceptable(&scene, 1, &c.camera, &coverage));
            assert_eq!(c.views.len(), CHECK_TIMES.len());
        }
        let mut fixed = pools[1][0].camera.clone();
        fixed.end = fixed.start;
        fixed.motion = None;
        scene.camera_settings.path_length_min = 0.;
        assert!(!acceptable(&scene, 1, &fixed, &coverage));
        scene.camera_settings = before.camera_settings.clone();
        scene.camera_diagnostics = before.camera_diagnostics.clone();
        assert_eq!(scene, before);
    }
}
