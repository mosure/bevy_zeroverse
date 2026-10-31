//! Bounded exploration of clear ceiling patches shared by mounted slots.
//! Sampling guidance only: destination optics, complete swept clearance and
//! downward content are requalified before the rig solver sees each proposal.
use super::*;

pub(super) fn expand(
    scene: &mut IndoorManifest,
    coverage: &Coverage,
    space: &MountSpace,
    prototypes: &[Prototype],
    pools: &mut [Vec<Candidate>],
    budget: &mut paths::Budget,
) {
    let sources: Vec<_> = pools
        .iter()
        .flatten()
        .filter(|c| c.camera.is_ceiling_mounted())
        .map(|c| c.camera.clone())
        .collect();
    if sources.is_empty() {
        return;
    }
    for (index, prototype) in prototypes.iter().enumerate() {
        if !scene.camera_settings.ceiling_mounted(scene.seed, index) {
            continue;
        }
        let mut rng = stream(scene.seed, 0x4d4f_554e_5400 + index as u64);
        for trial in 0..192 {
            if pools[index].len() >= 64 {
                break;
            }
            if !budget.path(scene) {
                return;
            }
            let source = &sources[rng.random_range(0..sources.len())];
            let mount = source.ceiling_mount.as_ref().unwrap();
            let offset = if trial < sources.len() {
                Vec2::ZERO
            } else {
                Vec2::new(rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0))
                    * rng.random_range(0.05..=1.25)
            };
            let mut prior = *prototype;
            prior.2 = mount.standoff_m;
            let mut camera =
                mounted_at(scene, index, &prior, mount.surface.xz() + offset, &mut rng);
            if scene.camera_settings.camera_moves(scene.seed, index) && source.motion.is_some() {
                let normal = camera.ceiling_mount.as_ref().unwrap().inward_normal;
                let delta = source.end - source.start;
                super::super::trajectory::set_rail(&mut camera, delta - normal * normal.dot(delta));
            }
            if pools[index].iter().any(|c| c.camera == camera)
                || !space.fit_rail(scene, index, &mut camera, &mut rng)
                || !scene
                    .camera_settings
                    .accepts_motion(scene.seed, index, &camera)
            {
                continue;
            }
            let views: Vec<_> = CHECK_TIMES
                .into_iter()
                .map(|t| coverage.view(&camera, scene.camera_aspect_ratio, t))
                .collect();
            if !super::super::overview::suitable(scene, &camera, &views) {
                continue;
            }
            pools[index].push(Candidate {
                track: Track::new(&camera),
                camera,
                views,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::IndoorLayout;

    #[test]
    fn shared_mounting_patches_requalify_each_slots_calibration_and_motion() {
        let mut scene =
            IndoorManifest::generate_with_humans(16, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let policy = CameraSettings {
            ceiling_mount: Some(CeilingCameraSettings {
                motion_fraction: 1.,
                ..default()
            }),
            ..default()
        };
        scene.resample_cameras(1, policy, 1.).unwrap();
        let camera = scene.cameras.pop().unwrap();
        let coverage = Coverage::for_ceiling_search(&scene);
        let space = MountSpace::new(&scene);
        let prototypes: Vec<_> = (0..3)
            .map(|i| {
                (
                    scene.camera_settings.lens_for(scene.seed, i),
                    110. + i as f32 * 5.,
                    0.20,
                    i as f32 * 0.8,
                )
            })
            .collect();
        let mut pools = vec![
            vec![Candidate {
                track: Track::new(&camera),
                camera,
                views: Vec::new(),
            }],
            vec![],
            vec![],
        ];
        let before = scene.clone();
        expand(
            &mut scene,
            &coverage,
            &space,
            &prototypes,
            &mut pools,
            &mut paths::Budget::default(),
        );
        for index in 1..3 {
            assert!(!pools[index].is_empty());
            for candidate in &pools[index] {
                let camera = &candidate.camera;
                let mount = camera.ceiling_mount.as_ref().unwrap();
                assert_eq!(camera.fisheye, prototypes[index].0);
                assert_eq!(camera.fov_degrees, prototypes[index].1);
                assert_eq!(
                    mount.rotation,
                    Transform::IDENTITY
                        .looking_to(mount.inward_normal, Vec3::Z)
                        .rotation
                        * Quat::from_rotation_z(prototypes[index].3)
                );
                assert!(scene
                    .camera_settings
                    .accepts_motion(scene.seed, index, camera));
                assert!(space.clear(&scene, camera));
                assert!(super::super::super::overview::suitable(
                    &scene,
                    camera,
                    &candidate.views
                ));
            }
        }
        assert!(scene.camera_diagnostics.recovery_path_proposals <= 192 * 3);
        scene.camera_diagnostics = before.camera_diagnostics.clone();
        assert_eq!(scene, before);
    }
}
