//! Extend promising partial rigs on their actual navigable floor level.
//! Independent proposals can fill every slot with ground-floor paths while the
//! only shared ceiling/room reference is above a mezzanine. Repeating global
//! sampling or cloning one rare upper-floor path cannot fill a diverse rig.
use super::*;

pub(super) fn expand(
    scene: &mut IndoorManifest,
    coverage: &Coverage,
    prototypes: &[Prototype],
    pools: &mut [Vec<Candidate>],
    solver: &solver::Solver,
    contract: MixedRigContract,
    budget: &mut paths::Budget,
) {
    let policy = scene.camera_settings.multiview.as_ref().unwrap().clone();
    let references = solver.ranked_references(pools[0].len(), pools.len());
    for (rank, &reference_index) in references.iter().take(2).enumerate() {
        let reference = pools[0][reference_index].camera.clone();
        if reference.is_ceiling_mounted() || scene.camera_settings.handheld.is_some() {
            continue;
        }
        let reference_track = Track::new(&reference);
        // Reuse *compatible* paths, not merely individually clear paths. Several
        // slots may initially contain the same rare upper-level path; deform it
        // independently and let all-pair spacing reject coincident trajectories.
        let mut sources = Vec::new();
        let compatible = solver.compatible_candidates(reference_index);
        for &(slot, candidate) in &compatible {
            let camera = &pools[slot][candidate].camera;
            if !camera.is_ceiling_mounted()
                && !sources.iter().any(|c: &IndoorCamera| {
                    c.start == camera.start && c.end == camera.end && c.target == camera.target
                })
            {
                sources.push(camera.clone());
            }
        }
        if sources.is_empty() {
            continue;
        }
        for index in 1..pools.len() {
            if scene.camera_settings.ceiling_mounted(scene.seed, index) {
                continue;
            }
            let initial = pools[index].len();
            let local: Vec<_> = compatible
                .iter()
                .filter(|(slot, _)| *slot == index)
                .map(|&(_, i)| i)
                .collect();
            let mut rng = stream(
                scene.seed,
                0x434f_4e44_0000 + index as u64 * 2 + rank as u64,
            );
            for trial in 0..192 {
                if pools[index].len() >= initial + 24 || !budget.path(scene) {
                    break;
                }
                let mut camera = if !local.is_empty() && trial % 2 == 0 {
                    let source = &pools[index][local[rng.random_range(0..local.len())]].camera;
                    // Two synchronized trajectories can occupy different
                    // sections of one narrow aisle without leaving free space.
                    let fraction = rng.random_range(0.51..0.90);
                    let start = if trial % 6 == 0 {
                        0.
                    } else if trial % 6 == 2 {
                        1. - fraction
                    } else {
                        rng.random_range(0.0..=1. - fraction)
                    };
                    let Some(camera) = paths::section(source, start, start + fraction) else {
                        continue;
                    };
                    camera
                } else {
                    paths::nearby_path(&sources[rng.random_range(0..sources.len())], &mut rng)
                };
                camera.fisheye = prototypes[index].lens;
                camera.fov_degrees = prototypes[index].fov_degrees;
                let pdf = scene.camera_settings.intrinsics_for(scene.seed, index);
                if trial >= 96
                    && camera.fisheye.is_none()
                    && pdf.fov_y_min_degrees != pdf.fov_y_max_degrees
                    && pdf.spread > 0.
                {
                    camera.fov_degrees = pdf.sample(&mut rng);
                    scene.camera_diagnostics.recovery_intrinsic_draws += 1;
                }
                scene
                    .camera_settings
                    .prepare_motion(scene.seed, index, &mut camera, &mut rng);
                let track = Track::new(&camera);
                if !contract.accepts_geometry(
                    &policy,
                    &[&reference, &camera],
                    &[&reference_track, &track],
                    false,
                ) || !paths::acceptable(scene, index, &camera, coverage)
                {
                    continue;
                }
                let views = CHECK_TIMES
                    .into_iter()
                    .map(|t| coverage.view(&camera, scene.camera_aspect_ratio, t))
                    .collect();
                pools[index].push(Candidate {
                    camera,
                    track,
                    views,
                });
            }
        }
    }
}
