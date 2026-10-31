//! Bounded constraint search over reusable, individually qualified paths.
//! Greedy prefix search can strand a narrow slot behind an unsuitable wide
//! reference. Separate path feasibility from rig compatibility so references
//! can change without rebuilding every narrow-camera trajectory.
use super::placement::{surface, MountSpace};
use super::*;
use crate::scene::procedural_indoor::{
    cameras::{
        coverage::{Coverage, VisibilityView},
        diversity::Track,
        multiview::{pair_samples_cached, CHECK_TIMES},
        rig::MixedRigContract,
    },
    layout::{stream, IndoorManifest},
};
use rand::Rng;
mod mounts;
mod paths;
mod solver;

type Prototype = (Option<crate::calibration::FisheyeSensor>, f32, f32, f32);
struct Candidate {
    camera: IndoorCamera,
    track: Track,
    views: Vec<VisibilityView>,
}

pub(super) fn sample(
    scene: &mut IndoorManifest,
    coverage: &Coverage,
    space: &MountSpace,
    prototypes: &[Prototype],
    order: &[usize],
    contract: MixedRigContract,
) -> bool {
    let count = prototypes.len();
    let mut pools: Vec<Vec<Candidate>> = (0..count).map(|_| Vec::new()).collect();
    let mut solver = solver::Solver::default();
    let mut path_budget = paths::Budget::default();
    for (camera, &index) in std::mem::take(&mut scene.cameras).into_iter().zip(order) {
        let track = Track::new(&camera);
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
    for pass in 0..2 {
        let mut proposal_order: Vec<_> = (0..count).collect();
        // Populate constrained peer paths before the broad reference can reuse them.
        if pass == 1 {
            proposal_order.sort_by_key(|&i| (i == 0, pools[i].len()));
        }
        for index in proposal_order {
            let prototype = &prototypes[index];
            let mut rng = stream(scene.seed, 0x504f_4f4c_0000 + index as u64 * 2 + pass);
            // Domain construction is bounded independently of rejection rate.
            // Reuse successful local paths as well as independent origins.
            let mounted_slot = scene.camera_settings.ceiling_mounted(scene.seed, index);
            // Busy ceilings may exclude the initial body standoff everywhere.
            // Retry within its configured range only after that slot has no
            // feasible paths; its mounting role, lens and roll remain fixed.
            let redraw_standoff = mounted_slot && pass == 1 && pools[index].is_empty();
            let attempts = if redraw_standoff { 384 } else { 96 };
            for _ in 0..attempts {
                if pools[index].len() >= 24 * (pass as usize + 1) {
                    break;
                }
                scene.camera_diagnostics.peer_proposals += 1;
                let camera = if scene.camera_settings.ceiling_mounted(scene.seed, index) {
                    if !pools[0].is_empty() {
                        scene
                            .cameras
                            .push(pools[0][rng.random_range(0..pools[0].len())].camera.clone());
                    }
                    let mut adjusted = *prototype;
                    if redraw_standoff {
                        let range = scene
                            .camera_settings
                            .ceiling_mount
                            .as_ref()
                            .unwrap()
                            .standoff_m;
                        adjusted.2 = rng.random_range(range[0]..=range[1]);
                    }
                    let mut camera = mounted(scene, index, &adjusted, &mut rng);
                    if rng.random_bool(0.5) {
                        if let Some(camera) = &mut camera {
                            super::coherent::motion(
                                scene, index, count, camera, contract, &mut rng,
                            );
                        }
                    }
                    scene.cameras.clear();
                    camera.and_then(|mut camera| {
                        space
                            .fit_rail(scene, index, &mut camera, &mut rng)
                            .then_some(camera)
                    })
                } else {
                    let parent_slot = if pass == 1 && index == 0 {
                        (1..count)
                            .filter(|&i| {
                                !pools[i].is_empty()
                                    && !scene.camera_settings.ceiling_mounted(scene.seed, i)
                            })
                            .min_by_key(|&i| pools[i].len())
                            .unwrap_or(index)
                    } else if !pools[index].is_empty() && rng.random_bool(0.5) {
                        index
                    } else {
                        rng.random_range(0..count)
                    };
                    if !pools[parent_slot].is_empty()
                        && !scene
                            .camera_settings
                            .ceiling_mounted(scene.seed, parent_slot)
                    {
                        let parent = rng.random_range(0..pools[parent_slot].len());
                        scene
                            .cameras
                            .push(pools[parent_slot][parent].camera.clone());
                    }
                    let camera = if pass == 1
                        && rng.random_bool(0.75)
                        && scene.camera_settings.camera_moves(scene.seed, index)
                        && scene.camera_settings.handheld.is_none()
                    {
                        paths::centered(
                            scene,
                            index,
                            prototype.1,
                            coverage,
                            &mut rng,
                            &mut path_budget,
                        )
                    } else {
                        super::room::propose(
                            scene,
                            index,
                            &mut rng,
                            coverage,
                            (index != 0).then_some(prototype.1),
                            false,
                        )
                    };
                    scene.cameras.clear();
                    camera
                };
                let Some(camera) = camera else { continue };
                if !scene
                    .camera_settings
                    .accepts_motion(scene.seed, index, &camera)
                    || (camera.is_ceiling_mounted() && !space.clear(scene, &camera))
                {
                    continue;
                }
                let views: Vec<_> = CHECK_TIMES
                    .into_iter()
                    .map(|t| coverage.view(&camera, scene.camera_aspect_ratio, t))
                    .collect();
                if camera.is_ceiling_mounted() && !super::overview::suitable(scene, &camera, &views)
                {
                    continue;
                }
                let track = Track::new(&camera);
                pools[index].push(Candidate {
                    camera,
                    track,
                    views,
                });
            }
        }

        if pass == 1 {
            paths::reuse(scene, coverage, prototypes, &mut pools, &mut path_budget);
            paths::perturb(scene, coverage, &mut pools, &mut path_budget);
        }
        if solver.solve(
            scene,
            coverage,
            &pools,
            contract,
            if pass == 0 { 6144 } else { 3072 },
        ) {
            return true;
        }
    }
    // Ceiling fixtures can leave only small connected mounting patches. Reuse
    // those patches across slots, with each slot's own optics and orientation,
    // instead of treating one unlucky independent draw as an impossible rig.
    mounts::expand(
        scene,
        coverage,
        space,
        prototypes,
        &mut pools,
        &mut path_budget,
    );
    if solver.solve(scene, coverage, &pools, contract, 2048) {
        return true;
    }
    // Intrinsics are a proposal PDF. Only after fixed-calibration recovery fails,
    // redraw room pinholes from that same PDF on already feasible paths. Explicit
    // fixed priors, fisheye sensors, mount roles and all admission tests stay intact.
    paths::resample_intrinsics(scene, coverage, prototypes, &mut pools, &mut path_budget);
    let found = solver.solve(scene, coverage, &pools, contract, 0);
    if !found {
        scene.cameras.clear();
    }
    found
}

fn mounted(
    scene: &IndoorManifest,
    index: usize,
    &(lens, fov, standoff, roll): &Prototype,
    rng: &mut impl Rng,
) -> Option<IndoorCamera> {
    let mounting = scene.camera_settings.ceiling_mount.as_ref()?;
    let (lo, hi) = scene.primary_room_bounds();
    let lo = lo + Vec2::splat(mounting.wall_clearance_m);
    let hi = hi - Vec2::splat(mounting.wall_clearance_m);
    let p = Vec2::new(rng.random_range(lo.x..hi.x), rng.random_range(lo.y..hi.y));
    Some(mounted_at(
        scene,
        index,
        &(lens, fov, standoff, roll),
        p,
        rng,
    ))
}

fn mounted_at(
    scene: &IndoorManifest,
    index: usize,
    &(lens, fov, standoff, roll): &Prototype,
    p: Vec2,
    rng: &mut impl Rng,
) -> IndoorCamera {
    let mounting = scene.camera_settings.ceiling_mount.as_ref().unwrap();
    let (surface, normal) = surface(scene, p);
    let start = surface + normal * standoff;
    let rotation =
        Transform::IDENTITY.looking_to(normal, Vec3::Z).rotation * Quat::from_rotation_z(roll);
    let mut camera = IndoorCamera {
        ceiling_mount: Some(CeilingCameraMount {
            surface,
            inward_normal: normal,
            rotation,
            standoff_m: standoff,
            body_radius_m: mounting.body_radius_m,
        }),
        fisheye: lens,
        principal_point_offset: None,
        start,
        end: start,
        target: start + normal * 2.,
        fov_degrees: fov,
        motion: None,
    };
    super::trajectory::propose(scene, index, &mut camera, rng);
    camera
}
