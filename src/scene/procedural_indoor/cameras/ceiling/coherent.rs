//! Proposals for the mixed baseline program. Correlate narrow rigs before
//! rejection; never change roles, calibration, requested travel or clearance.
use super::*;
use crate::scene::procedural_indoor::{
    cameras::{
        coverage::VisibilityView, rig::MixedRigContract, viewpoint::ViewpointProgram, CameraMotion,
    },
    layout::{IndoorManifest, CAMERA_CLEARANCE},
};
use rand::{seq::IteratorRandom, Rng};

pub(super) fn anchor(scene: &IndoorManifest, mounted: bool) -> Option<&IndoorCamera> {
    scene
        .cameras
        .iter()
        .find(|c| c.is_ceiling_mounted() == mounted)
        .or_else(|| scene.cameras.first())
}

pub(super) fn point(
    scene: &IndoorManifest,
    mounted: bool,
    contract: MixedRigContract,
    rng: &mut impl Rng,
) -> Option<Vec2> {
    let anchor = anchor(scene, mounted)?;
    let policy = scene.camera_settings.multiview.as_ref()?;
    let max = if anchor.is_ceiling_mounted() == mounted {
        policy.max_baseline
    } else {
        contract.max_cross_mount_horizontal_m
    };
    let angle = rng.random_range(0.0..std::f32::consts::TAU);
    let min = if anchor.is_ceiling_mounted() != mounted {
        policy.min_baseline.max(policy.min_reference_baseline)
    } else {
        policy.min_baseline
    };
    let distance = rng.random_range(min..=max);
    Some(anchor.start.xz() + Vec2::new(angle.cos(), angle.sin()) * distance)
}

/// A narrow rig needs compatible travel before any peer exists. Choose a
/// proposal cap inside the requested ranges, accounting for fixed peers. This
/// is not admission: the exact requested minimum and maximum still apply.
fn travel_cap(scene: &IndoorManifest, count: usize, contract: MixedRigContract) -> f32 {
    let policy = &scene.camera_settings;
    let mut max = 0.40 + 3.0 * contract.baseline;
    for index in 0..count {
        if !policy.camera_moves(scene.seed, index) {
            max = max.min(policy.multiview.as_ref().unwrap().max_baseline * 0.5);
        } else {
            let limit = if policy.ceiling_mounted(scene.seed, index) {
                policy.ceiling_mount.as_ref().unwrap().travel_m[1]
            } else {
                policy.path_length_max
            };
            max = max.min(limit);
        }
    }
    max
}

pub(super) fn motion(
    scene: &IndoorManifest,
    index: usize,
    count: usize,
    camera: &mut IndoorCamera,
    contract: MixedRigContract,
    rng: &mut impl Rng,
) {
    if !scene.camera_settings.camera_moves(scene.seed, index) {
        camera.end = camera.start;
        camera.motion = None;
        return;
    }
    // Explicit handheld increments remain authoritative.
    if !camera.is_ceiling_mounted() && scene.camera_settings.handheld.is_some() {
        return;
    }
    let [min, max] = if camera.is_ceiling_mounted() {
        scene
            .camera_settings
            .ceiling_mount
            .as_ref()
            .unwrap()
            .travel_m
    } else {
        [
            scene
                .camera_settings
                .path_length_min
                .max(0.001)
                .min(scene.camera_settings.path_length_max),
            scene.camera_settings.path_length_max,
        ]
    };
    if max <= 0. {
        return;
    }
    let common = travel_cap(scene, count, contract).max(min).min(max);
    // Wide rigs need not inherit the shortest member's travel range.
    let cap = common + (max - common) * contract.baseline.powi(2);
    let a = anchor(scene, camera.is_ceiling_mounted());
    let delta = a.map_or(camera.end - camera.start, |a| a.end - a.start);
    let length = if a.is_none() && contract.baseline < 0.5 {
        // Do not clamp almost every long walkthrough proposal onto the same
        // rail length. Shorter legal sweeps can fit between ceiling fixtures,
        // and room-to-room travel remains a continuous sampled distribution.
        rng.random_range(min.ln()..=cap.ln()).exp()
    } else if delta.length() > 1e-4 {
        (delta.length()
            * rng.random_range(1.0 - 0.5 * contract.baseline..=1.0 + 0.5 * contract.baseline))
        .clamp(min, cap)
    } else {
        rng.random_range(min.ln()..=cap.ln()).exp()
    };
    let direction = if delta.xz().length_squared() > 1e-8 {
        delta.xz().normalize()
    } else {
        let angle = rng.random_range(0.0..std::f32::consts::TAU);
        Vec2::new(angle.cos(), angle.sin())
    };
    let normal = camera
        .ceiling_mount
        .as_ref()
        .map_or(Vec3::Y, |m| m.inward_normal);
    let direction = Quat::from_rotation_y(rng.random_range(-1.0..=1.0) * contract.baseline)
        * Vec3::new(direction.x, 0., direction.y);
    let delta = (direction - normal * normal.dot(direction)).normalize() * length;
    camera.end = camera.start + delta;
    camera.motion = Some(CameraMotion {
        orientations: camera.ceiling_mount.as_ref().map(|m| [m.rotation; 2]),
        control: [camera.start + delta / 3., camera.start + delta * (2. / 3.)],
        target_end: camera.target + delta,
        roll: [0.; 2],
        route: Vec::new(),
    });
}

pub(super) fn room(
    scene: &IndoorManifest,
    index: usize,
    fov: f32,
    reference: Option<&VisibilityView>,
    contract: MixedRigContract,
    rng: &mut impl Rng,
) -> Option<IndoorCamera> {
    let p = point(scene, false, contract, rng)?;
    let floor = scene.floor_height(p);
    let upper = (scene.ceiling_height(p)
        - scene.program.as_ref().map_or(0.48, |p| p.light_drop)
        - CAMERA_CLEARANCE
        - 0.08)
        .min(floor + 3.25);
    let a = anchor(scene, false)?;
    if !a.is_ceiling_mounted() {
        // Reuse navigated topology for room peers. This preserves curved and
        // waypoint paths; ceiling-compatible straight proposals are only needed
        // when mounting constraints actually require them.
        let policy = scene.camera_settings.multiview.as_ref()?;
        let local = super::super::multiview::MultiViewSettings {
            min_reference_baseline: if std::ptr::eq(a, &scene.cameras[0]) {
                policy.min_reference_baseline
            } else {
                0.
            },
            ..policy.clone()
        };
        let (lo, hi) = scene.primary_room_bounds();
        let reachable = policy
            .max_baseline
            .min(lo.distance(hi))
            .max(local.min_reference_baseline)
            .max(policy.min_baseline);
        let mut camera = super::super::multiview::propose(
            a,
            a.path_length().max(0.001),
            reachable,
            Vec2::new(floor + 0.7 - a.start.y, upper - a.start.y),
            &local,
            &scene.camera_settings.intrinsics,
            rng,
        );
        camera.fov_degrees = fov;
        if (a.fisheye != scene.camera_settings.lens_for(scene.seed, index)
            || (a.fov_degrees - fov).abs() > 12.)
            && rng.random_bool(0.5)
        {
            // A narrow peer aimed at the broad anchor's center may see only a
            // blank wall or one furniture surface. Sample another reference-
            // visible landmark instead of resampling that slot's calibration.
            if let Some(target) = reference.and_then(|view| {
                view.surfaces()
                    .filter(|p| {
                        p.distance(camera.start) > 1.0
                            && scene.in_primary_room(*p, 0.)
                            && p.y < camera.start.y + 0.5
                    })
                    .choose(rng)
            }) {
                let delta = target - camera.target;
                camera.target = target;
                if let Some(motion) = &mut camera.motion {
                    motion.target_end += delta;
                }
            }
        }
        scene
            .camera_settings
            .prepare_motion(scene.seed, index, &mut camera, rng);
        return Some(camera);
    }
    let y = ViewpointProgram::sample(scene.seed).height(floor, upper, rng)?;
    if y < floor + 0.70 || y > upper {
        return None;
    }
    let start = Vec3::new(p.x, y, p.y);
    // Nested fields of view need a shared surface, not a shared optical axis.
    // Retain the room anchor's gaze in close rigs; aim an opposite-mount peer at
    // visible primary-room content, never the ceiling above its optical center.
    let target = reference?
        .surfaces()
        .filter(|p| p.y < start.y && p.distance(start) > 1.0 && scene.in_primary_room(*p, 0.))
        .choose(rng)?;
    let mut camera = IndoorCamera {
        ceiling_mount: None,
        fisheye: scene.camera_settings.lens_for(scene.seed, index),
        principal_point_offset: None,
        start,
        end: start,
        target,
        fov_degrees: fov,
        motion: None,
    };
    scene
        .camera_settings
        .prepare_motion(scene.seed, index, &mut camera, rng);
    Some(camera)
}
