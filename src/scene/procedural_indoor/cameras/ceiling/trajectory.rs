//! Ceiling rails stay on one supporting plane. No interpolation across soffit
//! edges: the exact piecewise ceiling intervals and continuous swept body/lens
//! envelopes are checked before a route is accepted.
use super::{placement::surface, *};
use crate::scene::procedural_indoor::{cameras::CameraMotion, layout::IndoorManifest};
use rand::Rng;

pub(super) fn propose(
    scene: &IndoorManifest,
    index: usize,
    camera: &mut IndoorCamera,
    rng: &mut impl Rng,
) {
    if !scene.camera_settings.camera_moves(scene.seed, index) {
        return;
    }
    let policy = scene.camera_settings.ceiling_mount.as_ref().unwrap();
    let length = rng
        .random_range(policy.travel_m[0].ln()..=policy.travel_m[1].ln())
        .exp();
    let angle = rng.random_range(0.0..std::f32::consts::TAU);
    let normal = camera.ceiling_mount.as_ref().unwrap().inward_normal;
    let direction = Vec3::new(angle.cos(), 0., angle.sin());
    let tangent = (direction - normal * normal.dot(direction)).normalize();
    set_rail(camera, tangent * length);
}

/// One authoritative straight, constant-orientation rail representation for
/// initial proposals and bounded clearance recovery.
pub(super) fn set_rail(camera: &mut IndoorCamera, delta: Vec3) {
    camera.end = camera.start + delta;
    camera.motion = Some(CameraMotion {
        orientations: Some([camera.ceiling_mount.as_ref().unwrap().rotation; 2]),
        control: [
            camera.start.lerp(camera.end, 1. / 3.),
            camera.start.lerp(camera.end, 2. / 3.),
        ],
        target_end: camera.target + camera.end - camera.start,
        roll: [0.; 2],
        route: Vec::new(),
    });
}

pub(super) fn plane_clear(scene: &IndoorManifest, camera: &IndoorCamera) -> bool {
    let mount = camera.ceiling_mount.as_ref().unwrap();
    let delta = camera.end - camera.start;
    if !camera.start.is_finite()
        || !camera.end.is_finite()
        || !mount.surface.is_finite()
        || !mount.inward_normal.is_finite()
    {
        return false;
    }
    if let Some(motion) = &camera.motion {
        // A rail is a straight, constant-orientation cubic. Reject malformed
        // archived trajectories instead of validating a different runtime path.
        if motion.control.iter().any(|p| !p.is_finite())
            || !motion.target_end.is_finite()
            || motion.roll != [0.; 2]
            || !motion.route.is_empty()
            || motion.control[0].distance(camera.start.lerp(camera.end, 1. / 3.)) > 1e-5
            || motion.control[1].distance(camera.start.lerp(camera.end, 2. / 3.)) > 1e-5
            || motion.target_end.distance(camera.target + delta) > 1e-5
            || motion.orientations != Some([mount.rotation; 2])
        {
            return false;
        }
    } else if camera.start != camera.end {
        return false;
    }
    // A soffit may be thinner than any uniform time grid. Test its boundaries
    // and all intervening intervals, where the roof is an affine function.
    let mut times = vec![0., 1.];
    if let Some(envelope) = &scene.envelope {
        for (patch, origin) in envelope
            .soffits
            .iter()
            .map(|p| (p, mount.surface.xz()))
            .chain(
                envelope
                    .floor_patches
                    .iter()
                    .map(|p| (p, camera.start.xz())),
            )
        {
            for axis in 0..2 {
                let d = delta.xz()[axis];
                if d.abs() > 1e-8 {
                    for boundary in [patch.min[axis], patch.max[axis]] {
                        let t = (boundary - origin[axis]) / d;
                        if t > 0. && t < 1. {
                            times.push(t);
                        }
                    }
                }
            }
        }
        if !envelope.segment_clear(
            scene.room_size,
            camera.start,
            camera.end,
            mount.body_radius_m,
        ) {
            return false;
        }
    }
    times.sort_by(f32::total_cmp);
    let interior: Vec<_> = times.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect();
    times.extend(interior);
    times.into_iter().all(|t| {
        let expected = mount.surface + delta * t;
        let (actual, normal) = surface(scene, expected.xz());
        actual.distance(expected) < 1e-4
            && normal.distance(mount.inward_normal) < 1e-5
            && (camera.start + delta * t).y
                >= scene.floor_height((camera.start + delta * t).xz()) + 1.5
    })
}

/// Reserve the entire body sweep even when two rails traverse it at different
/// times. This also protects against playback reversal and independent motion.
pub(crate) fn bodies_overlap(a: &IndoorCamera, b: &IndoorCamera) -> bool {
    match (a.ceiling_sweep_bounds(), b.ceiling_sweep_bounds()) {
        (Some((lo, hi)), Some((a, b))) => lo.cmplt(b).all() && hi.cmpgt(a).all(),
        (Some(bounds), None) => room_intersects(b, bounds),
        (None, Some(bounds)) => room_intersects(a, bounds),
        _ => false,
    }
}

fn room_intersects(camera: &IndoorCamera, (lo, hi): (Vec3, Vec3)) -> bool {
    use crate::{
        camera::TrajectorySampler,
        scene::procedural_indoor::layout::{segment_hits_box, CAMERA_CLEARANCE},
    };
    let path = camera.runtime_trajectory();
    // The Bezier convex hull contains the entire curve, including between
    // temporal checkpoints. Waypoint paths are checked segment by segment.
    let points = match path {
        TrajectorySampler::CubicBezier { positions, .. } => positions.to_vec(),
        TrajectorySampler::WaypointPath { positions, .. } => {
            let pad = Vec3::splat(CAMERA_CLEARANCE);
            return positions
                .windows(2)
                .any(|w| segment_hits_box(w[0], w[1], lo - pad, hi + pad));
        }
        _ => vec![camera.start, camera.end],
    };
    let a = points
        .iter()
        .copied()
        .fold(Vec3::splat(f32::INFINITY), Vec3::min)
        - Vec3::splat(CAMERA_CLEARANCE);
    let b = points
        .iter()
        .copied()
        .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max)
        + Vec3::splat(CAMERA_CLEARANCE);
    a.cmplt(hi).all() && b.cmpgt(lo).all()
}
