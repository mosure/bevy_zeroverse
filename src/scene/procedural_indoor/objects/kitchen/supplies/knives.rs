//! Fitted knife blocks: empty sockets and individually sized, separated handles.
use super::*;
use crate::scene::procedural_indoor::placement::{Footprint, SurfacePattern};

#[derive(Debug, Clone, serde::Serialize)]
pub struct KnifeBlockProgram {
    pub body_fraction: f32,
    pub taper: f32,
    pub roundness: f32,
    pub slots: usize,
    pub occupancy: f32,
    pub body: Surface,
    pub handle: Surface,
}
pub fn parameters(o: &IndoorObject) -> KnifeBlockProgram {
    let mut r = stream(o.seed, 0x4b4e494645505247);
    KnifeBlockProgram {
        body_fraction: r.random_range(0.46..0.65),
        taper: r.random_range(0.78..1.),
        roundness: r.random_range(2.2..8.0),
        slots: r.random_range(2..=7),
        occupancy: r.random_range(0.50..1.),
        body: [Surface::Wood, Surface::Metal, Surface::Plastic][r.random_range(0..3)],
        handle: [Surface::WoodEdge, Surface::Plastic, Surface::Metal][r.random_range(0..3)],
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Knife {
    pub footprint: Footprint,
    pub handle: Vec3,
    pub blade: f32,
    pub occupied: bool,
}
pub(crate) fn contents(o: &IndoorObject, p: &KnifeBlockProgram) -> Vec<Knife> {
    let mut r = stream(o.seed, 0x4b4e494645534c54);
    // A rectangle strictly inside even the roundest body profile.
    let half = o.size.xz() * 0.45 * p.taper * 0.66;
    let pattern = SurfacePattern::sample(&mut r);
    let mut result: Vec<Knife> = Vec::with_capacity(p.slots);
    for _ in 0..p.slots {
        let handle = Vec3::new(
            r.random_range(0.012_f32..0.023).min(o.size.x * 0.17),
            r.random_range(0.075_f32..0.13)
                .min(o.size.y * (1. - p.body_fraction) * 0.82),
            r.random_range(0.018_f32..0.032).min(o.size.z * 0.18),
        );
        for _ in 0..12 {
            let yaw = pattern.yaw(&mut r);
            let Some(q) = pattern.propose((-half, half), handle.xz(), yaw, 0.002, &mut r) else {
                continue;
            };
            if result.iter().any(|v| v.footprint.overlaps(q, 0.003)) {
                continue;
            }
            result.push(Knife {
                footprint: q,
                handle,
                blade: (o.size.y * (1. - p.body_fraction) - handle.y)
                    .min(r.random_range(0.005..0.020)),
                occupied: r.random_bool(p.occupancy as f64),
            });
            break;
        }
    }
    result
}
pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let label = o.kind.class_name();
    let h = s.y * p.body_fraction;
    let half = s.xz() * 0.45;
    a.part(p.body, label).housing(
        &[(0., half * 0.92), (h * 0.12, half), (h, half * p.taper)],
        p.roundness,
        Transform::IDENTITY,
    );
    for v in contents(o, &p) {
        let tf = Transform::from_xyz(v.footprint.center.x, h, v.footprint.center.y)
            .with_rotation(Quat::from_rotation_y(v.footprint.yaw));
        a.part(Surface::Ink, label).cuboid(
            Vec3::new(v.handle.x * 0.30, 0.0006, v.handle.z * 0.82),
            0.0001,
            tf.with_translation(tf.transform_point(Vec3::Y * 0.0004)),
        );
        if !v.occupied {
            continue;
        }
        a.part(Surface::Metal, label).cuboid(
            Vec3::new(v.handle.x * 0.14, v.blade + 0.004, v.handle.z * 0.72),
            0.0002,
            tf.with_translation(tf.transform_point(Vec3::Y * (v.blade * 0.5 - 0.001))),
        );
        a.part(p.handle, label).cuboid(
            v.handle,
            v.handle.min_element() * 0.18,
            tf.with_translation(tf.transform_point(Vec3::Y * (v.blade + v.handle.y * 0.5))),
        );
        // Paired rivets remain attached to each handle, not a repeated global row.
        if p.handle != Surface::Metal {
            for t in [0.25, 0.73] {
                let pos = Vec3::new(v.handle.x * 0.5, v.blade + v.handle.y * t, 0.);
                a.part(Surface::Metal, label).ellipsoid(
                    Vec3::new(0.0005, 0.0013, 0.0013),
                    tf.with_translation(tf.transform_point(pos)),
                );
            }
        }
    }
}
