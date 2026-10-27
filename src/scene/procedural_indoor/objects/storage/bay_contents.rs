//! The same products used elsewhere in a room, fitted to a real cabinet bay.
//! Sampling usable center intervals avoids a preferred center or clamp-at-edge.
use super::*;
use crate::scene::procedural_indoor::placement::SurfacePattern;

pub(crate) fn item(
    parent: &IndoorObject,
    seed: u64,
    kind: ObjectKind,
    lo: Vec3,
    hi: Vec3,
) -> Option<IndoorObject> {
    let space = hi - lo;
    if space.min_element() < 0.025 {
        return None;
    }
    let mut r = stream(seed, 0x424159434f4e5445);
    let size = if kind == ObjectKind::Vase {
        let d = space.x.min(space.z) * r.random_range(0.38..0.76);
        Vec3::new(d, (d * r.random_range(0.75..2.3)).min(space.y * 0.88), d)
    } else {
        space
            * Vec3::new(
                r.random_range(0.43..0.84),
                r.random_range(0.38..0.82),
                r.random_range(0.45..0.86),
            )
    };
    let pattern = SurfacePattern::sample(&mut r);
    for _ in 0..8 {
        let yaw = pattern.yaw(&mut r);
        let Some(place) = pattern.propose((lo.xz(), hi.xz()), size.xz(), yaw, 0.004, &mut r) else {
            continue;
        };
        let mut object = parent.clone();
        object.kind = kind;
        object.seed = seed;
        object.finish_seed = None;
        object.size = size;
        object.position = Vec3::new(place.center.x, lo.y, place.center.y);
        object.yaw = yaw;
        object.variant = r.random_range(0..4);
        object.cable_path = None;
        object.interaction_target = None;
        object.support = None;
        return Some(object);
    }
    None
}

pub(super) fn build(
    a: &mut Assembly,
    parent: &IndoorObject,
    seed: u64,
    kind: ObjectKind,
    lo: Vec3,
    hi: Vec3,
) {
    if let Some(o) = item(parent, seed, kind, lo, hi) {
        let child = super::super::build_object(&o);
        super::super::public::append_at(a, child, o.transform());
    }
}
