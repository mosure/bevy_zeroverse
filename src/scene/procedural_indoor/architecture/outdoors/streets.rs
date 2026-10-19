//! Near-window hardscape. Separate pavers/kerbs give metric perspective cues;
//! all strips remain outside the primary room and neighboring room footprint.
use super::*;

pub(super) fn build(a: &mut Assembly, scene: &IndoorManifest) {
    let mut rng = stream(scene.seed, 6370);
    let half = scene.room_size.xz() * 0.5;
    let lo = Vec2::new(-half.x - 0.55, -half.y - 0.55);
    let hi = Vec2::new(half.x + 0.55, half.y + NEIGHBOR_DEPTH + 0.55);
    let district = District::sample(scene.seed);
    let width = district.paving_width;
    let finish = super::super::super::materials::exterior::label(
        Surface::Concrete,
        "floor",
        scene.seed ^ 6370,
    );
    // Four disjoint perimeter strips with mitre-free corner allocation.
    let strips = [
        (
            Vec2::new(lo.x - width, lo.y - width),
            Vec2::new(lo.x, hi.y + width),
        ),
        (
            Vec2::new(hi.x, lo.y - width),
            Vec2::new(hi.x + width, hi.y + width),
        ),
        (Vec2::new(lo.x, lo.y - width), Vec2::new(hi.x, lo.y)),
        (Vec2::new(lo.x, hi.y), Vec2::new(hi.x, hi.y + width)),
    ];
    let pitch = district.paving_pitch;
    for (index, (min, max)) in strips.into_iter().enumerate() {
        let extent = max - min;
        let axis = if extent.x > extent.y {
            Vec2::X
        } else {
            Vec2::Y
        };
        let cross = Vec2::new(axis.y, axis.x);
        let count = (extent.dot(axis) / pitch).ceil() as usize;
        let cell = extent.dot(axis) / count as f32;
        for i in 0..count {
            let center =
                (min + max) * 0.5 + axis * ((i as f32 + 0.5) * cell - extent.dot(axis) * 0.5);
            let size = axis * (cell - 0.009) + cross * (width - 0.008);
            a.box_part(
                Surface::Concrete,
                &finish,
                Vec3::new(center.x, -0.035, center.y),
                Vec3::new(size.x, 0.07, size.y),
                0.003,
            );
        }
        // A continuous curb borders the outer edge; expansion gaps separate blocks.
        let outward = match index {
            0 => Vec2::NEG_X,
            1 => Vec2::X,
            2 => Vec2::NEG_Y,
            _ => Vec2::Y,
        };
        for i in 0..count {
            let center = (min + max) * 0.5
                + axis * ((i as f32 + 0.5) * cell - extent.dot(axis) * 0.5)
                + outward * (width * 0.5 - 0.035);
            let size = axis * (cell - 0.012) + cross * 0.055;
            a.box_part(
                Surface::Concrete,
                "other_structure",
                Vec3::new(center.x, 0.0325, center.y),
                Vec3::new(size.x, 0.065, size.y),
                0.006,
            );
        }
        // Occasional short metal railing on the remote half of a walkway.
        // Railings are capped, with posts anchored into the paving below them.
        if !rng.random_bool(district.railing_probability as f64) {
            continue;
        }
        let start =
            (min + max) * 0.5 - axis * extent.dot(axis) * 0.26 + outward * (width * 0.5 - 0.08);
        let end =
            (min + max) * 0.5 + axis * extent.dot(axis) * 0.26 + outward * (width * 0.5 - 0.08);
        let height = rng.random_range(0.75..1.15);
        let point = |p: Vec2, y| Vec3::new(p.x, y, p.y);
        for y in [height * 0.48, height] {
            a.part(Surface::Metal, "other_structure")
                .rod(point(start, y), point(end, y), 0.016);
        }
        let posts = (start.distance(end) / 1.0).ceil() as usize;
        for i in 0..=posts {
            let p = start.lerp(end, i as f32 / posts as f32);
            a.part(Surface::Metal, "other_structure")
                .rod(point(p, 0.), point(p, height), 0.018);
        }
    }
}
