//! Small kitchen products use their own construction streams. The outer metric
//! envelope is sampled before admission; every fold, pump and handle stays in it.
pub mod knives;
use super::*;
pub mod bread;
pub mod dispenser;
pub mod paper_towel;
pub mod rack;
pub mod sponge;
#[cfg(test)]
mod tests;
pub mod utensils;

#[derive(Debug, Clone, serde::Serialize)]
pub struct TowelProgram {
    pub layers: usize,
    pub thickness_weights: Vec<f32>,
    pub fold_width: f32,
    pub fold_depth: f32,
    pub offset_fraction: f32,
    pub skew: f32,
    pub finish: Surface,
}
pub fn towel_parameters(o: &IndoorObject) -> TowelProgram {
    let mut r = stream(o.seed, 0x544f57454c464f4c);
    let layers = r.random_range(2..=6);
    TowelProgram {
        layers,
        thickness_weights: (0..layers).map(|_| r.random_range(0.65..1.35)).collect(),
        fold_width: r.random_range(0.66..0.98),
        fold_depth: r.random_range(0.65..0.98),
        offset_fraction: r.random_range(0.006..0.035),
        skew: r.random_range(-0.035..0.035),
        finish: if r.random_bool(0.5) {
            Surface::Fabric
        } else {
            Surface::FabricAlt
        },
    }
}
pub(super) fn towel(a: &mut Assembly, o: &IndoorObject) {
    let p = towel_parameters(o);
    let mut r = stream(o.seed, 0x544f57454c504c41);
    let s = o.size;
    let sum = p.thickness_weights.iter().sum::<f32>();
    let mut y = 0.;
    let mut extent = s.xz() * Vec2::new(p.fold_width, p.fold_depth);
    let mut tf = Transform::IDENTITY;
    for weight in p.thickness_weights {
        let h = s.y * weight / sum;
        // Fit each fold on the flat part of the previous fold. Accumulated
        // rotation and translation cannot create an unsupported layer or overhang.
        let angle = r.random_range(-1.0..1.0) * p.skew;
        let available = extent * (1. - p.offset_fraction);
        let half = crate::scene::procedural_indoor::placement::rotated_half(extent, angle);
        let scale = (available * 0.5 / half).min_element().min(1.);
        let next = extent * scale;
        let half = crate::scene::procedural_indoor::placement::rotated_half(next, angle);
        let freedom = (extent * 0.5 - half) * 0.7;
        let shift = Vec3::new(
            freedom.x * r.random_range(-1.0..1.0),
            0.,
            freedom.y * r.random_range(-1.0..1.0),
        );
        tf = tf.mul_transform(
            Transform::from_translation(shift).with_rotation(Quat::from_rotation_y(angle)),
        );
        a.part(p.finish, o.kind.class_name()).cuboid(
            Vec3::new(next.x, h, next.y),
            h * 0.22,
            tf.with_translation(tf.translation + Vec3::Y * (y + h * 0.5)),
        );
        extent = next;
        y += h;
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BoardProgram {
    pub roundness: f32,
    pub body_fraction: f32,
    pub handle_width: f32,
    pub handle: bool,
    pub edge_fraction: f32,
    pub finish: Surface,
    pub body_scale: Vec2,
    pub handle_offset: f32,
    pub feet_fraction: f32,
}
pub fn board_parameters(o: &IndoorObject) -> BoardProgram {
    let mut r = stream(o.seed, 0x435554424f415244);
    BoardProgram {
        body_scale: Vec2::new(r.random_range(0.74..1.), r.random_range(0.72..1.)),
        handle_offset: r.random_range(-0.23..0.23),
        feet_fraction: if r.random_bool(0.24) {
            r.random_range(0.06..0.15)
        } else {
            0.
        },
        roundness: r.random_range(2.0..10.),
        body_fraction: r.random_range(0.58..0.90),
        handle_width: r.random_range(0.17..0.36),
        handle: r.random_bool(0.40),
        edge_fraction: r.random_range(0.10..0.32),
        finish: if r.random_bool(0.73) {
            Surface::Wood
        } else {
            Surface::Plastic
        },
    }
}
pub(super) fn board(a: &mut Assembly, o: &IndoorObject) {
    let p = board_parameters(o);
    let s = o.size;
    let width = s.x * p.body_scale.x * if p.handle { p.body_fraction } else { 1. };
    let cx = if p.handle { (width - s.x) * 0.5 } else { 0. };
    let half = Vec2::new(width, s.z * p.body_scale.y) * 0.5;
    let bottom = s.y * p.feet_fraction;
    let edge = s.y * p.edge_fraction;
    let inset = Vec2::splat(edge).min(half * 0.05);
    a.part(p.finish, o.kind.class_name()).housing(
        &[
            (bottom, half - inset),
            (bottom + edge, half),
            (s.y - edge, half),
            (s.y, half - inset),
        ],
        p.roundness,
        Transform::from_xyz(cx, 0., 0.),
    );
    if bottom > 0. {
        for x in [-0.42, 0.42] {
            for z in [-0.42, 0.42] {
                a.part(Surface::Rubber, o.kind.class_name()).cylinder(
                    (bottom * 1.1).min(half.min_element() * 0.10),
                    bottom + 0.0002,
                    Transform::from_xyz(cx + x * half.x, bottom * 0.5 + 0.0001, z * half.y),
                );
            }
        }
    }

    if p.handle {
        // A real attached end-grain handle, partly embedded in the board edge.
        let start = cx + width * 0.40;
        let end = s.x * 0.5;
        a.box_part(
            p.finish,
            o.kind.class_name(),
            Vec3::new(
                (start + end) * 0.5,
                (s.y + bottom) * 0.5,
                s.z * p.handle_offset * p.body_scale.y,
            ),
            Vec3::new(
                end - start,
                (s.y - bottom) * 0.82,
                s.z * p.body_scale.y * p.handle_width,
            ),
            edge.min(0.004),
        );
    }
}

/// Physical preparation surface excludes the handle and rounded edge corners.
fn board_top(o: &IndoorObject) -> (Vec2, Vec2, f32) {
    let p = board_parameters(o);
    let width = o.size.x * p.body_scale.x * if p.handle { p.body_fraction } else { 1. };
    let center = Vec2::new(
        if p.handle {
            (width - o.size.x) * 0.5
        } else {
            0.
        },
        0.,
    );
    let half = Vec2::new(width, o.size.z * p.body_scale.y) * 0.5;
    let inset = Vec2::splat(o.size.y * p.edge_fraction).min(half * 0.05);
    (center, half - inset, p.roundness)
}
pub fn board_top_bounds(o: &IndoorObject) -> (Vec2, Vec2) {
    let (c, h, p) = board_top(o);
    let h = h * 2_f32.powf(-1. / p);
    (c - h, c + h)
}
pub(crate) fn board_supports(o: &IndoorObject, lo: Vec2, hi: Vec2, margin: f32) -> bool {
    let (c, h, p) = board_top(o);
    [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)]
        .into_iter()
        .all(|q| {
            let v = ((q - c).abs() + Vec2::splat(margin)) / h;
            v.x.powf(p) + v.y.powf(p) <= 1. + 1e-5
        })
}
