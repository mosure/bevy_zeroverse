//! Small kitchen products use their own construction streams. The outer metric
//! envelope is sampled before admission; every fold, pump and handle stays in it.
pub mod knives;
use super::*;
pub mod bread;
pub mod dispenser;
pub mod rack;
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
}
pub fn board_parameters(o: &IndoorObject) -> BoardProgram {
    let mut r = stream(o.seed, 0x435554424f415244);
    BoardProgram {
        roundness: r.random_range(2.0..10.),
        body_fraction: r.random_range(0.68..0.86),
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
    let width = s.x * if p.handle { p.body_fraction } else { 1. };
    let cx = (width - s.x) * 0.5;
    let half = Vec2::new(width, s.z) * 0.5;
    let edge = s.y * p.edge_fraction;
    let inset = Vec2::splat(edge).min(half * 0.05);
    a.part(p.finish, o.kind.class_name()).housing(
        &[
            (0., half - inset),
            (edge, half),
            (s.y - edge, half),
            (s.y, half - inset),
        ],
        p.roundness,
        Transform::from_xyz(cx, 0., 0.),
    );
    if p.handle {
        // A real attached end-grain handle, partly embedded in the board edge.
        let start = cx + width * 0.40;
        let end = s.x * 0.5;
        a.box_part(
            p.finish,
            o.kind.class_name(),
            Vec3::new((start + end) * 0.5, s.y * 0.5, 0.),
            Vec3::new(end - start, s.y * 0.82, s.z * p.handle_width),
            edge.min(0.004),
        );
    }
}
