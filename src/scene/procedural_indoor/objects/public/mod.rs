//! Bounded, multipart public-space assemblies. All parameters are sampled before
//! admission; construction, collision envelopes and catalog receipts share them.
mod education;
mod exhibits;
mod hospitality;
#[cfg(test)]
mod tests;
mod transit;
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PublicItemProgram {
    pub bays: usize,
    pub shelves: usize,
    pub opening: f32,
    pub taper: f32,
    pub curvature: f32,
    pub contents: bool,
    pub glazed: bool,
    pub controls: usize,
    pub finish: Surface,
}
pub fn parameters(o: &IndoorObject) -> PublicItemProgram {
    let mut r = stream(o.seed, 8201);
    let mut finish = stream(o.finish_seed.unwrap_or(o.seed), 8202);
    PublicItemProgram {
        bays: r.random_range(2..=4),
        shelves: r.random_range(2..=4),
        opening: if r.random_bool(0.30) {
            0.
        } else {
            r.random_range(0.25..1.0)
        },
        taper: r.random_range(0.60..0.96),
        curvature: r.random_range(-0.9..0.9),
        contents: r.random_bool(0.92),
        glazed: r.random_bool(0.55),
        controls: r.random_range(1..=3),
        finish: [Surface::Wood, Surface::Metal, Surface::Plastic][finish.random_range(0..3)],
    }
}
pub fn dimensions(kind: ObjectKind, seed: u64) -> Vec3 {
    use ObjectKind::*;
    let size = Vec3::from_array(match kind {
        Elevator => [2.35, 2.75, 1.85],
        ServiceCounter => [2.2, 0.94, 0.74],
        DisplayCase => [1.4, 1.34, 0.72],
        TrayReturn => [0.88, 1.30, 0.62],
        DrinkDispenser => [0.36, 0.52, 0.36],
        QueueBarrier => [1.2, 0.92, 0.35],
        Lectern => [0.68, 1.18, 0.54],
        Locker => [1.2, 1.92, 0.48],
        TeachingBoard => [2.4, 1.95, 0.65],
        ExhibitPlinth => [0.85, 0.95, 0.85],
        Sculpture => [0.50, 0.70, 0.48],
        InformationStand => [0.45, 1.15, 0.40],
        ExhibitionPanel => [2.0, 2.18, 0.50],
        ExtractionHood => [1.4, 0.55, 0.64],
        Banquette => [2.1, 1.12, 0.68],
        _ => unreachable!("public program item"),
    });
    let mut r = stream(seed, 8203);
    size * Vec3::new(
        r.random_range(0.88..1.15),
        r.random_range(0.96..1.04),
        r.random_range(0.92..1.06),
    )
}
pub fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    match o.kind {
        ObjectKind::Elevator | ObjectKind::QueueBarrier => transit::build(a, o, &p),
        ObjectKind::ServiceCounter
        | ObjectKind::DisplayCase
        | ObjectKind::TrayReturn
        | ObjectKind::DrinkDispenser
        | ObjectKind::ExtractionHood
        | ObjectKind::Banquette => hospitality::build(a, o, &p),
        ObjectKind::Lectern | ObjectKind::Locker | ObjectKind::TeachingBoard => {
            education::build(a, o, &p)
        }
        ObjectKind::ExhibitPlinth
        | ObjectKind::Sculpture
        | ObjectKind::InformationStand
        | ObjectKind::ExhibitionPanel => exhibits::build(a, o, &p),
        _ => unreachable!("public program item"),
    }
}
/// Coordinates are fractions of the pre-admitted object envelope, with Y=0 at contact.
fn panel(
    a: &mut Assembly,
    o: &IndoorObject,
    surface: Surface,
    label: &str,
    p: [f32; 3],
    size: [f32; 3],
) {
    a.box_part(
        surface,
        label,
        Vec3::from_array(p) * o.size,
        Vec3::from_array(size) * o.size,
        0.004,
    );
}
fn rod(
    a: &mut Assembly,
    o: &IndoorObject,
    surface: Surface,
    label: &str,
    p: [f32; 3],
    q: [f32; 3],
    radius: f32,
) {
    a.part(surface, label).rod(
        Vec3::from_array(p) * o.size,
        Vec3::from_array(q) * o.size,
        radius,
    );
}
fn append_at(a: &mut Assembly, child: Assembly, tf: Transform) {
    for ((surface, label), mut g) in child.parts {
        for p in &mut g.positions {
            *p = tf.transform_point(Vec3::from_array(*p)).to_array();
        }
        for n in &mut g.normals {
            *n = (tf.rotation * Vec3::from_array(*n)).to_array();
        }
        a.part(surface, &label).append(g);
    }
}
