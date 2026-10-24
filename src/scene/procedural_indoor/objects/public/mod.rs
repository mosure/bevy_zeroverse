//! Bounded, multipart public-space assemblies. All parameters are sampled before
//! admission; construction, collision envelopes and catalog receipts share them.
pub mod banquette;
pub mod barrier;
pub mod casework;
pub mod dispenser;
#[cfg(test)]
mod diversity_tests;
pub mod education;
pub mod elevator;
mod exhibits;
pub mod furnishing;
mod hospitality;
#[cfg(test)]
mod repetition_tests;
pub mod sculpture;
pub mod service;
pub mod signage;
mod supports;
#[cfg(test)]
mod tests;
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
    let mut r = stream(seed, 8203);
    // Metric product ranges, not one XYZ multiplier for unrelated categories.
    // Sample before admission: support areas, navigation and labels see these sizes.
    let (x, y, z) = match kind {
        Elevator => (
            (elevator::MIN_FOOTPRINT.x, 2.90),
            (2.50, 2.95),
            (elevator::MIN_FOOTPRINT.y, 2.20),
        ),
        ServiceCounter => ((1.45, 3.0), (0.82, 1.10), (0.60, 0.90)),
        DisplayCase => ((0.80, 1.95), (0.88, 1.92), (0.48, 0.84)),
        TrayReturn => ((0.65, 1.25), (0.86, 1.55), (0.46, 0.76)),
        DrinkDispenser => ((0.25, 0.60), (0.37, 0.73), (0.28, 0.46)),
        QueueBarrier => ((0.85, 1.85), (0.76, 1.08), (0.28, 0.44)),
        Lectern => ((0.44, 0.88), (0.92, 1.30), (0.38, 0.66)),
        Locker => ((0.72, 1.80), (1.45, 2.10), (0.38, 0.60)),
        TeachingBoard => ((1.10, 2.90), (1.55, 2.20), (0.48, 0.82)),
        ExhibitPlinth => ((0.42, 1.12), (0.32, 1.35), (0.40, 1.08)),
        Sculpture => ((0.24, 0.78), (0.26, 1.05), (0.22, 0.70)),
        InformationStand => ((0.32, 0.72), (0.84, 1.55), (0.30, 0.58)),
        ExhibitionPanel => ((0.90, 2.35), (1.42, 2.40), (0.38, 0.70)),
        ExtractionHood => ((0.70, 1.85), (0.34, 0.85), (0.48, 0.78)),
        Banquette => ((1.30, 3.0), (0.88, 1.25), (0.60, 0.85)),
        _ => unreachable!("public program item"),
    };
    Vec3::new(
        r.random_range(x.0..x.1),
        r.random_range(y.0..y.1),
        r.random_range(z.0..z.1),
    )
}
pub fn build(a: &mut Assembly, o: &IndoorObject) {
    match o.kind {
        ObjectKind::Elevator => elevator::build(a, o),
        ObjectKind::QueueBarrier => barrier::build(a, o),
        ObjectKind::DisplayCase | ObjectKind::TrayReturn | ObjectKind::ExhibitPlinth => {
            casework::build(a, o)
        }
        ObjectKind::DrinkDispenser => dispenser::build(a, o),
        ObjectKind::ServiceCounter => service::counter(a, o),
        ObjectKind::ExtractionHood => service::hood(a, o),
        ObjectKind::Banquette => banquette::build(a, o),
        ObjectKind::Locker => education::build(a, o),
        ObjectKind::Lectern | ObjectKind::TeachingBoard => furnishing::build(a, o),
        ObjectKind::Sculpture | ObjectKind::InformationStand | ObjectKind::ExhibitionPanel => {
            exhibits::build(a, o)
        }
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
pub(super) fn append_at(a: &mut Assembly, child: Assembly, tf: Transform) {
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
