//! Hollow desk cups and collision-separated writing implements. Geometry and
//! contents have independent streams; a bounded search never forces an overlap.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct PenHolderProgram {
    pub height_fraction: f32,
    pub base_radius_fraction: f32,
    pub mouth_radius_fraction: f32,
    pub belly_height: f32,
    pub wall_fraction: f32,
    pub sides: u32,
    pub finish: Surface,
    pub requested_pens: usize,
}

pub fn parameters(o: &IndoorObject) -> PenHolderProgram {
    let mut r = stream(o.seed, 0x50454e435550);
    PenHolderProgram {
        height_fraction: r.random_range(0.45..0.78),
        base_radius_fraction: r.random_range(0.68..1.0),
        mouth_radius_fraction: r.random_range(0.78..1.0),
        belly_height: r.random_range(0.25..0.72),
        wall_fraction: r.random_range(0.06..0.14),
        sides: [4, 6, 8, 28, 32, 32][r.random_range(0..6)],
        finish: [
            Surface::Wood,
            Surface::Metal,
            Surface::Plastic,
            Surface::Ceramic,
        ][r.random_range(0..4)],
        requested_pens: r.random_range(0..=9),
    }
}

#[derive(Debug, serde::Serialize)]
pub struct Pen {
    pub base: Vec3,
    pub top: Vec3,
    pub radius: f32,
    pub finish: Surface,
}

pub fn contents(o: &IndoorObject, p: &PenHolderProgram) -> Vec<Pen> {
    let mut r = stream(o.seed, 0x50454e46494c4c);
    let size = o.size;
    let radius = size.x.min(size.z) * 0.47;
    // Inscribed disk also fits square/hexagonal cups, including the narrow foot.
    let inner = radius
        * (p.base_radius_fraction.min(p.mouth_radius_fraction) - p.wall_fraction)
        * (std::f32::consts::PI / p.sides as f32).cos();
    let floor = (size.y * p.height_fraction * 0.06).min(0.006);
    let mut pens: Vec<Pen> = Vec::with_capacity(p.requested_pens);
    for _ in 0..p.requested_pens * 8 {
        if pens.len() == p.requested_pens {
            break;
        }
        let angle = r.random_range(0.0..std::f32::consts::TAU);
        let distance = r.random_range(0.0..1.0_f32).sqrt() * inner * 0.65;
        let foot = Vec2::new(angle.cos(), angle.sin()) * distance;
        let drift = Vec2::new(r.random_range(-0.14..0.14), r.random_range(-0.14..0.14)) * inner;
        let pen = Pen {
            base: Vec3::new(foot.x, floor, foot.y),
            top: Vec3::new(
                foot.x + drift.x,
                size.y * r.random_range(0.84..0.98),
                foot.y + drift.y,
            ),
            radius: r.random_range(0.002_f32..0.0038).min(inner * 0.1),
            finish: [Surface::Wood, Surface::Plastic, Surface::Ink][r.random_range(0..3)],
        };
        if pens.iter().all(|other| {
            super::super::objects::equipment::wires::segment_distance_squared(
                pen.base, pen.top, other.base, other.top,
            ) > ((pen.radius + other.radius + 0.0006) as f64).powi(2)
        }) {
            pens.push(pen);
        }
    }
    pens
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let radius = s.x.min(s.z) * 0.47;
    let height = s.y * p.height_fraction;
    let floor = (height * 0.06).min(0.006);
    let base = radius * p.base_radius_fraction;
    let mouth = radius * p.mouth_radius_fraction;
    let wall = radius * p.wall_fraction;
    a.part(p.finish, o.kind.class_name()).lathe(
        &[
            (0., 0.),
            (base, 0.),
            (radius, height * p.belly_height),
            (mouth, height),
            (mouth - wall, height),
            (radius - wall, height * p.belly_height),
            (base - wall, floor),
            (0., floor),
        ],
        p.sides,
        Transform::from_scale(Vec3::new(s.x / s.x.min(s.z), 1., s.z / s.x.min(s.z))),
    );
    for pen in contents(o, &p) {
        a.part(pen.finish, o.kind.class_name()).rod(
            pen.base,
            pen.base.lerp(pen.top, 0.93),
            pen.radius,
        );
        // A separate, attached cap, kept within the sampled rod envelope.
        a.part(Surface::Ink, o.kind.class_name()).rod(
            pen.base.lerp(pen.top, 0.93),
            pen.top,
            pen.radius * 1.03,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cups_are_hollow_and_contents_are_bounded_and_separated() {
        use crate::scene::procedural_indoor::objects::{audit, build_object, equipment::wires};
        let mut counts = std::collections::BTreeSet::new();
        let mut finishes = std::collections::BTreeSet::new();
        for seed in 0..128 {
            let mut o = audit::fixture(ObjectKind::PenHolder, seed);
            o.size =
                [Vec3::new(0.065, 0.12, 0.065), Vec3::new(0.115, 0.21, 0.115)][seed as usize % 2];
            let p = parameters(&o);
            let pens = contents(&o, &p);
            counts.insert(pens.len());
            finishes.insert(format!("{:?}", p.finish));
            audit::check(&o, &build_object(&o)).unwrap();
            let inner = o.size.x.min(o.size.z)
                * 0.47
                * (p.base_radius_fraction.min(p.mouth_radius_fraction) - p.wall_fraction)
                * (std::f32::consts::PI / p.sides as f32).cos();
            for (i, pen) in pens.iter().enumerate() {
                assert!(
                    pen.base.xz().length().max(pen.top.xz().length()) + pen.radius * 1.03 < inner
                );
                for other in &pens[..i] {
                    assert!(
                        wires::segment_distance_squared(pen.base, pen.top, other.base, other.top)
                            > ((pen.radius + other.radius) as f64 * 1.03).powi(2)
                    );
                }
            }
        }
        assert_eq!(finishes.len(), 4);
        assert!(counts.contains(&0) && counts.len() >= 8, "{counts:?}");
    }
}
