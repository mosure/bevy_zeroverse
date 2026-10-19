//! Connected sculptural constructions with independent form and finish streams.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct SculptureProgram {
    pub form: u32,
    pub elements: usize,
    pub turns: f32,
    pub bend: f32,
    pub phase: f32,
    pub breadth: f32,
    pub taper: f32,
    pub thickness: f32,
    pub surface: Surface,
    pub base: Surface,
}
pub fn parameters(o: &IndoorObject) -> SculptureProgram {
    let mut r = stream(o.seed, 8250);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 8251);
    SculptureProgram {
        form: r.random_range(0..5),
        elements: r.random_range(3..=8),
        turns: r.random_range(0.65..1.8),
        bend: r.random_range(-0.16..0.16),
        phase: r.random_range(0.0..TAU),
        breadth: r.random_range(0.18..0.34),
        taper: r.random_range(0.35..0.85),
        thickness: r.random_range(0.035..0.075),
        surface: [
            Surface::Ceramic,
            Surface::Metal,
            Surface::Chrome,
            Surface::Wood,
            Surface::Concrete,
        ][f.random_range(0..5)],
        base: [Surface::Metal, Surface::Wood, Surface::Concrete][f.random_range(0..3)],
    }
}

/// Fit tube thickness to the realized bends, including the closed-loop seam.
/// A thick sweep around a narrow ellipse can invert its inner triangles even
/// when every centerline point and vertex stays inside the admitted envelope.
fn sweep_radius(points: &[Vec3], desired: f32) -> f32 {
    let closed = points[0].distance_squared(*points.last().unwrap()) < 1e-12;
    let count = points.len() - usize::from(closed);
    let mut radius = desired;
    let range = if closed { 0..count } else { 1..count - 1 };
    for i in range {
        let a = points[i] - points[(i + count - 1) % count];
        let b = points[(i + 1) % count] - points[i];
        let cosine = a.normalize().dot(b.normalize()).clamp(-0.9999, 1.);
        let tan_half = ((1. - cosine) / (1. + cosine)).sqrt();
        if tan_half > 1e-5 {
            radius = radius.min(0.20 * a.length().min(b.length()) / tan_half);
        }
    }
    radius
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let label = o.kind.class_name();
    let radius = s.min_element() * p.thickness;
    panel(a, o, p.base, label, [0., 0.025, 0.], [0.94, 0.05, 0.94]);
    match p.form {
        0 => {
            // Joined stone/ceramic totem: a visible spine supports each lobe.
            let at = |t: f32| {
                Vec3::new(
                    p.bend * (TAU * t).sin(),
                    0.10 + 0.78 * t,
                    p.bend * (TAU * t + p.phase).sin(),
                ) * s
            };
            let pts: Vec<_> = (0..=p.elements)
                .map(|i| at(i as f32 / p.elements as f32))
                .collect();
            let radius = sweep_radius(&pts, radius);
            a.part(p.surface, label).tube(&pts, radius, 10);
            a.part(p.base, label)
                .rod(Vec3::new(pts[0].x, s.y * 0.025, pts[0].z), pts[0], radius);
            for i in 0..p.elements {
                let t = (i as f32 + 0.5) / p.elements as f32;
                a.part(p.surface, label).ellipsoid(
                    Vec3::new(
                        s.x * p.breadth * 0.70,
                        s.y * 0.36 / p.elements as f32,
                        s.z * p.breadth * 0.65,
                    ),
                    Transform::from_translation(at(t)),
                );
            }
        }
        1 => {
            // Welded continuous loop; depth undulation changes with the same curve.
            let pts: Vec<_> = (0..=48)
                .map(|i| {
                    let t = i as f32 / 48. * TAU;
                    Vec3::new(
                        p.breadth * t.sin(),
                        0.51 - 0.38 * t.cos(),
                        p.bend * (2. * t + p.phase).sin(),
                    ) * s
                })
                .collect();
            let radius = sweep_radius(&pts, radius);
            a.part(p.surface, label).tube(&pts, radius, 10);
            a.part(p.base, label)
                .rod(Vec3::new(pts[0].x, s.y * 0.025, pts[0].z), pts[0], radius);
        }
        2 => {
            // A folded, faceted column, distinct from the rounded loop/totem.
            let pts: Vec<_> = (0..=p.elements)
                .map(|i| {
                    let t = i as f32 / p.elements as f32;
                    let phase = t * p.turns * TAU + p.phase;
                    Vec3::new(p.bend * phase.sin(), 0.12 + 0.74 * t, p.bend * phase.cos()) * s
                })
                .collect();
            let radius = sweep_radius(&pts, s.min_element() * p.breadth * 0.26);
            a.part(p.surface, label).tube(&pts, radius, 4);
            a.part(p.base, label)
                .rod(Vec3::new(pts[0].x, s.y * 0.025, pts[0].z), pts[0], radius);
        }
        3 => {
            // Turned solid form, with a neck and shoulder continuously controlled.
            let r = s.x.min(s.z) * p.breadth;
            a.part(p.surface, label).lathe(
                &[
                    (0., s.y * 0.05),
                    (r * 0.45, s.y * 0.05),
                    (r * 0.65, s.y * 0.16),
                    (r, s.y * 0.37),
                    (r * 0.87, s.y * 0.62),
                    (r * p.taper, s.y * 0.80),
                    (r * p.taper * 1.08, s.y * 0.95),
                    (0., s.y * 0.95),
                ],
                24,
                Transform::IDENTITY,
            );
        }
        _ => {
            // Cantilevered kinetic-style discs remain attached to their armature.
            a.part(p.base, label)
                .rod(Vec3::Y * s.y * 0.05, Vec3::Y * s.y * 0.94, radius * 0.65);
            for i in 0..p.elements {
                let t = (i as f32 + 0.5) / p.elements as f32;
                let angle = p.phase + t * p.turns * TAU;
                let center = Vec3::new(
                    angle.cos() * p.breadth * 0.65,
                    0.15 + t * 0.68,
                    angle.sin() * p.breadth * 0.65,
                ) * s;
                a.part(p.base, label)
                    .rod(Vec3::new(0., center.y, 0.), center, radius * 0.50);
                a.part(p.surface, label).ellipsoid(
                    Vec3::new(s.x * 0.15, s.y * 0.022, s.z * 0.15),
                    Transform::from_translation(center),
                );
            }
        }
    }
}
