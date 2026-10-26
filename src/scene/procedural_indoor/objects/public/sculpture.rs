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
    pub base_width: Vec2,
    pub base_height: f32,
}
pub fn parameters(o: &IndoorObject) -> SculptureProgram {
    let mut r = stream(o.seed, 8250);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 8251);
    SculptureProgram {
        form: r.random_range(0..9),
        elements: r.random_range(2..=9),
        turns: r.random_range(0.65..1.8),
        bend: r.random_range(-0.16..0.16),
        phase: r.random_range(0.0..TAU),
        breadth: r.random_range(0.14..0.35),
        taper: r.random_range(0.35..0.85),
        thickness: r.random_range(0.035..0.075),
        surface: [
            Surface::Ceramic,
            Surface::Metal,
            Surface::Chrome,
            Surface::Wood,
            Surface::Concrete,
        ][f.random_range(0..5)],
        base_width: Vec2::new(r.random_range(0.64..0.94), r.random_range(0.64..0.94)),
        base_height: r.random_range(0.015..0.05),
        base: [Surface::Metal, Surface::Wood, Surface::Concrete][f.random_range(0..3)],
    }
}

/// Fit tube thickness to the realized bends, including the closed-loop seam.
/// A thick sweep around a narrow ellipse can invert its inner triangles even
/// when every centerline point and vertex stays inside the admitted envelope.
/// Tilted cross-sections must also stay above the sculpture's support plane.
fn sweep_radius(points: &[Vec3], desired: f32) -> f32 {
    let closed = points[0].distance_squared(*points.last().unwrap()) < 1e-12;
    let count = points.len() - usize::from(closed);
    let mut radius = desired;
    for i in 0..count {
        // Match Geometry::tube's endpoint and centered tangents. The vertical
        // extent of its circular cross-section is r * |tangent x up|,
        // independent of the transported frame or polygonal side count.
        let before = if closed {
            (i + count - 1) % count
        } else {
            i.saturating_sub(1)
        };
        let after = if closed {
            (i + 1) % count
        } else {
            (i + 1).min(count - 1)
        };
        let tangent = (points[after] - points[before]).normalize();
        let vertical_extent = tangent.cross(Vec3::Y).length();
        if vertical_extent > 1e-5 {
            radius = radius.min(points[i].y / vertical_extent * (1. - 4. * f32::EPSILON));
        }
    }
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
    panel(
        a,
        o,
        p.base,
        label,
        [0., p.base_height * 0.5, 0.],
        [p.base_width.x, p.base_height, p.base_width.y],
    );
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
            // Lobe count must not undersample the curved armature. Sparse
            // samples can turn adjacent sweep frames through a large angle
            // and invert triangles in squat, wide sculptures.
            let spine_segments = (p.elements * 4).max(24);
            let pts: Vec<_> = (0..=spine_segments)
                .map(|i| at(i as f32 / spine_segments as f32))
                .collect();
            let radius = sweep_radius(&pts, radius);
            a.part(p.surface, label).tube(&pts, radius, 10);
            a.part(p.base, label).rod(
                Vec3::new(pts[0].x, s.y * p.base_height * 0.5, pts[0].z),
                pts[0],
                radius,
            );
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
            a.part(p.base, label).rod(
                Vec3::new(pts[0].x, s.y * p.base_height * 0.5, pts[0].z),
                pts[0],
                radius,
            );
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
            a.part(p.base, label).rod(
                Vec3::new(pts[0].x, s.y * p.base_height * 0.5, pts[0].z),
                pts[0],
                radius,
            );
        }
        3 => {
            // Turned solid form, with a neck and shoulder continuously controlled.
            let r = s.x.min(s.z) * p.breadth;
            a.part(p.surface, label).lathe(
                &[
                    (0., s.y * p.base_height),
                    (r * 0.45, s.y * p.base_height),
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
        4 => {
            // Cantilevered kinetic-style discs remain attached to their armature.
            a.part(p.base, label).rod(
                Vec3::Y * s.y * p.base_height,
                Vec3::Y * s.y * 0.94,
                radius * 0.65,
            );
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
        5 => {
            // Continuous freestanding arch, with both ends fixed into its base.
            let points: Vec<_> = (0..=40)
                .map(|i| {
                    let t = i as f32 / 40. * PI;
                    Vec3::new(
                        t.cos() * p.breadth.min(p.base_width.x * 0.45 - 0.015),
                        p.base_height + t.sin() * 0.86,
                        p.bend * (2. * t).sin(),
                    ) * s
                })
                .collect();
            a.part(p.surface, label).tube(
                &points,
                sweep_radius(
                    &points,
                    (s.min_element() * p.thickness * 1.5).min(s.y * (0.99 - p.base_height - 0.86)),
                ),
                8,
            );
        }
        6 => {
            // Stacked cubist volumes: adjacent layers overlap vertically and
            // remain tied to a central armature rather than levitating.
            let n = p.elements.min(6);
            let step = s.y * 0.82 / n as f32;
            a.part(p.base, label)
                .rod(Vec3::Y * s.y * p.base_height, Vec3::Y * s.y * 0.91, radius);
            for i in 0..n {
                let t = (i as f32 + 0.5) / n as f32;
                let angle = p.phase + t * p.turns * PI;
                let width = s.x.min(s.z) * (0.28 + p.breadth * (1. - t * p.taper));
                a.part(p.surface, label).cuboid(
                    Vec3::new(width, step * 1.03, width * (0.55 + p.taper * 0.4)),
                    width * 0.04,
                    Transform::from_xyz(
                        s.x * p.bend * t * 0.4,
                        s.y * 0.06 + step * (i as f32 + 0.5),
                        s.z * p.bend * t * 0.4,
                    )
                    .with_rotation(Quat::from_rotation_y(angle)),
                );
            }
        }
        7 => {
            // Welded branching construction. Every broad plate has a real strut.
            a.part(p.base, label)
                .rod(Vec3::Y * s.y * p.base_height, Vec3::Y * s.y * 0.92, radius);
            for i in 0..p.elements {
                let t = (i as f32 + 0.5) / p.elements as f32;
                let angle = p.phase + t * p.turns * TAU;
                let root = Vec3::Y * s.y * (0.10 + t * 0.63);
                let tip = root
                    + Vec3::new(
                        angle.cos() * s.x * 0.27,
                        s.y * 0.10,
                        angle.sin() * s.z * 0.27,
                    );
                a.part(p.surface, label).rod(root, tip, radius * 0.6);
                a.part(p.surface, label).ellipsoid(
                    // Rotating an anisotropic plate must not exchange the
                    // short reservation axis with the long object's radius.
                    Vec3::new(s.x.min(s.z) * 0.14, s.y * 0.04, s.x.min(s.z) * 0.08),
                    Transform::from_translation(tip).with_rotation(Quat::from_rotation_y(angle)),
                );
            }
        }
        _ => {
            // Radial sheet sculpture: taper, twist and spacing change its silhouette.
            a.part(p.base, label)
                .rod(Vec3::Y * s.y * p.base_height, Vec3::Y * s.y * 0.91, radius);
            for i in 0..p.elements {
                let t = (i as f32 + 0.5) / p.elements as f32;
                let breadth = s.x.min(s.z) * (0.38 + p.breadth * (PI * t).sin());
                a.part(p.surface, label).cuboid(
                    Vec3::new(
                        breadth,
                        s.y * 0.64 / p.elements as f32,
                        s.x.min(s.z) * p.thickness,
                    ),
                    0.002,
                    Transform::from_xyz(0., s.y * (0.13 + t * 0.73), 0.)
                        .with_rotation(Quat::from_rotation_y(p.phase + t * p.turns * TAU)),
                );
            }
        }
    }
}
