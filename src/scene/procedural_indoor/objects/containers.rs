//! Hollow containers with bounded, continuously sampled proportions.
use super::*;

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct BinProgram {
    pub squareness: f32,
    pub taper: f32,
    pub rim_width: f32,
    pub lid: u8,
    pub finish: Surface,
}
pub fn bin_parameters(o: &IndoorObject) -> BinProgram {
    let mut rng = stream(o.seed, 6310);
    BinProgram {
        squareness: rng.random_range(2.0..7.0),
        taper: rng.random_range(0.72..0.98),
        rim_width: rng.random_range(0.025..0.070),
        lid: rng.random_range(0..5),
        finish: [Surface::Metal, Surface::Plastic, Surface::Chrome][rng.random_range(0..3)],
    }
}
fn outline(size: Vec2, power: f32) -> Vec<Vec2> {
    (0..48)
        .map(|i| {
            let a = i as f32 * TAU / 48.;
            Vec2::new(
                a.cos().signum() * a.cos().abs().powf(2. / power),
                a.sin().signum() * a.sin().abs().powf(2. / power),
            ) * size
                * 0.5
        })
        .collect()
}
/// Consecutive rings form the outer wall, rolled rim and inner wall. No solid
/// top, coincident lid, or fake dark plane hides the opening.
fn shell(g: &mut Geometry, shape: &[Vec2], rings: &[(f32, f32)]) {
    for pair in rings.windows(2) {
        for i in 0..shape.len() {
            let j = (i + 1) % shape.len();
            let p = |k: usize, ring: (f32, f32)| {
                Vec3::new(shape[k].x * ring.0, ring.1, shape[k].y * ring.0)
            };
            let points = [p(i, pair[0]), p(i, pair[1]), p(j, pair[1]), p(j, pair[0])];
            let start = g.positions.len() as u32;
            // Smooth around the circumference, sharp at profile changes.
            for (k, v) in [i, i, j, j].into_iter().zip(points) {
                let tangent =
                    shape[(k + 1) % shape.len()] - shape[(k + shape.len() - 1) % shape.len()];
                let along = p(k, pair[1]) - p(k, pair[0]);
                let n = along.cross(Vec3::new(tangent.x, 0., tangent.y)).normalize();
                g.positions.push(v.to_array());
                g.normals.push(n.to_array());
                g.uvs
                    .push([k as f32 / shape.len() as f32 * shape[k].length() * TAU, v.y]);
            }
            g.indices
                .extend([start, start + 1, start + 2, start, start + 2, start + 3]);
        }
    }
}

pub(super) fn bin(a: &mut Assembly, o: &IndoorObject) {
    let p = bin_parameters(o);
    let s = o.size;
    let label = o.kind.class_name();
    let shape = outline(s.xz() * 0.96, p.squareness);
    let top = s.y * if p.lid == 1 || p.lid == 2 { 0.90 } else { 0.96 };
    let thickness = 0.022;
    shell(
        a.part(p.finish, label),
        &shape,
        &[
            (p.taper, 0.008),
            (1., top),
            (1. - thickness, top + 0.002),
            (p.taper - thickness, 0.012),
        ],
    );
    a.part(p.finish, label).profile_slab(
        &shape.iter().map(|v| *v * p.taper).collect::<Vec<_>>(),
        0.012,
        0.001,
        Transform::from_xyz(0., 0.006, 0.),
    );
    shell(
        a.part(Surface::Plastic, label),
        &shape,
        &[
            (1.005, top - 0.018),
            (1.01, top + 0.006),
            (1. - p.rim_width, top + 0.006),
            (1. - p.rim_width, top - 0.018),
        ],
    );
    if p.lid == 1 || p.lid == 2 {
        // Pedal lid and sensor lid share a separated hinge seam.
        a.part(p.finish, label).profile_slab(
            &shape,
            s.y * 0.05,
            0.003,
            Transform::from_xyz(0., top + s.y * 0.03 + 0.009, 0.),
        );
        if p.lid == 1 {
            a.box_part(
                Surface::Rubber,
                label,
                Vec3::new(0., s.y * 0.055, s.z * 0.43),
                Vec3::new(s.x * 0.35, s.y * 0.04, s.z * 0.13),
                0.002,
            );
        } else {
            a.box_part(
                Surface::Ink,
                label,
                Vec3::new(0., top + s.y * 0.055 + 0.010, 0.),
                Vec3::new(s.x * 0.23, 0.001, s.z * 0.10),
                0.0003,
            );
        }
    } else if p.lid == 3 {
        // Recycling divider: both chambers remain genuinely open.
        a.box_part(
            Surface::Plastic,
            label,
            Vec3::new(0., top * 0.51, 0.),
            Vec3::new(0.008, top * 0.97, s.z * 0.68),
            0.002,
        );
    } else if p.lid == 4 {
        // Ribbed moulding, with a radius outside the wall and below the lip.
        for i in 0..16 {
            let q = shape[i * 3];
            a.part(p.finish, label).rod(
                Vec3::new(q.x * p.taper * 1.004, 0.025, q.y * p.taper * 1.004),
                Vec3::new(q.x * 1.005, top - 0.025, q.y * 1.005),
                0.0015,
            );
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct TissueProgram {
    pub box_height: f32,
    pub opening: Vec2,
    pub lean: f32,
    pub fold: f32,
    pub phase: f32,
    pub finish: Surface,
}
pub fn tissue_parameters(o: &IndoorObject) -> TissueProgram {
    let mut rng = stream(o.seed, 6320);
    TissueProgram {
        box_height: rng.random_range(0.48..0.74),
        opening: Vec2::new(rng.random_range(0.38..0.64), rng.random_range(0.13..0.25)),
        lean: rng.random_range(-0.14..0.14),
        fold: rng.random_range(0.06..0.18),
        phase: rng.random_range(0.0..TAU),
        finish: [
            Surface::Art,
            Surface::Paper,
            Surface::Wood,
            Surface::FabricAlt,
            Surface::Plastic,
        ][rng.random_range(0..5)],
    }
}
pub(super) fn tissues(a: &mut Assembly, o: &IndoorObject) {
    let p = tissue_parameters(o);
    let s = o.size;
    let label = o.kind.class_name();
    let h = s.y * p.box_height;
    let t = (s.y * 0.028).clamp(0.001, 0.003);
    let bevel = t * 0.25;
    a.box_part(
        p.finish,
        label,
        Vec3::Y * t * 0.5,
        Vec3::new(s.x, t, s.z),
        bevel,
    );
    for sign in [-1., 1.] {
        a.box_part(
            p.finish,
            label,
            Vec3::new(sign * (s.x - t) * 0.5, h * 0.5, 0.),
            Vec3::new(t, h - t, s.z),
            bevel,
        );
        a.box_part(
            p.finish,
            label,
            Vec3::new(0., h * 0.5, sign * (s.z - t) * 0.5),
            Vec3::new(s.x - 2. * t, h - t, t),
            bevel,
        );
    }
    let hole = s.xz() * p.opening;
    // Four top panels surround a real rectangular slot.
    for sign in [-1., 1.] {
        a.box_part(
            p.finish,
            label,
            Vec3::new(sign * (s.x + hole.x) * 0.25, h, 0.),
            Vec3::new((s.x - hole.x) * 0.5, t, s.z),
            bevel,
        );
        a.box_part(
            p.finish,
            label,
            Vec3::new(0., h, sign * (s.z + hole.y) * 0.25),
            Vec3::new(hole.x, t, (s.z - hole.y) * 0.5),
            bevel,
        );
    }
    let g = a.part(Surface::Paper, "paper");
    // A pinched, pleated sheet opens above the slot. Finite thickness and
    // separate front/back normals avoid translucent planes and flicker.
    let sheet = |u: f32, v: f32| {
        let width = s.x * (0.13 + 0.14 * v);
        Vec3::new(
            u * width + s.x * p.lean * v * v,
            h - 0.001 + (s.y - h - 0.003) * v * (0.92 + 0.07 * (PI * u + p.phase).cos()),
            hole.y * 0.22 * (u * PI * 2. + p.phase).sin()
                + s.z * p.fold * v * (PI * u + p.phase).sin(),
        )
    };
    for sign in [-1., 1.] {
        let start = g.positions.len() as u32;
        for y in 0..=8 {
            for x in 0..=10 {
                let u = x as f32 / 5. - 1.;
                let v = y as f32 / 8.;
                let normal = (sheet(u + 0.001, v) - sheet(u - 0.001, v))
                    .cross(sheet(u, v + 0.001) - sheet(u, v - 0.001))
                    .normalize();
                g.positions
                    .push((sheet(u, v) + normal * sign * 0.00018).to_array());
                g.normals.push((normal * sign).to_array());
                g.uvs.push([u * s.x * 0.25, v * (s.y - h)]);
            }
        }
        for y in 0..8 {
            for x in 0..10 {
                let i = start + y * 11 + x;
                g.indices.extend(if sign > 0. {
                    [i, i + 1, i + 12, i, i + 12, i + 11]
                } else {
                    [i, i + 12, i + 1, i, i + 11, i + 12]
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        layout::{IndoorLayout, IndoorManifest},
        validation,
    };
    #[test]
    fn container_and_fixture_programs_are_bounded_replayable_and_diverse() {
        let mut scene =
            IndoorManifest::generate_with_humans(42, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let template = scene.objects[0].clone();
        scene.objects.clear();
        let mut bins = std::collections::BTreeSet::new();
        let mut switches = std::collections::BTreeSet::new();
        let mut tissues = std::collections::BTreeSet::new();
        for seed in 0..128 {
            for (kind, size) in [
                (ObjectKind::TrashCan, Vec3::new(0.27, 0.32, 0.26)),
                (ObjectKind::TrashCan, Vec3::new(0.46, 0.68, 0.42)),
                (ObjectKind::TissueBox, Vec3::new(0.14, 0.08, 0.10)),
                (ObjectKind::TissueBox, Vec3::new(0.24, 0.14, 0.16)),
                (ObjectKind::WallOutlet, Vec3::new(0.075, 0.10, 0.018)),
                (ObjectKind::WallOutlet, Vec3::new(0.225, 0.125, 0.018)),
                (ObjectKind::LightSwitch, Vec3::new(0.075, 0.10, 0.018)),
                (ObjectKind::LightSwitch, Vec3::new(0.225, 0.125, 0.018)),
            ] {
                let mut o = template.clone();
                o.kind = kind;
                o.size = size;
                o.seed = seed;
                o.id = scene.objects.len();
                o.position = Vec3::ZERO;
                o.support = None;
                o.solid = false;
                assert_eq!(bin_parameters(&o), bin_parameters(&o));
                bins.insert(bin_parameters(&o).lid);
                switches.insert(wall::parameters(&o).style);
                tissues.insert(serde_json::to_string(&tissue_parameters(&o)).unwrap());
                let a = build_object(&o);
                let (lo, hi) = a.bounds();
                assert!(
                    lo.y < 0.003,
                    "missing support contact {kind:?} {seed}: {lo}"
                );
                assert!(
                    a.parts.values().map(|g| g.indices.len() / 3).sum::<usize>() < 5000,
                    "unbounded mesh budget"
                );
                if kind == ObjectKind::TissueBox {
                    assert!(hi.y > size.y * 0.94, "tissue disappeared");
                }
                scene.objects.push(o);
            }
        }
        validation::validate_geometry(&scene).unwrap();
        assert_eq!(bins.len(), 5);
        assert_eq!(switches.len(), 6);
        assert_eq!(tissues.len(), 128);
    }
}
