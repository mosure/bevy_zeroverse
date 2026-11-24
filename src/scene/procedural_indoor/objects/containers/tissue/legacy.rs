//! Archived box variants use their original rectangular slot and sheet.
use super::*;

pub(super) fn parameters(o: &IndoorObject) -> TissueProgram {
    let mut rng = stream(o.seed, 6320);
    TissueProgram {
        dispenser: None,
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
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &TissueProgram) {
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
