//! Place the unfolded print on the two outside boards and the exposed spine.
use super::*;
use crate::scene::procedural_indoor::materials::paper::jacket::SPINE;

pub(super) fn board(
    a: &mut Assembly,
    outline: &[Vec2],
    stock: (Surface, &str),
    print: &str,
    size: Vec3,
    tf: Transform,
    top: bool,
) {
    let (width, t, depth) = (size.x, size.y, size.z);
    let profile = Geometry::slab_profile(outline, t, t * 0.12);
    let g = a.part(stock.0, stock.1);
    let outside = if top { t * 0.5 } else { -t * 0.5 };
    g.slab_cap(outline, profile.1, Vec2::ZERO, -outside, &tf, |p| p);
    g.slab_edge(outline, t, profile, &tf);
    a.part(Surface::PrintedPaper, print).slab_cap(
        outline,
        profile.1,
        Vec2::ZERO,
        outside,
        &tf,
        |p| {
            let across = p.x / width + 0.5;
            Vec2::new(
                if top {
                    SPINE[1] + (1. - SPINE[1]) * across
                } else {
                    SPINE[0] * (1. - across)
                },
                p.y / depth + 0.5,
            )
        },
    );
}

pub(super) fn spine(g: &mut Geometry, size: Vec3, bevel: f32, tf: Transform) {
    g.mapped_cuboid(size, bevel, tf, |n, _, _, du, dv, _| {
        move |x, y| {
            if n == Vec3::NEG_X {
                Vec2::new(
                    SPINE[0] + (SPINE[1] - SPINE[0]) * (y / (2. * dv) + 0.5),
                    x / (2. * du) + 0.5,
                )
            } else {
                // Binding returns are covered by the boards or are narrow cut edges.
                Vec2::new(SPINE[0], 0.)
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_jacket_replaces_one_cap_and_reads_upright_in_each_orientation() {
        let size = Vec3::new(0.178, 0.0017, 0.245);
        let outline = cover_outline(size.x, size.z, 0.004);
        for top in [false, true] {
            let mut a = Assembly::default();
            board(
                &mut a,
                &outline,
                (Surface::Leather, "books"),
                "books#finish24#uv_local",
                size,
                Transform::IDENTITY,
                top,
            );
            let mut old = Geometry::default();
            old.profile_slab(&outline, size.y, size.y * 0.12, Transform::IDENTITY);
            let mut points = a
                .parts
                .values()
                .flat_map(|g| g.positions.iter().map(|p| p.map(f32::to_bits)))
                .collect::<Vec<_>>();
            points.sort();
            let mut old_points = old
                .positions
                .iter()
                .map(|p| p.map(f32::to_bits))
                .collect::<Vec<_>>();
            old_points.sort();
            assert_eq!(points, old_points);
            assert_eq!(
                a.parts.values().map(|g| g.indices.len()).sum::<usize>(),
                old.indices.len()
            );
            let print = &a.parts[&(Surface::PrintedPaper, "books#finish24#uv_local".to_owned())];
            for ((p, n), uv) in print.positions.iter().zip(&print.normals).zip(&print.uvs) {
                assert_eq!(n[1], if top { 1. } else { -1. });
                assert!((uv[1] - (p[2] / size.z + 0.5)).abs() < 1e-6);
                let across = if top {
                    (uv[0] - SPINE[1]) / (1. - SPINE[1])
                } else {
                    1. - uv[0] / SPINE[0]
                };
                assert!((across - (p[0] / size.x + 0.5)).abs() < 1e-6);
            }
        }
    }
}
