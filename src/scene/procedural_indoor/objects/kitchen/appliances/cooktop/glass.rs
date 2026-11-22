use super::*;
use crate::scene::procedural_indoor::{
    envelope::polygon,
    materials::cooktop::{GLASS_LABEL, PRINT_RADIUS},
};

/// Circular zones can have overlapping bounding rectangles. The power bisector
/// between two disjoint disks separates both disks and partitions the top into
/// convex, gap-free UV charts, each with one complete printed ring.
fn zone_cell(cap: &[Vec2], burner: &Burner, others: &[Burner]) -> Vec<Vec2> {
    let mut cell = cap.to_vec();
    for other in others.iter().filter(|b| b.center != burner.center) {
        let normal = burner.center - other.center;
        let offset = 0.5
            * (normal.dot(burner.center + other.center) - burner.radius_m.powi(2)
                + other.radius_m.powi(2));
        cell = polygon::clip(&cell, normal, offset);
    }
    cell
}

pub(super) fn build(a: &mut Assembly, size: Vec3, base: Vec3, burners: &[Burner]) {
    let half = size.xz() * 0.5;
    let outline = [
        -half,
        Vec2::new(half.x, -half.y),
        half,
        Vec2::new(-half.x, half.y),
    ];
    let thickness = size.y * 0.24;
    let g = a.part(Surface::Ceramic, GLASS_LABEL);
    let start = g.positions.len();
    let mut charts = 0..0;
    g.profile_slab_with_top(
        &outline,
        thickness,
        0.0025,
        Transform::from_translation(base + Vec3::Y * (thickness * 0.5)),
        |g, outline, inset, y, tf| {
            let cap: Vec<_> = outline.iter().map(|p| p * inset).collect();
            charts.start = g.positions.len();
            if burners.is_empty() {
                g.slab_cap(&cap, 1., Vec2::ZERO, y, tf, |p| p + Vec2::splat(2.));
            }
            for burner in burners {
                let cell = zone_cell(&cap, burner, burners);
                g.slab_cap(&cell, 1., burner.center, y, tf, |point| {
                    (point - burner.center) * (PRINT_RADIUS / burner.radius_m) + 0.5
                });
            }
            charts.end = g.positions.len();
        },
    );
    // Wrapped metric sidewall coordinates would repeat print down the bevel.
    // These vertices use the unprinted border, with valid UV derivatives.
    for uv in &mut g.uvs[start..charts.start] {
        uv[0] += 2.;
    }
    for uv in &mut g.uvs[charts.end..] {
        uv[0] += 2.;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::objects::{audit, kitchen};

    #[test]
    fn print_charts_tile_one_flat_surface_and_contain_each_variable_radius() {
        for seed in 0..512 {
            let o = audit::fixture(ObjectKind::Cooktop, seed);
            let mut p = kitchen::parameters(&o);
            p.gas = false;
            let plan = super::super::parameters(seed, o.size, &p);
            let mut a = Assembly::default();
            build(&mut a, o.size, Vec3::ZERO, &plan.burners);
            let g = &a.parts[&(Surface::Ceramic, GLASS_LABEL.to_owned())];
            let top = o.size.y * 0.24;
            let half = o.size.xz() * 0.5;
            let inset = 1. - 0.0025 / half.length();
            let cap = [
                -half,
                Vec2::new(half.x, -half.y),
                half,
                Vec2::new(-half.x, half.y),
            ]
            .map(|p| p * inset);
            let mut area = 0.;
            for b in &plan.burners {
                let cell = zone_cell(&cap, b, &plan.burners);
                assert!(polygon::contains(&cell, b.center, b.radius_m - 0.0001));
                area += polygon::area(&cell);
            }
            assert!((area - polygon::area(&cap)).abs() < 0.00001);
            let triangles = g
                .indices
                .as_chunks::<3>()
                .0
                .iter()
                .filter_map(|tri| {
                    let [a, b, c] =
                        [tri[0], tri[1], tri[2]].map(|i| Vec3::from_array(g.positions[i as usize]));
                    ([a, b, c].iter().all(|p| (p.y - top).abs() < 1e-7))
                        .then_some((b - a).cross(c - a).y * 0.5)
                })
                .collect::<Vec<_>>();
            assert!(triangles.iter().all(|area| *area > 0.));
            assert!((triangles.iter().sum::<f32>() - area).abs() < 0.00001);
            // The print costs a handful of coplanar, adjacent triangles per zone.
            assert!(g.indices.len() / 3 < 96);
        }
    }
}
