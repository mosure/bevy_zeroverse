//! Metric slab charts follow the installed run, including the sink cutout and folded ends.
use super::*;
use crate::scene::procedural_indoor::materials::worktop_finish;

pub(super) const RUN_THICKNESS: f32 = 0.034;
pub(super) const CUPBOARD_THICKNESS: f32 = 0.032;

fn origin(o: &IndoorObject) -> Vec2 {
    (Quat::from_rotation_y(-o.yaw) * o.position).xz()
}

fn chart(g: &mut Geometry, range: std::ops::Range<usize>, pos: Vec2, top: f32, outward: Vec3) {
    for (v, uv) in g.positions[range.clone()].iter().zip(&mut g.uvs[range]) {
        let p = Vec3::from_array(*v);
        *uv = (pos + p.xz() + outward.xz() * (top - p.y)).to_array();
    }
}

pub(super) fn solid(
    a: &mut Assembly,
    o: &IndoorObject,
    surface: Surface,
    center: Vec3,
    size: Vec3,
    bevel: f32,
    top: f32,
) {
    let g = a.part(surface, worktop_finish::label(o, surface));
    let start = g.positions.len();
    g.cuboid(size, bevel, Transform::from_translation(center));
    // Cuboid emits six separate charts in this order, including bevel vertices.
    let per_face = (g.positions.len() - start) / 6;
    let pos = origin(o);
    for (face, outward) in [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ]
    .into_iter()
    .enumerate()
    {
        let first = start + face * per_face;
        chart(g, first..first + per_face, pos, top, outward);
    }
}

fn corners((lo, hi): (Vec2, Vec2)) -> [Vec2; 4] {
    [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)]
}

pub(super) fn pierced(
    a: &mut Assembly,
    o: &IndoorObject,
    surface: Surface,
    outer: (Vec2, Vec2),
    aperture: (Vec2, Vec2),
    top: f32,
) {
    let g = a.part(surface, worktop_finish::label(o, surface));
    let outer = corners(outer);
    let inner = corners(aperture);
    let pos = origin(o);
    let at = |p: Vec2, h| Vec3::new(p.x, h, p.y);
    let bottom = top - RUN_THICKNESS;
    let mut quad = |points, normal| {
        let start = g.positions.len();
        g.metric_quad(points, normal);
        chart(g, start..g.positions.len(), pos, top, normal);
    };
    // Four trapezoids meet edge-to-edge on each broad face. The only vertical
    // faces belong to the slab perimeter and the physical basin reveal.
    for (i, n) in [Vec3::NEG_Z, Vec3::X, Vec3::Z, Vec3::NEG_X]
        .into_iter()
        .enumerate()
    {
        let next = (i + 1) % 4;
        for (h, normal) in [(top, Vec3::Y), (bottom, Vec3::NEG_Y)] {
            quad(
                [
                    at(outer[i], h),
                    at(outer[next], h),
                    at(inner[next], h),
                    at(inner[i], h),
                ],
                normal,
            );
        }
        for (p, normal) in [(&outer, n), (&inner, -n)] {
            quad(
                [
                    at(p[i], top),
                    at(p[next], top),
                    at(p[next], bottom),
                    at(p[i], bottom),
                ],
                normal,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{audit, build_object, part_label};
    use super::*;

    fn check_case_joints(kind: ObjectKind, island: bool) {
        const EPS: f32 = 1e-5;
        for seed in 0..32 {
            let mut o = audit::fixture(kind, seed);
            o.size = dimensions(kind, seed);
            if island {
                configure_island(&mut o);
            }
            for height in [0.86, 0.90, 0.94] {
                align_worktop(&mut o, height);
                let mesh = build_object(&o);
                let slab = mesh
                    .parts
                    .iter()
                    .find(|((_, key), _)| part_label(key) == "counter")
                    .unwrap()
                    .1;
                let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
                let mut underside = f32::NEG_INFINITY;
                for (&v, n) in slab.positions.iter().zip(&slab.normals) {
                    let v = Vec3::from(v);
                    lo = lo.min(v);
                    hi = hi.max(v);
                    // Waterfall returns extend to the floor. Use the slab's own lower face.
                    if n[1] < -0.99999 {
                        underside = underside.max(v.y);
                    }
                }
                assert!((hi.y - height).abs() < EPS && (0.02..0.07).contains(&(hi.y - underside)));
                let mut supported_ends = [false; 2];
                let mut backsplash_seated = false;
                for ((_, label), g) in &mesh.parts {
                    if part_label(label) != "cabinet" {
                        continue;
                    }
                    for (&v, n) in g.positions.iter().zip(&g.normals) {
                        if (v[1] - underside).abs() < EPS
                            && v[0].abs() > o.size.x * 0.35
                            && n[1] > 0.99
                        {
                            supported_ends[usize::from(v[0] > 0.)] = true;
                        }
                        if (v[1] - hi.y).abs() < EPS && v[2] < lo.z + 0.025 && n[1] < -0.99 {
                            backsplash_seated = true;
                        }
                    }
                    for tri in g.indices.as_chunks::<3>().0 {
                        let [a, b, c] = tri.map(|i| Vec3::from(g.positions[i as usize]));
                        let min = a.min(b).min(c);
                        let max = a.max(b).max(c);
                        if max.x < lo.x || min.x > hi.x || max.z < lo.z || min.z > hi.z {
                            continue;
                        }
                        assert!(
                            max.y <= underside + EPS
                                || (kind == ObjectKind::KitchenCupboard && min.y >= hi.y - EPS),
                            "{kind:?} seed {seed}: cabinet face {min:?}..{max:?} crosses slab {underside}..{}",
                            hi.y
                        );
                    }
                }
                assert_eq!(
                    supported_ends, [true; 2],
                    "{kind:?} seed {seed}: unsupported slab"
                );
                if kind == ObjectKind::KitchenCupboard {
                    assert!(
                        backsplash_seated,
                        "seed {seed}: backsplash must rest on the slab"
                    );
                }
            }
        }
    }

    #[test]
    fn sink_case_joints_support_the_pierced_slab() {
        check_case_joints(ObjectKind::KitchenSink, false);
    }

    #[test]
    fn cupboard_case_joints_keep_the_backsplash_above_the_slab() {
        check_case_joints(ObjectKind::KitchenCupboard, false);
    }

    #[test]
    fn counter_case_joints_support_solid_slabs_and_island_returns() {
        check_case_joints(ObjectKind::KitchenCounter, false);
        check_case_joints(ObjectKind::KitchenCounter, true);
    }

    #[test]
    fn sink_surround_is_manifold_and_chart_matches_installed_neighbor() {
        use std::collections::BTreeMap;
        let mut o = super::super::super::audit::fixture(ObjectKind::KitchenSink, 74);
        o.position = Vec3::new(2.37, 0.22, -3.71);
        o.yaw = 0.77;
        let mut a = Assembly::default();
        let outer = (Vec2::new(-0.62, -0.34), Vec2::new(0.62, 0.34));
        let hole = (Vec2::new(-0.35, -0.18), Vec2::new(0.22, 0.26));
        pierced(&mut a, &o, Surface::Wood, outer, hole, 0.92);
        let g = a.parts.values().next().unwrap();
        assert_eq!(g.indices.len() / 3, 32);
        let mut edges = BTreeMap::new();
        let mut area = 0.;
        for tri in g.indices.as_chunks::<3>().0 {
            let p = [tri[0], tri[1], tri[2]].map(|i| Vec3::from_array(g.positions[i as usize]));
            let cross = (p[1] - p[0]).cross(p[2] - p[0]);
            let normal = Vec3::from_array(g.normals[tri[0] as usize]);
            assert!(cross.dot(normal) > 0.);
            if normal.y > 0.99 {
                area += cross.length() * 0.5;
                for (&i, point) in tri.iter().zip(p) {
                    assert!(
                        (Vec2::from_array(g.uvs[i as usize]) - (origin(&o) + point.xz())).length()
                            < 1e-5
                    );
                }
            }
            for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                let mut k = [
                    p[a].to_array().map(f32::to_bits),
                    p[b].to_array().map(f32::to_bits),
                ];
                k.sort();
                *edges.entry(k).or_insert(0) += 1;
            }
        }
        assert!(edges.values().all(|&v| v == 2));
        assert!(
            (area - ((outer.1 - outer.0).element_product() - (hole.1 - hole.0).element_product()))
                .abs()
                < 1e-5
        );
        let mut adjacent = o.clone();
        adjacent.position += Quat::from_rotation_y(o.yaw) * Vec3::X * 1.24;
        let mut b = Assembly::default();
        solid(
            &mut b,
            &adjacent,
            Surface::Wood,
            Vec3::Y * (0.92 - 0.017),
            Vec3::new(1.24, 0.034, 0.68),
            0.,
            0.92,
        );
        let right = b.parts.values().next().unwrap();
        let uv = |geometry: &Geometry, point: Vec3| {
            geometry
                .positions
                .iter()
                .enumerate()
                .find_map(|(i, &p)| {
                    (geometry.normals[i][1] > 0.99 && Vec3::from_array(p).distance(point) < 1e-5)
                        .then(|| Vec2::from_array(geometry.uvs[i]))
                })
                .unwrap()
        };
        for z in [-0.34, 0.34] {
            assert!(
                uv(g, Vec3::new(0.62, 0.92, z)).distance(uv(right, Vec3::new(-0.62, 0.92, z)))
                    < 1e-5
            );
        }
    }
}
