use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object_at_version};
use std::collections::{BTreeMap, BTreeSet};

fn triangles(a: &Assembly) -> impl Iterator<Item = [Vec3; 3]> + '_ {
    a.parts.values().flat_map(|g| {
        g.indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|ids| ids.map(|i| Vec3::from_array(g.positions[i as usize])))
    })
}

/// Intersect the actual mesh rather than sampling the same procedural cross section.
fn hits(a: &Assembly, origin: Vec3, direction: Vec3) -> bool {
    let (origin, direction) = (origin.as_dvec3(), direction.as_dvec3());
    triangles(a).any(|[v, b, c]| {
        let [v, b, c] = [v, b, c].map(|p| p.as_dvec3());
        let (ba, ca) = (b - v, c - v);
        let h = direction.cross(ca);
        let det = ba.dot(h);
        if det.abs() < 1e-14 {
            return false;
        }
        let s = origin - v;
        let u = s.dot(h) / det;
        let q = s.cross(ba);
        let v = direction.dot(q) / det;
        (0. ..=1.).contains(&u) && v >= 0. && u + v <= 1. && ca.dot(q) / det > 0.
    })
}

#[test]
fn pierced_sculpture_bores_are_open_and_welded_across_material_and_uv_seams() {
    let mut counts = [0; 2];
    let mut stocks = BTreeSet::new();
    let mut axes = [0; 2];
    for seed in 0..168 {
        let mut o = audit::fixture(ObjectKind::Sculpture, seed);
        o.size = super::super::dimensions(o.kind, seed);
        o.finish_seed = Some(seed.wrapping_add(73));
        let p = parameters(&o);
        let Some(c) = p.carving else { continue };
        let mut a = Assembly::default();
        c.build(&mut a, p.surface);
        audit::check(&o, &a).unwrap();
        stocks.insert((p.surface, c.bore_surface));
        let holes = c.apertures.len();
        counts[holes - 1] += 1;
        axes[usize::from(c.axis == CarvingAxis::Horizontal)] += 1;
        let mut vertices = BTreeSet::new();
        let mut edges = BTreeMap::<_, (usize, i32)>::new();
        let mut faces = 0;
        for tri in triangles(&a) {
            let points = tri.map(|q| q.to_array().map(f32::to_bits));
            vertices.extend(points);
            for (u, v) in [(0, 1), (1, 2), (2, 0)] {
                let (x, y) = (points[u], points[v]);
                let entry = edges.entry((x.min(y), x.max(y))).or_default();
                entry.0 += 1;
                entry.1 += if x < y { 1 } else { -1 };
            }
            faces += 1;
        }
        assert!(
            edges.values().all(|e| *e == (2, 0)),
            "open or inverted seam at seed {seed}"
        );
        assert_eq!(
            vertices.len() as isize - edges.len() as isize + faces,
            2 - 2 * holes as isize
        );
        let rot = Quat::from_rotation_y(c.mount_yaw_radians);
        let center = Vec2::new(0., c.foot_y_m + c.height_m * 0.5);
        let axis =
            (rot * Vec3::new(c.bore_shear * c.half_width_m / c.half_depth_m, 0., 1.)).normalize();
        for eye in &c.apertures {
            let v = (eye.bottom + eye.top) * 0.5;
            for (u, solid) in [
                (eye.center, false),
                (eye.center + 0.27 * eye.radius, false),
                (if eye.center > 0. { -0.81 } else { 0.81 }, true),
            ] {
                let (mid, _) = c.section(u, v);
                let flat = center.lerp(mid.xy(), 1. - c.face_inset).extend(mid.z);
                let at = rot * flat;
                for sign in [-1., 1.] {
                    let dir = axis * sign;
                    assert_eq!(hits(&a, at - dir * 3., dir), solid, "seed {seed}, {u}, {v}");
                }
            }
        }
        // The entire contact area fits on the plinth under the installed yaw.
        let (half_x, half_z) = (
            o.size.x * p.base_width.x * 0.5,
            o.size.z * p.base_width.y * 0.5,
        );
        let feet: Vec<_> = a
            .parts
            .values()
            .flat_map(|g| &g.positions)
            .filter(|v| v[1] <= o.size.y * p.base_height)
            .collect();
        assert!(feet.len() >= 4);
        assert!(feet
            .iter()
            .all(|v| v[0].abs() < half_x && v[2].abs() < half_z));
    }
    assert!(counts.into_iter().all(|n| n >= 15));
    assert!(stocks.len() >= 6);
    assert!(axes.into_iter().all(|n| n >= 8));
}

#[test]
fn sculpture_finish_and_archived_case_version_preserve_shape_and_pose() {
    let mut carved = 0;
    let mut changed_cases = 0;
    for seed in 0..96 {
        let mut o = audit::fixture(ObjectKind::Sculpture, seed);
        assert!(parameters_at(&o, 305).carving.is_none());
        let p = parameters(&o);
        o.finish_seed = Some(seed.wrapping_add(143));
        let q = parameters(&o);
        if let (Some(mut before), Some(after)) = (p.carving, q.carving) {
            before.bore_surface = after.bore_surface;
            assert_eq!(
                serde_json::to_value(before).unwrap(),
                serde_json::to_value(after).unwrap()
            );
            carved += 1;
        } else {
            assert_eq!(p.form, q.form);
        }
        let mut case = audit::fixture(ObjectKind::DisplayCase, seed);
        case.variant = 16;
        let old = build_object_at_version(&case, None, 305);
        let new = build_object_at_version(&case, None, 306);
        for (key, g) in &old.parts {
            if super::super::super::part_label(&key.1) != "sculpture" {
                let other = &new.parts[key];
                assert_eq!(g.positions, other.positions);
                assert_eq!(g.indices, other.indices);
            }
        }
        if new.parts.iter().any(|(key, g)| {
            super::super::super::part_label(&key.1) == "sculpture"
                && old
                    .parts
                    .get(key)
                    .is_none_or(|prior| prior.positions != g.positions)
        }) {
            changed_cases += 1;
        }
    }
    assert!(carved > 20 && changed_cases > 4);
}
