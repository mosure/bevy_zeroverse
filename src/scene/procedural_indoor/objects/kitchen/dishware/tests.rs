use super::*;
use crate::scene::procedural_indoor::{
    materials::variants,
    objects::{audit, part_label},
};
use std::collections::BTreeMap;

/// Ray intervals of the emitted shell, including the foot and the rolled lip.
fn vertical_hits(a: &Assembly, at: Vec2) -> Option<(f32, f32)> {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for g in a.parts.values() {
        for tri in g.indices.as_chunks::<3>().0 {
            let [p, q, s] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
            let (e, f) = (q - p, s - p);
            let det = e.xz().perp_dot(f.xz());
            if det.abs() < 1e-12 {
                continue;
            }
            let d = at - p.xz();
            let (u, v) = (d.perp_dot(f.xz()) / det, e.xz().perp_dot(d) / det);
            if u >= -1e-5 && v >= -1e-5 && u + v <= 1.00001 {
                let y = p.y + e.y * u + f.y * v;
                lo = lo.min(y);
                hi = hi.max(y);
            }
        }
    }
    (lo <= hi).then_some((lo, hi))
}

#[test]
fn neighbouring_plates_touch_on_the_foot_and_clear_the_well_and_rim() {
    let mut staggered = 0;
    let mut squared = 0;
    for seed in 0..72 {
        let size = [Vec3::new(0.14, 0.027, 0.165), Vec3::new(0.39, 0.115, 0.24)][seed as usize % 2];
        let p = parameters(size, 0, 7, seed);
        let form = p.plate.as_ref().unwrap();
        squared += usize::from(form.outline_exponent > 2.);
        staggered += usize::from(!form.stack_offsets_m.is_empty());
        let pitch = p.interior_floor();
        assert!(p.stack_height() <= size.y + 1e-6 && pitch < p.height);
        let mut single = p.clone();
        single.count = 1;
        let mut mesh = Assembly::default();
        build(&mut mesh, Transform::IDENTITY, &single, seed);
        let offsets: Vec<_> = std::iter::once(Vec2::ZERO)
            .chain(form.stack_offsets_m.iter().copied())
            .collect();
        // Also test a centred pair on every construction.
        let (low, high) = offsets
            .windows(2)
            .max_by(|a, b| {
                a[0].distance_squared(a[1])
                    .total_cmp(&b[0].distance_squared(b[1]))
            })
            .map_or((Vec2::ZERO, Vec2::ZERO), |w| (w[0], w[1]));
        let gap = |at: Vec2| -> Option<f32> {
            Some(pitch + vertical_hits(&mesh, at)?.0 - vertical_hits(&mesh, at + high - low)?.1)
        };
        for k in (0..p.segments).step_by((p.segments as usize / 6).max(1)) {
            let (sn, cs) = (TAU * k as f32 / p.segments as f32).sin_cos();
            let unit = Vec2::new(cs, sn);
            let e = form.outline_exponent;
            let rim = unit * p.aspect / (cs.abs().powf(e) + sn.abs().powf(e)).powf(1. / e);
            assert!(
                gap(rim * form.foot_radius_m).unwrap().abs() < 3e-6,
                "lost support in {seed}"
            );
            for r in [
                p.radius * p.foot * 0.22,
                p.radius * p.shoulder,
                p.radius - form.rim_band_m,
                p.radius * 0.94,
            ] {
                if let Some(g) = gap(rim * r) {
                    assert!(
                        g > form.foot_height_m * 0.12,
                        "overlap or fused dishes in {seed}: {g}"
                    );
                }
            }
        }
        assert!((vertical_hits(&mesh, Vec2::ZERO).unwrap().0 - form.foot_height_m).abs() < 1e-6);
    }
    assert!(staggered > 30 && squared > 12);
}

#[test]
fn glaze_patches_weld_into_one_shell_and_share_the_body_texture_maps() {
    let mut colours = [0; 2];
    for seed in 0..48 {
        let mut p = parameters(Vec3::new(0.28, 0.09, 0.24), 0, 1, seed);
        p.surface = Surface::Ceramic;
        let mut a = Assembly::default();
        build(&mut a, Transform::IDENTITY, &p, seed);
        let o = audit::fixture(ObjectKind::Dishware, seed);
        let mut fit = o;
        fit.size = Vec3::new(0.28, 0.09, 0.24);
        audit::check(&fit, &a).unwrap();
        let base = variants::object_slot(p.surface, seed, None);
        let accent = PlateForm::rim_slot(p.surface, seed);
        assert_eq!(a.parts.len(), 1 + usize::from(accent.is_some()));
        colours[usize::from(accent.is_some())] += 1;
        if let Some(slot) = accent {
            assert_ne!(base, slot);
            assert_eq!(
                base % variants::structure_count(p.surface),
                slot % variants::structure_count(p.surface)
            );
        }
        let mut edges = BTreeMap::new();
        for ((s, label), g) in &a.parts {
            assert_eq!((*s, part_label(label)), (Surface::Ceramic, "dishware"));
            // The flat well uses actual metres on both axes, including ovals.
            for (pos, uv) in g.positions.iter().zip(&g.uvs) {
                if (pos[1] - p.interior_floor()).abs() < 1e-7 {
                    assert_eq!(*uv, [pos[0], pos[2]]);
                }
            }
            for triangle in g.indices.as_chunks::<3>().0 {
                let v = triangle.map(|i| {
                    g.positions[i as usize].map(|c| if c == 0. { 0 } else { c.to_bits() })
                });
                for i in 0..3 {
                    let (a, b) = (v[i], v[(i + 1) % 3]);
                    let sign = if a < b { 1i32 } else { -1 };
                    let edge = edges.entry((a.min(b), a.max(b))).or_insert((0, 0));
                    edge.0 += 1;
                    edge.1 += sign;
                }
            }
        }
        assert!(
            edges.values().all(|v| *v == (2, 0)),
            "open shell or overlapping faces in {seed}"
        );
    }
    assert!(colours.iter().all(|&n| n >= 8));
}
