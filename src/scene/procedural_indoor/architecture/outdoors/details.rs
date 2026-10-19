//! Opaque near-field facade relief. Glazing stays backed by real building mass.
use super::*;

/// Columns carry two cross-beams, which in turn carry every roof slat.
pub(super) fn pergola(a: &mut Assembly, p: &Parcel) {
    for x in [-0.43, 0.43] {
        for z in [-0.43, 0.43] {
            box_at(
                a,
                p,
                Surface::WoodEdge,
                "other_structure",
                Vec3::new(x * p.size.x, 1.30, z * p.size.y),
                Vec3::new(0.095, 2.54, 0.095),
                0.005,
            );
        }
    }
    for z in [-0.43, 0.43] {
        box_at(
            a,
            p,
            Surface::WoodEdge,
            "other_structure",
            Vec3::new(0., 2.57, z * p.size.y),
            Vec3::new(p.size.x * 0.96, 0.12, 0.095),
            0.003,
        );
    }
    for i in 0..7 {
        box_at(
            a,
            p,
            Surface::WoodEdge,
            "other_structure",
            Vec3::new((i as f32 / 6. - 0.5) * p.size.x * 0.9, 2.63, 0.),
            Vec3::new(0.07, 0.12, p.size.y * 0.94),
            0.003,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pergola_roof_has_continuous_cross_support_and_rotated_planting_clearance() {
        let scene = IndoorManifest::generate_with_humans(
            6,
            super::super::super::super::layout::IndoorLayout::Mixed,
            0.65,
            0,
            0.,
        )
        .unwrap();
        for seed in 0..48 {
            let p = Parcel {
                center: Vec2::new(14., -11.),
                size: Vec2::new(2.2 + seed as f32 * 0.05, 2.0 + seed as f32 * 0.04),
                yaw: seed as f32 * 0.23,
                kind: 3,
                seed,
            };
            let inverse = Transform::from_translation(Vec3::new(p.center.x, 0., p.center.y))
                .with_rotation(Quat::from_rotation_y(p.yaw))
                .compute_affine()
                .inverse();
            let mut plants = Assembly::default();
            garden(&mut plants, &scene, &p);
            for v in plants.parts.values().flat_map(|g| &g.positions) {
                let v = inverse.transform_point3(Vec3::from_array(*v));
                assert!(
                    v.x.abs() < p.size.x * 0.45 && v.z.abs() < p.size.y * 0.22,
                    "plant escaped rotated bed: {v}"
                );
                assert!(v.y < 2.50, "plant intersects cross-beam or slats");
            }
            let mut frame = Assembly::default();
            pergola(&mut frame, &p);
            for i in 0..25 {
                let point = Vec2::new((i as f32 / 24. - 0.5) * p.size.x * 0.90, p.size.y * 0.43);
                let mut supported = false;
                for g in frame.parts.values() {
                    for tri in g.indices.as_chunks::<3>().0 {
                        let v: [Vec3; 3] = std::array::from_fn(|k| {
                            inverse.transform_point3(Vec3::from_array(g.positions[tri[k] as usize]))
                        });
                        let a = v[1].xz() - v[0].xz();
                        let b = v[2].xz() - v[0].xz();
                        let q = point - v[0].xz();
                        let det = a.perp_dot(b);
                        if det.abs() < 1e-8 {
                            continue;
                        }
                        let u = q.perp_dot(b) / det;
                        let w = a.perp_dot(q) / det;
                        if u >= -1e-4 && w >= -1e-4 && u + w <= 1.0001 {
                            let height = v[0].y + (v[1].y - v[0].y) * u + (v[2].y - v[0].y) * w;
                            supported |= (2.50..2.53).contains(&height);
                        }
                    }
                }
                assert!(
                    supported,
                    "no cross-beam under roof at seed {seed}, {point}"
                );
            }
        }
    }
}
