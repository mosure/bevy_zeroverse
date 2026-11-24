use super::*;

/// Height where an upward ray touches a triangle. Used against constructed
/// hardware so a detached cosmetic glide cannot pass the contact-count check.
fn ray_height(point: Vec2, [a, b, c]: [Vec3; 3]) -> Option<f32> {
    let ab = b - a;
    let ac = c - a;
    let p = point - a.xz();
    let det = ab.xz().perp_dot(ac.xz());
    if det.abs() < 1e-9 {
        return None;
    }
    let u = p.perp_dot(ac.xz()) / det;
    let v = ab.xz().perp_dot(p) / det;
    (u >= -1e-4 && v >= -1e-4 && u + v <= 1.0001).then_some(a.y + u * ab.y + v * ac.y)
}

fn check_contact(g: &Geometry, frame: &Assembly, pad_height: f32) -> usize {
    // The cap of each turned rubber glide has a downward facing centre.
    // Horizontal cap vertices on the rim have different x/z coordinates.
    let mut centers = Vec::new();
    for tri in g.indices.as_chunks::<3>().0 {
        let points = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
        if points.iter().all(|p| p.y.abs() < 1e-6) {
            // Lathes have a shared centre coordinate across all fan triangles.
            for point in points {
                centers.push(point.xz().to_array().map(f32::to_bits));
            }
        }
    }
    centers.sort_unstable();
    let mut verified = 0;
    for group in centers
        .chunk_by(|a, b| a == b)
        .filter(|group| group.len() == 12)
    {
        let point = Vec2::from_array(group[0].map(f32::from_bits));
        let touches = frame
            .parts
            .iter()
            .filter(|((stock, _), _)| *stock != Surface::Rubber)
            .any(|(_, mesh)| {
                mesh.indices.as_chunks::<3>().0.iter().any(|t| {
                    ray_height(
                        point,
                        t.map(|i| Vec3::from_array(mesh.positions[i as usize])),
                    )
                    .is_some_and(|y| y >= -1e-6 && y <= pad_height + 1e-5)
                })
            });
        assert!(touches, "detached glide at {point:?}");
        verified += 1;
    }
    verified
}
#[test]
fn frame_breaks_follow_depth_and_floor_bearings_fit_each_piece() {
    let mut joined = 0;
    let mut separate = 0;
    let mut levelers = 0;
    let mut contacts = std::collections::BTreeSet::new();
    for seed in 0..256 {
        let mut o =
            crate::scene::procedural_indoor::objects::audit::fixture(ObjectKind::Sofa, seed);
        o.size = Vec3::new(
            2.45 + (seed % 6) as f32 * 0.19,
            0.94,
            if seed % 2 == 0 { 1.76 } else { 0.96 },
        );
        let p = parameters(&o);
        let widths = module_widths_for(&o, &p);
        let f = Foundation::new(&o, &p, &widths);
        let mut cursor = -o.size.x * 0.5 + p.arm_width;
        for (i, &width) in widths.iter().enumerate() {
            let x = cursor + width * 0.5;
            let section = f
                .sections
                .iter()
                .find(|s| s.left <= x && x < s.right)
                .unwrap();
            assert!(section.left <= cursor + 1e-5 && section.right + 1e-5 >= cursor + width);
            assert!((section.back - section.front - module_depth(&o, &p, i)).abs() < 1e-5);
            cursor += width;
        }
        let mut observed_contacts = 0;
        for s in &f.sections {
            let mut a = Assembly::default();
            f.hardware(
                &mut a,
                "sofa",
                p.leg_height,
                s,
                &mut TimberCuts::new(seed, 0),
            );
            let (lo, hi) = a.bounds();
            assert!(lo.x >= s.left && hi.x <= s.right && lo.z >= s.front && hi.z <= s.back);
            assert!(lo.y.abs() < 1e-6 && hi.y >= p.leg_height && hi.y < p.seat_height - p.cushion);
            assert!(s.station_x(1) - s.station_x(0) <= 1.68 + 1e-5);
            let rubber = a.parts.get(&(Surface::Rubber, "sofa".to_owned()));
            if let Some(g) = rubber {
                // Every closed twelve-sided glide has its own centre on the
                // floor and twelve triangular faces meeting at that centre.
                let verified = check_contact(g, &a, f.glide_height);
                let ground_triangles = g
                    .indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .filter(|tri| tri.iter().all(|&i| g.positions[i as usize][1].abs() < 1e-6))
                    .count();
                assert_eq!(ground_triangles % 12, 0);
                assert_eq!(verified, ground_triangles / 12);
                observed_contacts += verified;
            } else {
                assert!(matches!(f.base, Base::Pedestal | Base::Plinth));
            }
            levelers += usize::from(s.center_levelers && s.intervals > 1);
        }
        assert_eq!(observed_contacts, f.glide_count());
        if f.glide_count() > 0 {
            contacts.insert(f.glide_count());
        }
        joined += usize::from(f.sections.len() < widths.len());
        separate += usize::from(f.sections.len() == widths.len());
        o.finish_seed = Some(seed + 987);
        let refinished = Foundation::new(&o, &p, &widths);
        assert_eq!(refinished.metadata()["sections"], f.metadata()["sections"]);
        assert_eq!(refinished.glide_count(), f.glide_count());
    }
    assert!(joined > 65 && separate > 30 && levelers > 10 && contacts.len() >= 6);
}
