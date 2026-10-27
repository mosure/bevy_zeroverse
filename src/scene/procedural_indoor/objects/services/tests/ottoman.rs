use super::*;
use std::collections::BTreeSet;

#[test]
fn ottoman_extremes_remain_grounded_bounded_labelled_and_in_budget() {
    let mut bases = BTreeSet::new();
    let mut cap_counts = [0; 2];
    for seed in 0..96 {
        for size in [
            Vec3::new(0.40, 0.30, 0.29),
            Vec3::new(0.65, 0.43, 0.62),
            Vec3::new(0.97, 0.50, 1.08),
        ] {
            let mut o = audit::fixture(ObjectKind::Ottoman, seed);
            o.size = size;
            let p = super::super::ottoman::parameters(&o);
            let a = build_object(&o);
            let check = audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed}/{size:?}: {e}"));
            assert!(check.triangles < 2400);
            assert!(a.parts.len() <= 3);
            let (lo, hi) = a.bounds();
            assert!(lo.y.abs() < 1e-5 && (hi.y - size.y).abs() < 1e-5);
            assert!(a
                .parts
                .keys()
                .all(|(_, label)| part_label(label) == o.kind.class_name()));
            for ((surface, _), g) in &a.parts {
                if matches!(
                    surface,
                    Surface::Fabric | Surface::FabricAlt | Surface::Leather
                ) {
                    let mut welded = std::collections::BTreeMap::new();
                    for (v, n) in g.positions.iter().zip(&g.normals) {
                        if let Some(previous) = welded.insert(v.map(f32::to_bits), *n) {
                            assert!(
                                Vec3::from_array(previous).distance(Vec3::from_array(*n)) < 1e-4,
                                "hard cloth ring/seam: {seed}/{size:?}"
                            );
                        }
                    }
                }
            }
            if p.cap_fraction > 0. {
                let bottom = (p.base_height - p.rail * 0.8 * 0.35).max(0.);
                let contact_y = size.y - (size.y - bottom) * p.cap_fraction;
                let contact_vertices = a
                    .parts
                    .values()
                    .flat_map(|g| g.positions.iter().zip(&g.normals))
                    .filter(|(v, n)| (v[1] - contact_y).abs() < 1e-5 && n[1] > 0.999)
                    .count();
                assert!(
                    contact_vertices >= 32,
                    "no broad cushion contact surface: {seed}/{size:?}"
                );
            }
            bases.insert(p.base);
            cap_counts[usize::from(p.cap_fraction > 0.)] += 1;
        }
    }
    assert_eq!(bases.len(), 4);
    assert!(cap_counts.into_iter().all(|n| n > 60));
}

#[test]
fn ottoman_finish_interventions_preserve_geometry_normals_and_metric_uvs() {
    let geometry = |a: &Assembly, metric_uvs: bool| {
        let mut vertices: Vec<_> = a
            .parts
            .values()
            .flat_map(|g| {
                g.indices.iter().map(|&i| {
                    let i = i as usize;
                    (
                        g.positions[i].map(f32::to_bits),
                        g.normals[i].map(f32::to_bits),
                        if metric_uvs {
                            g.uvs[i].map(f32::to_bits)
                        } else {
                            [0; 2]
                        },
                    )
                })
            })
            .collect();
        vertices.sort_unstable();
        vertices
    };
    let mut finishes = BTreeSet::new();
    for seed in 0..12 {
        let mut o = audit::fixture(ObjectKind::Ottoman, seed);
        let raw = |o: &IndoorObject| {
            let mut a = Assembly::default();
            super::super::ottoman::build(&mut a, o);
            a
        };
        let reference = geometry(&build_object(&o), false);
        let metric_reference = geometry(&raw(&o), true);
        for finish in 0..64 {
            o.finish_seed = Some(finish);
            let a = build_object(&o);
            assert!(
                reference == geometry(&a, false),
                "positions/normals changed: {seed}/{finish}"
            );
            // Material assignment adds substrate-specific UV phase offsets in
            // the common object builder. The constructor preserves metric UVs.
            assert!(
                metric_reference == geometry(&raw(&o), true),
                "metric UVs changed: {seed}/{finish}"
            );
            let p = super::super::ottoman::parameters(&o);
            finishes.insert((p.body, p.top));
            let replay = build_object(&o);
            for (key, g) in &a.parts {
                assert_eq!(g.positions, replay.parts[key].positions);
                assert_eq!(g.indices, replay.parts[key].indices);
            }
        }
    }
    assert!(finishes.len() >= 5, "accepted finish pairs: {finishes:?}");
}
