use super::*;

#[test]
fn banquette_back_height_does_not_change_the_adult_seating_datum() {
    for seed in 0..64 {
        let mut o = super::super::audit::fixture(ObjectKind::Banquette, seed);
        let mut seats = Vec::new();
        for height in [0.88, 1.25] {
            o.size.y = height;
            let a = super::super::build_object(&o);
            super::super::audit::check(&o, &a).unwrap();
            let seat = a
                .parts
                .iter()
                .filter(|((surface, _), _)| *surface == Surface::Fabric)
                .flat_map(|(_, g)| &g.positions)
                .filter(|p| p[2] > 0.)
                .map(|p| p[1])
                .fold(f32::NEG_INFINITY, f32::max);
            assert!((0.435..0.485).contains(&seat), "seed {seed}: {seat}");
            seats.push(seat);
        }
        assert_eq!(seats[0], seats[1]);
    }
}

#[test]
fn display_contents_remain_supported_at_small_physical_scales() {
    for kind in [
        ObjectKind::Sculpture,
        ObjectKind::Vase,
        ObjectKind::Bread,
        ObjectKind::Dishware,
    ] {
        for seed in 0..128 {
            let mut o = super::super::audit::fixture(kind, seed);
            o.size = hospitality::content_size(kind, seed, &mut stream(seed, 8210));
            let a = super::super::build_object(&o);
            super::super::audit::check(&o, &a).unwrap_or_else(|e| panic!("{e}"));
            // Independently supported content must not pierce its shelf, even
            // when a whole display's looser outer envelope would hide it.
            assert!(a.bounds().0.y >= -1e-5, "{kind:?}/{seed}: {:?}", a.bounds());
        }
    }
}

#[test]
fn public_items_fit_admitted_bounds_with_valid_semantic_parts() {
    for (kind, variant) in ObjectKind::ALL
        .into_iter()
        .filter(|k| k.is_program_item())
        .map(|k| (k, None))
        .chain([(ObjectKind::Chair, Some(10)), (ObjectKind::Chair, Some(11))])
    {
        for seed in 0..32 {
            let mut o = super::super::audit::fixture(kind, seed);
            if let Some(v) = variant {
                o.variant = v;
                o.size = Vec3::new(0.72, 1.04, 0.72);
            }
            let a = super::super::build_object(&o);
            let (lo, hi) = a.bounds();
            let half = o.size * 0.5;
            assert!(lo.is_finite() && hi.is_finite(), "{kind:?} {seed}");
            assert!(
                lo.x >= -half.x - 0.003
                    && lo.z >= -half.z - 0.003
                    && lo.y >= -0.003
                    && hi.x <= half.x + 0.003
                    && hi.z <= half.z + 0.003
                    && hi.y <= o.size.y + 0.003,
                "{kind:?} {seed}: {lo:?} {hi:?}, {:?}",
                o.size
            );
            for ((_, label), g) in a.parts {
                assert!(
                    SemanticLabel::from_label(super::super::part_label(&label)).is_some(),
                    "{label}"
                );
                assert!(!g.indices.is_empty());
                for n in g.normals {
                    assert!(
                        (Vec3::from_array(n).length() - 1.).abs() < 1e-4,
                        "{kind:?}/{seed}/{label}: {n:?}"
                    );
                }
                for t in g.indices.as_chunks::<3>().0 {
                    let a = Vec3::from_array(g.positions[t[0] as usize]);
                    let b = Vec3::from_array(g.positions[t[1] as usize]);
                    let c = Vec3::from_array(g.positions[t[2] as usize]);
                    assert!(
                        (b - a).cross(c - a).length_squared() > 1e-18,
                        "{kind:?} {seed} {label}: degenerate triangle"
                    );
                }
            }
        }
    }
}
