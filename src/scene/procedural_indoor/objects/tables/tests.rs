use super::*;
fn object(kind: ObjectKind, seed: u64) -> IndoorObject {
    IndoorObject {
        id: 0,
        kind,
        position: Vec3::ZERO,
        size: Vec3::new(1.4, 0.74, 2.4),
        yaw: 0.,
        variant: 0,
        seed,
        solid: true,
        support: None,
        neighbor: false,
        interaction_target: None,
    }
}
#[test]
fn structural_clearance_encloses_all_rendered_support_families() {
    let mut families = std::collections::BTreeSet::new();
    for seed in 0..96 {
        for kind in [ObjectKind::Desk, ObjectKind::Table, ObjectKind::CoffeeTable] {
            let mut o = object(kind, seed);
            if kind == ObjectKind::CoffeeTable {
                o.size = Vec3::new(1.5, 0.4, 0.75);
            }
            families.insert(parameters(&o).support);
            let c = Clearance::new(&o);
            let mesh = super::super::build_object(&o);
            for p in mesh.parts.values().flat_map(|g| &g.positions) {
                let p = Vec3::from_array(*p);
                assert!(
                    c.hits_capsule(p, p, 0.00002),
                    "{kind:?} seed={seed} vertex={p}"
                );
            }
        }
    }
    assert_eq!(families.len(), 6);
}
#[test]
fn chair_clearance_bounds_rendered_shell_arms_and_bases() {
    for family in 0..super::super::chairs::FAMILIES {
        for seed in 0..64 {
            let mut o = object(ObjectKind::Chair, seed);
            o.variant = family;
            o.size = Vec3::new(
                0.68,
                if family == 6 {
                    0.48
                } else if family == 7 {
                    0.79
                } else {
                    1.35
                },
                0.68,
            );
            let boxes = super::super::chairs::clearance_boxes(&o);
            let mesh = super::super::build_object(&o);
            for p in mesh.parts.values().flat_map(|g| &g.positions) {
                let p = Vec3::from_array(*p);
                assert!(
                    boxes
                        .iter()
                        .any(|&(a, b)| p.cmpge(a - Vec3::splat(0.003)).all()
                            && p.cmple(b + Vec3::splat(0.003)).all()),
                    "family={family} seed={seed} vertex={p} boxes={boxes:?}"
                );
            }
        }
    }
}
#[test]
fn top_leg_and_pedestal_collisions_are_not_waived_for_tucked_chairs() {
    use super::super::super::layout::furniture_overlap;
    let mut table = object(ObjectKind::Table, 0);
    table.seed = (0..100)
        .find(|&seed| {
            table.seed = seed;
            parameters(&table).support == 2
        })
        .unwrap();
    let mut chair = object(ObjectKind::Chair, 0);
    chair.id = 1;
    chair.size = Vec3::new(0.68, 0.48, 0.68);
    chair.variant = 6;
    chair.interaction_target = Some(0);
    chair.position = Vec3::new(0., 0., table.size.z * 0.5 + 0.10);
    assert!(
        !furniture_overlap(&table, &chair, 0.008),
        "backless seat should fit under the overhang"
    );
    chair.position.y = table.size.y - 0.47;
    assert!(
        furniture_overlap(&table, &chair, 0.008),
        "raised seat intersects slab"
    );
    chair.position = Vec3::ZERO;
    assert!(
        furniture_overlap(&table, &chair, 0.008),
        "base/pedestal must block the chair"
    );
    let c = Clearance::new(&table);
    assert!(c.hits_capsule(Vec3::new(0., 0.3, 0.), Vec3::new(0., 0.5, 0.), 0.07));
    assert!(c.hits_capsule(Vec3::new(0., 0.8, 1.), Vec3::new(0., 0.6, 1.), 0.04));
}
