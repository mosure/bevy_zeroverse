use super::*;
#[test]
fn long_conference_frames_remain_bounded_grounded_and_in_the_clearance_oracle() {
    let mut styles = std::collections::BTreeSet::new();
    let mut counts = std::collections::BTreeSet::new();
    for seed in 0..96 {
        for size in [
            Vec3::new(1.1, 0.72, 3.1),
            Vec3::new(1.6, 0.78, 5.8),
            Vec3::new(9.2, 0.75, 2.4),
        ] {
            let mut o = object(ObjectKind::Table, seed);
            o.size = size;
            let p = parameters(&o);
            styles.insert(p.bay_style);
            counts.insert(support_bay_count(&o));
            let mesh = super::super::build_object(&o);
            let (lo, hi) = mesh.bounds();
            assert!(
                lo.cmpge(Vec3::new(-size.x * 0.5, -0.001, -size.z * 0.5) - Vec3::splat(1e-6))
                    .all(),
                "{seed}/{size}: {lo}"
            );
            assert!(
                hi.cmple(Vec3::new(size.x * 0.5, size.y + 0.004, size.z * 0.5) + Vec3::splat(1e-6))
                    .all(),
                "{seed}/{size}: {hi}"
            );
            assert!(lo.y.abs() < 1e-5, "grounded {seed}");
            let solid = Clearance::new(&o);
            super::super::audit::check(&o, &mesh).unwrap_or_else(|e| panic!("{seed}/{size}: {e}"));
            for part in mesh.parts.values() {
                for p in &part.positions {
                    let p = Vec3::from_array(*p);
                    assert!(
                        solid.hits_capsule(p, p, 0.0001),
                        "missing structure {seed}: {p}"
                    );
                }
            }
        }
    }
    assert_eq!(styles.len(), 6);
    assert!(counts.len() >= 4);
}

fn object(kind: ObjectKind, seed: u64) -> IndoorObject {
    IndoorObject {
        finish_seed: None,
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
        cable_path: None,
        cable_target: None,
    }
}
#[test]
fn structural_clearance_encloses_all_rendered_support_families() {
    let mut families = std::collections::BTreeSet::new();
    for seed in 0..96 {
        for kind in [
            ObjectKind::Desk,
            ObjectKind::Table,
            ObjectKind::CoffeeTable,
            ObjectKind::SideTable,
        ] {
            let mut o = object(kind, seed);
            if kind == ObjectKind::CoffeeTable {
                o.size = Vec3::new(1.5, 0.4, 0.75);
            }
            families.insert(parameters(&o).support);
            let c = Clearance::new(&o);
            let mesh = super::super::build_object(&o);
            let top = outline(&o);
            for p in mesh.parts.values().flat_map(|g| &g.positions) {
                let p = Vec3::from_array(*p);
                if p.y > o.size.y + 0.0002 {
                    assert!(
                        supports_outline(&top, p.xz(), p.xz(), 0.),
                        "unsupported grommet {seed}: {p}"
                    );
                }
                assert!(
                    c.hits_capsule(p, p, 0.00002),
                    "{kind:?} seed={seed} vertex={p}"
                );
            }
        }
    }
    assert_eq!(families.len(), 10);
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

#[test]
fn support_programs_remain_grounded_bounded_and_non_degenerate_at_small_sizes() {
    let mut scene = super::super::super::layout::IndoorManifest::generate_with_humans(
        17,
        super::super::super::layout::IndoorLayout::Mixed,
        0.5,
        0,
        0.,
    )
    .unwrap();
    scene.objects.clear();
    for seed in 0..96 {
        for (kind, size) in [
            (ObjectKind::SideTable, Vec3::new(0.45, 0.52, 0.45)),
            (ObjectKind::CoffeeTable, Vec3::new(0.80, 0.36, 0.60)),
            (ObjectKind::Table, Vec3::new(1.1, 0.74, 3.4)),
            (ObjectKind::Desk, Vec3::new(1.4, 0.74, 0.70)),
        ] {
            let mut o = object(kind, seed);
            o.size = size;
            o.id = scene.objects.len();
            let assembly = super::super::build_object(&o);
            let (lo, hi) = assembly.bounds();
            assert!(
                lo.cmpge(Vec3::new(-size.x * 0.5, -0.003, -size.z * 0.5) - Vec3::splat(1e-5))
                    .all(),
                "{kind:?}/{seed}: {lo:?}"
            );
            assert!(
                hi.cmple(Vec3::new(size.x * 0.5, size.y + 0.004, size.z * 0.5) + Vec3::splat(1e-5))
                    .all(),
                "{kind:?}/{seed}: {hi:?}"
            );
            assert!(lo.y <= 0.002, "grounded {kind:?}/{seed}: {lo:?}");
            scene.objects.push(o);
        }
    }
    super::super::super::validation::validate_geometry(&scene).unwrap();
}

#[test]
fn trestle_stretchers_physically_connect_to_end_frames() {
    #[derive(Default)]
    struct Rods(Vec<(Vec3, Vec3, f32)>);
    impl StructureSink for Rods {
        type Part = Self;
        fn part(&mut self, _: Surface, _: &str) -> &mut Self {
            self
        }
    }
    impl PartSink for Rods {
        fn cuboid(&mut self, _: Vec3, _: f32, _: Transform) {}
        fn cylinder(&mut self, _: f32, _: f32, _: Transform) {}
        fn lathe(&mut self, _: &[(f32, f32)], _: u32, _: Transform) {}
        fn rod(&mut self, a: Vec3, b: Vec3, r: f32) {
            self.0.push((a, b, r));
        }
        fn tube(&mut self, p: &[Vec3], r: f32, _: u32) {
            for points in p.windows(2) {
                self.rod(points[0], points[1], r);
            }
        }
    }
    for seed in 0..128 {
        let o = object(ObjectKind::Table, seed);
        let mut p = parameters(&o);
        p.support = 6;
        let mut rods = Rods::default();
        build_structure(&mut rods, &o, &p, &outline(&o));
        let rails: Vec<_> = rods
            .0
            .iter()
            .enumerate()
            .filter(|(_, (a, b, _))| (a.y - b.y).abs() < 1e-6 && a.y < o.size.y * 0.60)
            .collect();
        assert_eq!(rails.len(), 3);
        for (index, &(a, b, radius)) in rails {
            for endpoint in [a, b] {
                assert!(
                    rods.0.iter().enumerate().any(|(other, &(c, d, r))| {
                        if other == index {
                            return false;
                        }
                        let delta = d - c;
                        let closest = c + delta
                            * ((endpoint - c).dot(delta) / delta.length_squared()).clamp(0., 1.);
                        endpoint.distance(closest) <= radius + r + 1e-5
                    }),
                    "disconnected trestle rail seed={seed}, endpoint={endpoint}"
                );
            }
        }
    }
}

#[test]
fn rounded_polygon_top_blends_keep_convex_supported_sampling_domains() {
    use super::super::super::envelope::polygon;
    let mut sides = std::collections::BTreeSet::new();
    for seed in 0..1024 {
        for size in [Vec3::new(0.45, 0.52, 0.45), Vec3::new(1.1, 0.74, 3.4)] {
            let mut o = object(ObjectKind::Table, seed);
            o.size = size;
            let p = parameters(&o);
            if p.polygon_blend > 0. {
                sides.insert(p.polygon_sides);
            }
            let top = outline(&o);
            assert!(polygon::contains(&top, Vec2::ZERO, 0.), "fan origin {seed}");
            for i in 0..top.len() {
                let edge = top[(i + 1) % top.len()] - top[i];
                let next = top[(i + 2) % top.len()] - top[(i + 1) % top.len()];
                assert!(edge.perp_dot(next) >= -1e-6, "concave {seed}/{size}");
                assert!(edge.length() > 1e-5, "degenerate boundary {seed}");
                assert!((top[i].abs() - size.xz() * 0.5).max_element() <= 1e-5);
            }
            // Corners outside the true outline must never be accepted as a
            // support merely because they lie inside the table's AABB.
            for x in [-0.49, 0., 0.49] {
                for z in [-0.49, 0., 0.49] {
                    let point = Vec2::new(size.x * x, size.z * z);
                    if supports(&o, point, point, 0.) {
                        assert!(polygon::contains(&top, point, 0.));
                    }
                }
            }
        }
    }
    assert_eq!(sides.len(), 6);
}
