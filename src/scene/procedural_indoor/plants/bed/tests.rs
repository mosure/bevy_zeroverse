use super::*;
use crate::scene::procedural_indoor::objects::{self, audit};

pub(super) fn fixture(seed: u64) -> IndoorObject {
    let mut o = audit::fixture(ObjectKind::PlantingBed, seed);
    let mut r = stream(seed, 7516);
    o.size = Vec3::new(
        r.random_range(2.15..3.8),
        r.random_range(1.30..2.95),
        r.random_range(1.08..1.85),
    );
    o.variant = (seed % u64::from(SPECIES)) as u32;
    o
}

// Independent vertical intersection with the triangles that will be uploaded.
fn heights(mesh: &Geometry, p: Vec2) -> impl Iterator<Item = f32> + '_ {
    mesh.indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(move |tri| {
            let [a, b, c] = tri.map(|i| Vec3::from_array(mesh.positions[i as usize]));
            let (u, v, q) = (b.xz() - a.xz(), c.xz() - a.xz(), p - a.xz());
            let area = u.perp_dot(v);
            if area.abs() < 1e-9 {
                return None;
            }
            let (x, y) = (q.perp_dot(v) / area, u.perp_dot(q) / area);
            (x >= -1e-5 && y >= -1e-5 && x + y <= 1.00001)
                .then_some(a.y + x * (b.y - a.y) + y * (c.y - a.y))
        })
}

#[test]
fn courtyard_roots_meet_the_rendered_grade_and_coping_has_one_visible_face() {
    for seed in 0..96 {
        let o = fixture(seed);
        let garden = Garden::sample(&o);
        let mut soil = Assembly::default();
        garden.ground.mesh(&mut soil);
        let surface = soil.parts.values().next().unwrap();
        let mut edge = Assembly::default();
        garden
            .edging
            .mesh(&mut edge, o.size.xz() * 0.5, &garden.ground);
        audit::check(&o, &edge).unwrap();
        for root in &garden.crowns {
            let ys: Vec<_> = heights(surface, root.root_m.xz()).collect();
            assert!(
                !ys.is_empty() && ys.iter().all(|y| (y - root.root_m.y).abs() < 2e-6),
                "{seed}: root {ys:?}"
            );
            assert!(root.root_m.y < garden.edging.height_m && root.width_m > 0.);
            let mut child = o.clone();
            child.kind = ObjectKind::Plant;
            child.seed = root.seed;
            child.variant = root.species;
            child.size = Vec3::new(root.width_m, root.height_m, root.width_m);
            let mut plant = Assembly::default();
            build_plant(&mut plant, &child, VERSION, Roots::Bed);
            // Rooted tissue crosses grade even for an off-center stem base.
            let yaw = Quat::from_rotation_y(root.yaw_radians);
            let crosses = plant.parts.values().any(|g| {
                g.indices.as_chunks::<3>().0.iter().any(|tri| {
                    let diff = tri.map(|j| {
                        let x = root.root_m + yaw * Vec3::from_array(g.positions[j as usize]);
                        x.y - garden.ground.at(x.xz()).height_m
                    });
                    diff.iter().any(|d| *d <= 0.003) && diff.iter().any(|d| *d >= -0.003)
                })
            });
            assert!(crosses, "floating crown {} in {seed}", root.species);
        }
        let ratio = o.size.xz() * 0.5 / garden.ground.radii_m;
        for i in 0..SIDES {
            let inner =
                garden.ground.boundary[i].lerp(garden.ground.boundary[(i + 1) % SIDES], 0.38);
            let q = inner.lerp(inner * ratio, 0.46);
            let visible = edge
                .parts
                .values()
                .flat_map(|g| heights(g, q))
                .filter(|y| (*y - garden.edging.height_m).abs() < 1e-5)
                .count();
            assert_eq!(visible, 1, "{seed}/{i}: overlapping or missing coping");
            assert!(heights(surface, q).next().is_none());
        }
    }
}

#[test]
fn courtyard_plantings_span_both_axes_and_sample_compatible_companions() {
    let mut counts = [0usize; 4];
    let mut quadrants = [0usize; 4];
    let mut front_back = 0;
    let mut centered = 0;
    for seed in 0..512 {
        let o = fixture(seed);
        let garden = Garden::sample(&o);
        counts[garden.crowns.len()] += 1;
        assert_eq!(garden.crowns[0].species, o.variant);
        for c in &garden.crowns {
            let aspect = c.height_m / c.width_m;
            match c.species {
                4 => assert!((0.61..=1.11).contains(&aspect), "fern {seed}"),
                7 => assert!((0.44..=0.86).contains(&aspect), "rosette {seed}"),
                _ => {}
            }
        }
        let a = garden.crowns[0].root_m.xz() / garden.ground.radii_m;
        quadrants[usize::from(a.x > 0.) * 2 + usize::from(a.y > 0.)] += 1;
        front_back += usize::from(
            garden
                .crowns
                .iter()
                .any(|c| (c.root_m.z / garden.ground.radii_m.y).abs() > 0.35),
        );
        centered += usize::from(a.length() < 0.19);
        for (i, c) in garden.crowns.iter().enumerate().skip(1) {
            assert!(garden.collection.companions().contains(&c.species));
            for other in &garden.crowns[..i] {
                assert!(
                    c.root_m.xz().distance(other.root_m.xz())
                        > garden.ground.radii_m.min_element() * 0.36,
                    "{seed}"
                );
            }
        }
    }
    assert!(counts[1..].iter().all(|&n| n >= 45));
    assert!(quadrants.iter().all(|&n| n >= 80));
    assert!(centered > 75 && front_back > 180);
}

#[test]
fn courtyard_installed_beds_stay_inside_the_reserved_box_and_keep_plant_semantics() {
    for seed in 0..96 {
        let o = fixture(seed);
        let a = objects::build_object(&o);
        audit::check(&o, &a).unwrap();
        let (lo, hi) = a.bounds();
        assert!(
            lo.y.abs() < 1e-5 && hi.y <= o.size.y + 0.002,
            "{seed}: {lo} {hi}"
        );
        assert!(a
            .parts
            .keys()
            .all(|(_, label)| objects::part_label(label) == "plant"));
    }
}

#[test]
fn archived_courtyard_beds_retain_the_straight_sided_flat_soil_construction() {
    let o = fixture(18);
    let old = objects::build_object_at_version(&o, None, VERSION - 1);
    let new = objects::build_object_at_version(&o, None, VERSION);
    for (a, flat) in [(&old, true), (&new, false)] {
        let soil = a
            .parts
            .iter()
            .find(|((s, _), _)| *s == Surface::Soil)
            .unwrap()
            .1;
        let y = soil.positions[0][1];
        assert_eq!(soil.positions.iter().all(|p| (p[1] - y).abs() < 1e-6), flat);
        audit::check(&o, a).unwrap();
    }
}

#[test]
fn a_shared_specimen_replays_its_installed_collection_and_preserves_legacy_species_numbers() {
    for group in [
        Collection::Woodland,
        Collection::Tropical,
        Collection::DryGarden,
    ] {
        let mut o = fixture(19);
        assert!(group.specimens().contains(&13));
        assert_eq!(group.bed_variant(13, VERSION - 1), 13);
        o.variant = group.bed_variant(13, VERSION);
        let replay: IndoorObject =
            serde_json::from_slice(&serde_json::to_vec(&o).unwrap()).unwrap();
        let plan = Garden::sample(&replay);
        assert_eq!(plan.collection, group);
        assert_eq!(plan.crowns[0].species, 13);
        assert!(plan
            .crowns
            .iter()
            .skip(1)
            .all(|c| group.companions().contains(&c.species)));
    }
}
