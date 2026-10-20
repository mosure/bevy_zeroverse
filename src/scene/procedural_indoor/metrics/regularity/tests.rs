use super::*;
use crate::scene::procedural_indoor::{
    layout::{IndoorLayout, ObjectKind},
    objects::{self, audit},
    validation,
};

#[test]
fn regularity_counts_pairs_and_support_coordinates_without_world_axis_leakage() {
    let mut scene =
        IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    scene.objects.clear();
    let mut support = audit::fixture(ObjectKind::Desk, 1);
    support.id = 0;
    support.size = Vec3::new(2., 0.75, 1.);
    support.position = Vec3::new(2., 0.3, -4.);
    support.yaw = 0.73;
    scene.objects.push(support.clone());
    for (i, x) in [0., 0.4, -0.5].into_iter().enumerate() {
        let mut o = audit::fixture(ObjectKind::Notebook, 7);
        o.id = i + 1;
        o.support = Some(0);
        o.position = support
            .transform()
            .transform_point(Vec3::new(x, support.size.y, 0.));
        o.size = Vec3::new(if i == 2 { 0.25 } else { 0.20 }, 0.02, 0.30);
        o.yaw = [0., std::f32::consts::FRAC_PI_2, 0.3][i];
        scene.objects.push(o);
    }
    let stats = measure(&scene, &scene.objects[1..].iter().collect::<Vec<_>>());
    assert_eq!(stats.objects, 3);
    assert_eq!(stats.size_pairs, 6);
    assert_eq!(stats.equal_size_pairs_1cm, 2);
    assert_eq!(stats.axis_aligned_2deg, 2);
    assert_eq!(stats.supported, 3);
    assert_eq!(stats.support_centered_2pct, 1);
    let local = support_coordinates(&scene, &scene.objects[2]).unwrap();
    assert!((local - Vec3::new(0.2, 1., 0.)).length() < 1e-5);
    let singleton = measure(&scene, &[&scene.objects[1]]);
    assert_eq!(singleton.size_pairs, 0);
    assert_eq!(singleton.equal_size_pairs_1cm, 0);
}

#[derive(Serialize)]
struct Accepted {
    count: usize,
    min_m: Vec3,
    max_m: Vec3,
    size_cells_1cm: BTreeMap<String, usize>,
    axis_aligned_2deg: usize,
    supported: usize,
    support_centered_2pct: usize,
    screen_facing_targets: usize,
    screen_facing_aligned_2deg: usize,
}

impl Accepted {
    fn new() -> Self {
        Self {
            count: 0,
            min_m: Vec3::splat(f32::INFINITY),
            max_m: Vec3::splat(f32::NEG_INFINITY),
            size_cells_1cm: BTreeMap::new(),
            axis_aligned_2deg: 0,
            supported: 0,
            support_centered_2pct: 0,
            screen_facing_targets: 0,
            screen_facing_aligned_2deg: 0,
        }
    }
    fn add(&mut self, scene: &IndoorManifest, o: &IndoorObject) {
        self.count += 1;
        self.min_m = self.min_m.min(o.size);
        self.max_m = self.max_m.max(o.size);
        let cell = o.size.to_array().map(|v| (v * 100.).round() as i32);
        *self.size_cells_1cm.entry(format!("{cell:?}")).or_default() += 1;
        self.axis_aligned_2deg += usize::from(axis_distance(o.yaw) <= 2_f32.to_radians());
        if let Some(angle) = screen_facing_error(scene, o) {
            self.screen_facing_targets += 1;
            self.screen_facing_aligned_2deg += usize::from(angle.abs() <= 2_f32.to_radians());
            assert!(angle.abs() <= 0.19001);
        }
        if let Some(p) = support_coordinates(scene, o) {
            self.supported += 1;
            self.support_centered_2pct += usize::from(p.x.abs().max(p.z.abs()) <= 0.02);
        }
    }
}

#[test]
fn accepted_regularity_audit_covers_programs_and_preserves_geometry() {
    let mut report = BTreeMap::<String, BTreeMap<String, Accepted>>::new();
    let mut meshes = BTreeMap::<String, usize>::new();
    let mut showroom_kinds = BTreeMap::<String, usize>::new();
    let mut mixed_showrooms = 0;
    let mut multi_exhibit_showrooms = 0;
    let mut generation_ms = Vec::new();
    let mut schematic_rooms = Vec::new();
    for profile in IndoorLayout::PROFILES {
        for seed in 0..16 {
            let density = [0.35, 0.65, 0.95][seed as usize % 3];
            let started = std::time::Instant::now();
            let scene = IndoorManifest::generate_with_humans(seed, profile, density, 3, 0.25)
                .unwrap_or_else(|e| panic!("{profile:?}/{seed}: {e}"));
            generation_ms.push(started.elapsed().as_secs_f64() * 1000.);
            validation::validate_layout(&scene)
                .unwrap_or_else(|e| panic!("{profile:?}/{seed}: {e}"));
            if seed < 2 {
                validation::validate_geometry(&scene)
                    .unwrap_or_else(|e| panic!("{profile:?}/{seed} geometry: {e}"));
            }
            if seed == 0 {
                let replay =
                    IndoorManifest::generate_with_humans(seed, profile, density, 3, 0.25).unwrap();
                assert_eq!(
                    serde_json::to_vec(&scene).unwrap(),
                    serde_json::to_vec(&replay).unwrap()
                );
            }
            let mut exhibits = std::collections::BTreeSet::new();
            let mut exhibit_count = 0;
            for o in scene.objects.iter().filter(|o| !o.neighbor) {
                let kind = format!("{:?}", o.kind);
                report
                    .entry(format!("{profile:?}"))
                    .or_default()
                    .entry(kind.clone())
                    .or_insert_with(Accepted::new)
                    .add(&scene, o);
                let changed = matches!(
                    o.kind,
                    ObjectKind::Desk
                        | ObjectKind::Table
                        | ObjectKind::Chair
                        | ObjectKind::Cabinet
                        | ObjectKind::Bookcase
                        | ObjectKind::CoatRack
                        | ObjectKind::Keyboard
                        | ObjectKind::Mouse
                        | ObjectKind::Sculpture
                        | ObjectKind::DeskLamp
                        | ObjectKind::Vase
                        | ObjectKind::FloorLamp
                        | ObjectKind::ServiceCounter
                        | ObjectKind::ExtractionHood
                );
                if changed && *meshes.get(&kind).unwrap_or(&0) < 32 {
                    audit::check(o, &objects::build_object(o))
                        .unwrap_or_else(|e| panic!("{profile:?}/{seed}/{}: {e}", o.id));
                    *meshes.entry(kind).or_default() += 1;
                }
                if profile == IndoorLayout::Showroom
                    && o.support
                        .is_some_and(|id| scene.objects[id].kind == ObjectKind::ExhibitPlinth)
                {
                    exhibits.insert(format!("{:?}", o.kind));
                    exhibit_count += 1;
                    *showroom_kinds.entry(format!("{:?}", o.kind)).or_default() += 1;
                }
            }
            if profile == IndoorLayout::Showroom && exhibit_count > 1 {
                multi_exhibit_showrooms += 1;
                mixed_showrooms += usize::from(exhibits.len() > 1);
            }
            if seed < 3
                && matches!(
                    profile,
                    IndoorLayout::Cafe | IndoorLayout::Classroom | IndoorLayout::Showroom
                )
            {
                schematic_rooms.push(scene);
            }
        }
    }
    for profile in ["Classroom", "Cafe", "Cafeteria"] {
        let chairs = &report[profile]["Chair"];
        assert!(chairs.count >= 16, "{profile}: {}", chairs.count);
        assert!(
            chairs.size_cells_1cm.len() >= 12,
            "{profile}: chair sizes collapsed"
        );
        assert!((chairs.max_m - chairs.min_m).min_element() > 0.08);
        assert!(
            chairs.axis_aligned_2deg < chairs.count * 3 / 4,
            "{profile}: alignment collapsed"
        );
    }
    let desks = &report["Classroom"]["Desk"];
    assert!(desks.size_cells_1cm.len() >= 12);
    assert!((desks.max_m - desks.min_m).min_element() > 0.06);
    assert!(multi_exhibit_showrooms >= 8);
    assert!(mixed_showrooms * 2 > multi_exhibit_showrooms);
    assert_eq!(showroom_kinds.len(), 4);
    let total = |kind: &str, f: fn(&Accepted) -> usize| -> usize {
        report.values().filter_map(|k| k.get(kind)).map(f).sum()
    };
    let projectors = total("Projector", |s| s.supported);
    assert!(
        projectors >= 16,
        "projector admission collapsed: {projectors}"
    );
    assert!(total("Projector", |s| s.support_centered_2pct) * 4 < projectors);
    let screens = total("Monitor", |s| s.screen_facing_targets)
        + total("Laptop", |s| s.screen_facing_targets);
    let aligned = total("Monitor", |s| s.screen_facing_aligned_2deg)
        + total("Laptop", |s| s.screen_facing_aligned_2deg);
    assert!(screens > 64 && aligned * 2 < screens);
    for kind in ["Lectern", "TeachingBoard", "InformationStand", "FloorLamp"] {
        let count = total(kind, |s| s.count);
        assert!(count >= 8, "{kind}: admission collapsed");
        assert!(
            total(kind, |s| s.axis_aligned_2deg) * 2 < count,
            "{kind}: fixed heading"
        );
    }

    let root = std::path::Path::new("out/regularity_review");
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(root.join("accepted.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "generator_fingerprint": crate::CAPTURE_ENGINE_IDENTITY,
        "policy": POLICY,
        "rooms": 320,
        "full_room_mesh_checks": 40,
        "seeds_per_profile": [0, 15],
        "density_cycle": [0.35, 0.65, 0.95],
        "cameras_per_room": 3,
        "human_density": 0.25,
        "generation_ms": super::super::NumericDistribution::from_values(generation_ms),
        "timing_scope": "Single-thread CPU manifest generation including cameras and people; excludes mesh construction, render, validation and replay. Shared host, not a renderer throughput benchmark.",
        "accepted": report,
        "mesh_checks": meshes,
        "multi_exhibit_showrooms": multi_exhibit_showrooms,
        "mixed_exhibit_showrooms": mixed_showrooms,
        "showroom_exhibits": showroom_kinds,
    })).unwrap()).unwrap();
    std::fs::write(
        root.join("schematic-scenes.json"),
        serde_json::to_vec(&schematic_rooms).unwrap(),
    )
    .unwrap();
}

#[test]
fn regularity_metrics_export_has_explicit_denominators_and_finite_values() {
    let directory =
        std::env::temp_dir().join(format!("zeroverse-regularity-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let mut numeric = NumericCollector::new(&directory).unwrap();
    let mut desk_scenes = 0;
    let mut desks = 0;
    for seed in 0..8 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Classroom, 0.65, 0, 0.)
                .unwrap();
        record(&scene, &mut numeric).unwrap();
        let count = scene
            .objects
            .iter()
            .filter(|o| !o.neighbor && o.kind == ObjectKind::Desk)
            .count();
        desk_scenes += usize::from(count > 0);
        desks += count;
    }
    let report = numeric.finish().unwrap();
    assert!(desk_scenes > 0);
    assert_eq!(report["regularity_Desk_objects"].count, desk_scenes);
    assert_eq!(report["regularity_Desk_height_m"].count, desks);
    assert!(report["regularity_Desk_equal_size_pair_fraction_1cm"].max <= 1.);
    assert!(report["regularity_Notebook_support_x_fraction"].min >= -0.5);
    assert!(report["regularity_Notebook_support_x_fraction"].max <= 0.5);
    std::fs::remove_dir_all(directory).unwrap();
}
