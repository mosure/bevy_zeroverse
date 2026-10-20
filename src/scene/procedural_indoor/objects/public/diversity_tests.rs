use super::*;
use crate::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest},
    objects::audit,
    validation,
};
use std::collections::{BTreeMap, BTreeSet};

const REVIEWED: [ObjectKind; 21] = [
    ObjectKind::ServiceCounter,
    ObjectKind::ExtractionHood,
    ObjectKind::FloorLamp,
    ObjectKind::DisplayCase,
    ObjectKind::TrayReturn,
    ObjectKind::DrinkDispenser,
    ObjectKind::QueueBarrier,
    ObjectKind::Lectern,
    ObjectKind::TeachingBoard,
    ObjectKind::ExhibitPlinth,
    ObjectKind::InformationStand,
    ObjectKind::ExhibitionPanel,
    ObjectKind::Sculpture,
    ObjectKind::Vase,
    ObjectKind::Bowl,
    ObjectKind::Mug,
    ObjectKind::CoffeeCup,
    ObjectKind::WaterBottle,
    ObjectKind::Locker,
    ObjectKind::NoticeBoard,
    ObjectKind::Elevator,
];

#[test]
fn public_construction_programs_replay_with_metric_extremes_and_semantics() {
    let mut errors = Vec::new();
    for kind in REVIEWED {
        for seed in 0..64 {
            let mut o = audit::fixture(kind, seed);
            if kind == ObjectKind::Elevator && seed % 2 == 0 {
                o.variant = 20;
                o.size.y = 3.8;
            }
            o.size *= match seed % 3 {
                0 => Vec3::new(0.85, 1.15, 0.85),
                1 => Vec3::new(1.15, 0.85, 1.15),
                _ => Vec3::ONE,
            };
            let mesh = super::super::build_object(&o);
            if let Err(e) = audit::check(&o, &mesh) {
                errors.push(format!("{kind:?}/{seed}: {e}: {:?}", o.size));
                continue;
            }
            assert!(
                mesh.bounds().0.y >= -0.0001,
                "support penetration {kind:?}/{seed}"
            );
            let replay = super::super::build_object(&o);
            for (key, a) in mesh.parts {
                let b = &replay.parts[&key];
                assert_eq!(a.positions, b.positions);
                assert_eq!(a.indices, b.indices);
            }
        }
    }
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

#[test]
fn public_dimensions_have_category_specific_height_distributions() {
    for kind in [
        ObjectKind::DisplayCase,
        ObjectKind::TrayReturn,
        ObjectKind::ExhibitPlinth,
        ObjectKind::InformationStand,
        ObjectKind::DrinkDispenser,
        ObjectKind::TeachingBoard,
    ] {
        let sizes: Vec<_> = (0..128).map(|seed| dimensions(kind, seed)).collect();
        let min = sizes.iter().copied().reduce(Vec3::min).unwrap();
        let max = sizes.iter().copied().reduce(Vec3::max).unwrap();
        assert!(max.y / min.y > 1.35, "{kind:?}: {min:?}..{max:?}");
        let quantized: BTreeSet<_> = sizes
            .iter()
            .map(|s| s.to_array().map(|x| (x * 100.).round() as i32))
            .collect();
        assert!(quantized.len() >= 120, "{kind:?}");
    }
}

#[test]
fn plinths_retain_the_exact_flat_rectangular_support_contract() {
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::ExhibitPlinth, seed);
        let a = super::super::build_object(&o);
        audit::check(&o, &a).unwrap();
        assert!((a.bounds().1.y - o.size.y).abs() < 1e-5);
        let top_area: f32 = a
            .parts
            .values()
            .flat_map(|g| {
                g.indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(move |t| t.map(|i| Vec3::from_array(g.positions[i as usize])))
            })
            .filter(|p| p.iter().all(|v| (v.y - o.size.y).abs() < 1e-5))
            .map(|p| ((p[1] - p[0]).cross(p[2] - p[0]).y).max(0.) * 0.5)
            .sum();
        assert!(top_area > (o.size.x - 0.004) * (o.size.z - 0.004) * 0.99);
    }
}

#[test]
fn elevator_core_controls_stay_in_front_of_the_masonry_face() {
    for seed in 0..128 {
        let mut o = audit::fixture(ObjectKind::Elevator, seed);
        o.variant = 20;
        o.size.y = 2.66 + (seed % 21) as f32 * 0.10;
        let a = super::super::build_object(&o);
        audit::check(&o, &a).unwrap();
        assert!(elevator::car_height(&o) >= 2.50 - 1e-6);
        let controls: Vec<_> = a
            .parts
            .iter()
            .filter(|((_, label), _)| label == "control_panel")
            .collect();
        assert!(!controls.is_empty());
        assert!(controls
            .iter()
            .all(|(_, g)| g.positions.iter().all(|p| p[2] > o.size.z * 0.48)));
        assert!(a.parts.keys().any(|(_, l)| l == "wall"));
        assert!(a.parts.keys().any(|(_, l)| l == "door"));
    }
}

#[test]
fn gravity_dispenser_variants_have_no_fictitious_power_connector() {
    use super::super::equipment;
    let mut unpowered = 0;
    let mut powered = 0;
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::DrinkDispenser, seed);
        let port = equipment::source_power_port(&o);
        let normal = equipment::source_power_normal(&o);
        assert_eq!(port.is_some(), normal.is_some());
        assert!(equipment::power_port(&o, true).is_none());
        assert!(equipment::power_port_normal(&o, true).is_none());
        if let Some(port) = port {
            powered += 1;
            let a = super::super::build_object(&o);
            let inlet = &a.parts[&(Surface::Ink, o.kind.class_name().into())];
            assert!(inlet
                .positions
                .iter()
                .any(|p| Vec3::from_array(*p).distance(port) < o.size.x * 0.06));
        } else {
            unpowered += 1;
        }
    }
    assert!(powered > 32 && unpowered > 32);
}

#[test]
#[ignore = "bounded local CPU geometry/accepted-room study, writes out/public_regularity"]
fn public_object_regularity_population_qualification() {
    use crate::scene::procedural_indoor::architecture;
    use image::{Rgb, RgbImage};
    use serde_json::json;
    use std::{path::Path, time::Instant};
    let root = Path::new("out/public_regularity");
    std::fs::create_dir_all(root).unwrap();
    let mut rows = Vec::new();
    let mut sheet =
        RgbImage::from_pixel(8 * 168, REVIEWED.len() as u32 * 168, Rgb([241, 243, 246]));
    for (row, kind) in REVIEWED.into_iter().enumerate() {
        let fixed = audit::fixture(kind, 0).size;
        let mut descriptors = Vec::new();
        let mut times = Vec::new();
        let mut triangles = Vec::new();
        for seed in 0..64 {
            let mut o = audit::fixture(kind, seed);
            o.size = fixed;
            let start = Instant::now();
            let a = super::super::build_object(&o);
            times.push(start.elapsed().as_secs_f64() * 1000.);
            let check = audit::check(&o, &a).unwrap_or_else(|e| panic!("{kind:?}/{seed}: {e}"));
            triangles.push(check.triangles);
            descriptors.push(audit::descriptor(&o, &a));
            if seed < 8 {
                let thumb = crate::scene::procedural_indoor::geometry::preview::orthographic(
                    &a,
                    Vec3::Y * o.size.y * 0.5,
                    o.size.length() * 1.02,
                    168,
                    Vec3::new(0.7, 0.45, 2.),
                );
                image::imageops::replace(&mut sheet, &thumb, seed as i64 * 168, row as i64 * 168);
            }
        }
        let coverage: Vec<_> = [0.005, 0.015, 0.03]
            .into_iter()
            .map(
                |t| json!({"threshold":t,"representatives":audit::representatives(&descriptors,t)}),
            )
            .collect();
        rows.push(json!({"kind":kind,"fixed_size_m":fixed,"coverage":coverage,"construction_ms":times,"triangles":triangles}));
    }
    sheet.save(root.join("geometry-contact.png")).unwrap();
    let mut rooms = Vec::new();
    let mut counts = BTreeMap::<String, Vec<[f32; 3]>>::new();
    let mut failures = Vec::new();
    let mut elevator_cores = 0;
    let mut elevator_count = 0;
    let mut fixtures = BTreeMap::<String, Vec<[f32; 3]>>::new();
    for profile in IndoorLayout::PROFILES {
        for seed in 0..16 {
            let start = Instant::now();
            let s = match IndoorManifest::generate_with_humans(
                seed,
                profile,
                [0.35, 0.65, 0.95][seed as usize % 3],
                3,
                0.25,
            ) {
                Ok(s) => s,
                Err(e) => {
                    failures.push(json!({"profile":profile,"seed":seed,"stage":"generation","error":e.to_string()}));
                    continue;
                }
            };
            let manifest_ms = start.elapsed().as_secs_f64() * 1000.;
            for position in architecture::fixture_positions(&s) {
                fixtures
                    .entry(format!(
                        "{:?}",
                        architecture::fixtures::parameters_at(&s, position).shape
                    ))
                    .or_default()
                    .push(architecture::fixtures::size(&s, position).to_array());
            }
            if let Err(e) = validation::validate_layout(&s) {
                failures.push(json!({"profile":profile,"seed":seed,"stage":"layout","error":e}));
            }
            let mut mesh_ms = 0.;
            for o in s
                .objects
                .iter()
                .filter(|o| !o.neighbor && REVIEWED.contains(&o.kind))
            {
                counts
                    .entry(format!("{:?}", o.kind))
                    .or_default()
                    .push(o.size.to_array());
                elevator_count += usize::from(o.kind == ObjectKind::Elevator);
                elevator_cores += usize::from(elevator::is_core(o));
                let now = Instant::now();
                let a = super::super::build_object(o);
                mesh_ms += now.elapsed().as_secs_f64() * 1000.;
                if let Err(e) = audit::check(o, &a) {
                    failures.push(
                        json!({"profile":profile,"seed":seed,"stage":"object mesh","error":e}),
                    );
                }
            }
            if seed < 2 {
                if let Err(e) = validation::validate_geometry(&s) {
                    failures
                        .push(json!({"profile":profile,"seed":seed,"stage":"room mesh","error":e}));
                }
            }
            rooms.push(json!({"profile":profile,"seed":seed,"manifest_ms":manifest_ms,"reviewed_mesh_ms":mesh_ms,"objects":s.objects.len(),"cameras":s.cameras.len()}));
        }
    }
    let report = json!({"generator":crate::CAPTURE_ENGINE_IDENTITY,"scope":"CPU actual production geometry, fixed-envelope six-view depth descriptors; no PBR/SigLIP/training-utility claims", "geometry":rows,"accepted_dimensions_m":counts,"fixture_dimensions_m":fixtures,"rooms":rooms,"failures":failures,"elevator_cores":elevator_cores,"elevators":elevator_count});
    std::fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    assert!(
        failures.is_empty(),
        "{}",
        serde_json::to_string_pretty(&failures).unwrap()
    );
}
