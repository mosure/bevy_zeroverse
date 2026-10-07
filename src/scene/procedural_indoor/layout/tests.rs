use super::*;
use crate::scene::procedural_indoor::validation::{validate_geometry, validate_layout};

#[test]
fn tucked_seating_and_surface_programs_cover_occupied_and_empty_cases() {
    use super::super::{envelope::polygon, humans, objects};
    let mut total = 0;
    let mut seated_total = 0;
    let mut tucked = 0;
    let mut occupied = 0;
    let mut empty = 0;
    let mut occupied_poses = std::collections::BTreeMap::new();
    let mut hands_over_table = 0;
    let mut contact_people = 0;
    let mut contact_palms = 0;
    let mut max_compression = 0.0_f32;
    let mut max_skin_compression = 0.0_f32;
    let mut contact_errors = Vec::new();
    let mut supports = std::collections::BTreeSet::new();
    let mut notebook_cells = std::collections::BTreeSet::new();
    for seed in 0..512 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.7, 3, 0.7).unwrap();
        validate_layout(&scene).unwrap();
        let replay =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.7, 3, 0.7).unwrap();
        assert_eq!(scene, replay);
        for human in scene.humans.iter().filter(|h| h.worktop_contact.is_some()) {
            let contact = human.worktop_contact.as_ref().unwrap();
            let mesh = humans::build_human(human);
            let replay: humans::IndoorHuman =
                serde_json::from_str(&serde_json::to_string(human).unwrap()).unwrap();
            assert_eq!(*human, replay);
            assert!(
                humans::standing_at(human, human.position, human.yaw, human.neighbor)
                    .worktop_contact
                    .is_none()
            );
            for base in [5, 9] {
                let s = human.stature / 1.75;
                assert!(
                    (mesh.local_joints[base].distance(mesh.local_joints[base + 1]) - 0.29 * s)
                        .abs()
                        < 1e-5
                );
                assert!(
                    (mesh.local_joints[base + 1].distance(mesh.local_joints[base + 2]) - 0.26 * s)
                        .abs()
                        < 1e-5
                );
            }
            assert_eq!(
                mesh.contacts.len(),
                contact.palms.iter().filter(|&&v| v).count()
            );
            contact_people += 1;
            contact_palms += mesh.contacts.len();
            for report in &mesh.contacts {
                max_compression = max_compression.max(report.cloth_compression_metres);
                max_skin_compression = max_skin_compression.max(report.skin_compression_metres);
                if !(report.palm_gap_metres >= -1e-5
                    && report.palm_gap_metres < 0.003
                    && report.minimum_gap_metres >= -1e-5
                    && report.minimum_gap_metres < 0.003
                    && report.contact_vertices > 0
                    && report.cloth_compression_metres <= 0.025
                    && report.skin_compression_metres <= 0.006)
                {
                    contact_errors
                        .push(format!("seed={seed} human={} contact={report:?}", human.id));
                }
            }
            let table = &scene.objects[contact.table];
            let solid = objects::tables::Clearance::new(table);
            let tf =
                table.transform().compute_affine().inverse() * human.transform().compute_affine();
            for (surface, geometry) in &mesh.parts {
                for v in &geometry.positions {
                    let p = tf.transform_point3(Vec3::from_array(*v));
                    assert!(
                        p.y >= table.size.y - 1e-5 || !solid.hits_capsule(p, p, 0.0001),
                        "contact mesh clips seed={seed} human={} surface={surface:?} p={p}",
                        human.id
                    );
                }
            }
        }
        seated_total += scene
            .humans
            .iter()
            .filter(|h| h.chair.is_some() && !h.neighbor)
            .count();
        for chair in scene
            .objects
            .iter()
            .filter(|o| !o.neighbor && o.kind == ObjectKind::Chair)
        {
            let Some(table) = chair
                .interaction_target
                .and_then(|id| scene.objects.get(id))
            else {
                continue;
            };
            total += 1;
            let outline = objects::tables::outline(table);
            let inverse = table.transform().compute_affine().inverse();
            let tf = inverse * chair.transform().compute_affine();
            let mesh = objects::build_object(chair);
            let inserted = mesh.parts.values().flat_map(|g| &g.positions).any(|&v| {
                let p = Vec3::from_array(v);
                p.y > 0.42
                    && p.y < 0.48
                    && polygon::contains(&outline, tf.transform_point3(p).xz(), 0.)
            });
            if !inserted {
                continue;
            }
            tucked += 1;
            supports.insert(objects::tables::parameters(table).support);
            let solid = objects::tables::Clearance::new(table);
            for v in mesh.parts.values().flat_map(|g| &g.positions) {
                let p = tf.transform_point3(Vec3::from_array(*v));
                assert!(
                    !solid.hits_capsule(p, p, 0.001),
                    "chair mesh clipping seed={seed} chair={}",
                    chair.id
                );
            }
            if let Some(person) = scene.humans.iter().find(|h| h.chair == Some(chair.id)) {
                occupied += 1;
                *occupied_poses
                    .entry(format!("{:?}", person.pose))
                    .or_insert(0) += 1;
                let tf = inverse * person.transform().compute_affine();
                let mesh = humans::build_human(person);
                if [8, 12].into_iter().any(|i| {
                    let p = tf.transform_point3(mesh.local_joints[i]);
                    p.y > table.size.y + 0.02 && polygon::contains(&outline, p.xz(), 0.005)
                }) {
                    hands_over_table += 1;
                }
                for (surface, geometry) in &mesh.parts {
                    for v in &geometry.positions {
                        let p = tf.transform_point3(Vec3::from_array(*v));
                        assert!(
                            p.y >= table.size.y - 1e-5 || !solid.hits_capsule(p, p, if person.worktop_contact.is_some() { 0.0001 } else { 0.001 }),
                            "human mesh clipping seed={seed} person={} table={} surface={surface:?} p={p}",
                            person.id,
                            table.id
                        );
                    }
                }
            } else {
                empty += 1;
            }
        }
        for o in scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Notebook)
        {
            let Some(s) = o.support.map(|id| &scene.objects[id]) else {
                continue;
            };
            let p = s
                .transform()
                .compute_affine()
                .inverse()
                .transform_point3(o.position)
                / s.size;
            notebook_cells.insert(((p.x * 10.).floor() as i32, (p.z * 10.).floor() as i32));
        }
    }
    println!("seating_coverage total={total} seated_total={seated_total} tucked={tucked} occupied={occupied} empty={empty} support_families={} notebook_cells={}",supports.len(),notebook_cells.len());
    println!("tucked occupied poses={occupied_poses:?}");
    println!("tucked occupied with hands over table={hands_over_table}");
    println!("supported people={contact_people} palms={contact_palms} max_cloth_compression={max_compression}");
    println!("max_skin_compression={max_skin_compression}");
    assert!(contact_errors.is_empty(), "{}", contact_errors.join("\n"));
    assert!(contact_people >= 8 && contact_palms > contact_people);
    assert!(tucked > total / 8 && occupied >= 32 && empty >= 128);
    assert!(
        hands_over_table * 8 >= occupied && occupied_poses.len() == 3,
        "tucked seats lost their working/listening/talking mixture: {hands_over_table}/{occupied} tabletop reaches"
    );
    assert!(supports.len() >= 5 && notebook_cells.len() >= 30);
}

#[test]
fn reported_sunken_room_props_remain_on_their_supports() {
    for seed in [47_586_113, 47_359_576, 47_062_729] {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 4, 0.25).unwrap();
        for prop in scene.objects.iter().filter(|o| o.support.is_some()) {
            let support = &scene.objects[prop.support.unwrap()];
            assert!(
                (prop.position.y - support.position.y - support.size.y).abs() < 1e-5,
                "seed {seed}: {:?} {} at {} on {:?} {} at {} + {}",
                prop.kind,
                prop.id,
                prop.position.y,
                support.kind,
                support.id,
                support.position.y,
                support.size.y
            );
        }
        validate_layout(&scene).unwrap();
        validate_geometry(&scene).unwrap();
    }
}

#[test]
fn supported_props_keep_exact_heights_across_world_zero() {
    let mut scene =
        IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    scene.program = None; // Exercise placement without probabilistic clutter thinning.
    let mut support = scene.objects[0].clone();
    support.id = 0;
    support.kind = ObjectKind::Cabinet;
    support.size = Vec3::new(1., 0.42, 1.);
    support.yaw = 0.7;
    support.support = None;
    support.neighbor = false;
    support.interaction_target = None;
    for top in [
        -0.001, -0.0001, -0.00005, 0., 0.00005, 0.0001, 0.001, 0.42, 1.5,
    ] {
        support.position = Vec3::new(0., top - support.size.y, 0.);
        scene.envelope.as_mut().unwrap().floor_patches = vec![super::super::envelope::FloorPatch {
            min: Vec2::splat(-2.),
            max: Vec2::splat(2.),
            height: support.position.y,
        }];
        scene.objects = vec![support.clone()];
        let offset = Vec3::new(0.1, 0., -0.1);
        scene.prop(
            &support,
            ObjectKind::Mug,
            offset,
            Vec3::new(0.12, 0.1, 0.09),
            0.2,
            &mut stream(0, 99),
        );
        assert_eq!(scene.objects.len(), 2, "top={top}: prop must be retained");
        let prop = &scene.objects[1];
        let expected = support
            .transform()
            .transform_point(offset + Vec3::Y * support.size.y);
        assert_eq!(
            prop.position, expected,
            "top={top}: support contact must be exact"
        );
        assert_eq!(prop.support, Some(support.id));
    }
}

#[test]
fn reported_sunken_room_seeds_cover_density_and_activity_variants() {
    for seed in [47_586_113, 47_359_576, 47_062_729] {
        for layout in IndoorLayout::PROFILES {
            for density in [0.15, 0.65, 1.] {
                let scene =
                    IndoorManifest::generate_with_humans(seed, layout, density, 3, 0.25).unwrap();
                validate_layout(&scene).unwrap();
                validate_geometry(&scene).unwrap_or_else(|e| {
                    panic!("seed={seed},layout={layout:?},density={density}: {e}")
                });
            }
        }
    }
}
