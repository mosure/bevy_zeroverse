use super::*;
use crate::scene::procedural_indoor::{objects::audit, validation};
use std::collections::{BTreeMap, BTreeSet};
mod qualification;

fn fixture(kind: ObjectKind, seed: u64) -> (IndoorManifest, IndoorObject) {
    let mut scene =
        IndoorManifest::generate_with_humans(14, IndoorLayout::Conference, 0.65, 0, 0.).unwrap();
    scene.room_size = Vec3::new(12., 5., 12.);
    scene.envelope = None;
    scene.ceiling_installation = None;
    let mut support = audit::fixture(kind, seed);
    support.id = 0;
    support.position = if kind == ObjectKind::WallShelf {
        Vec3::Y * 1.3
    } else {
        Vec3::ZERO
    };
    support.yaw = 0.6;
    scene.objects = vec![support.clone()];
    (scene, support)
}

#[test]
fn all_plant_supports_get_varied_supported_canopies_and_replay_exactly() {
    let mut headings = BTreeSet::new();
    let mut quadrants = BTreeSet::new();
    for kind in [
        ObjectKind::Desk,
        ObjectKind::Table,
        ObjectKind::CoffeeTable,
        ObjectKind::SideTable,
        ObjectKind::Cabinet,
        ObjectKind::DrawerUnit,
        ObjectKind::Bookcase,
        ObjectKind::WallShelf,
        ObjectKind::KitchenCounter,
    ] {
        let (base, _) = fixture(kind, 0);
        let mut accepted = 0;
        for seed in 0..24 {
            let mut scene = base.clone();
            let mut support = scene.objects[0].clone();
            support.seed = seed;
            scene.objects[0] = support.clone();
            let program = BotanyProgram::sample(seed);
            let mut rng = stream(seed, 6262);
            let mut replay = scene.clone();
            let mut replay_rng = rng.clone();
            let placed = scene.plant_on_surface(&support, &program, &mut rng);
            assert_eq!(
                placed,
                replay.plant_on_surface(&support, &program, &mut replay_rng)
            );
            assert_eq!(scene.objects, replay.objects);
            if !placed {
                continue;
            }
            accepted += 1;
            let plant = scene.objects.last().unwrap();
            assert_eq!(plant.support, Some(support.id));
            assert!(!plant.solid);
            assert!((plant.position.y - support.position.y - support.size.y).abs() < 1e-5);
            assert!(scene.plant_surface_clear(plant, &support));
            audit::check(plant, &objects::build_object(plant)).unwrap();
            let local = support
                .transform()
                .compute_affine()
                .inverse()
                .transform_point3(plant.position);
            quadrants.insert((local.x > 0., local.z > 0.));
            headings.insert(((plant.yaw - support.yaw).rem_euclid(TAU) * 12. / TAU) as u32);
        }
        assert!(
            accepted >= 16,
            "{kind:?}: only {accepted}/24 fit attempts succeeded"
        );
    }
    assert_eq!(headings.len(), 12);
    assert_eq!(quadrants.len(), 4);
}

#[test]
fn surface_plants_respect_upper_cabinets_ceiling_and_clear_table_controls() {
    let (mut scene, support) = fixture(ObjectKind::KitchenCounter, 0);
    let mut blocker = audit::fixture(ObjectKind::KitchenCupboard, 1);
    blocker.id = 1;
    blocker.position = Vec3::Y * (support.size.y + 0.02);
    blocker.size = Vec3::new(4., 2., 4.);
    scene.objects.push(blocker);
    let program = BotanyProgram::sample(0);
    assert!(!scene.plant_on_surface(&support, &program, &mut stream(0, 6262)));
    scene.objects.truncate(1);
    scene.room_size.y = support.size.y + 0.04;
    assert!(!scene.plant_on_surface(&support, &program, &mut stream(0, 6262)));
    for seed in 0..24 {
        let (mut scene, _) = fixture(ObjectKind::CoffeeTable, seed);
        scene.clutter.probability = 0.;
        scene.seed = seed;
        scene.furnish_supported_botany();
        assert_eq!(scene.objects.len(), 1);
    }
}

#[test]
fn default_rooms_have_elevated_plants_beyond_work_desks_with_valid_meshes() {
    let mut supports = BTreeMap::<String, usize>::new();
    let mut elevated_rooms = 0;
    let mut rows = Vec::new();
    let mut timings = Vec::new();
    for seed in 0..64 {
        let begin = std::time::Instant::now();
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        timings.push(begin.elapsed().as_secs_f64() * 1000.);
        validation::validate_layout(&scene).unwrap_or_else(|e| panic!("room {seed}: {e}"));
        let mut elevated = false;
        for plant in scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Plant && !o.neighbor)
        {
            let support = plant.support.map(|id| &scene.objects[id]);
            let label = support.map_or("floor", |o| o.kind.class_name());
            *supports.entry(label.into()).or_default() += 1;
            elevated |= support.is_some();
            audit::check(plant, &objects::build_object(plant))
                .unwrap_or_else(|e| panic!("room {seed} plant {}: {e}", plant.id));
            let (radius, height) =
                plants::pot_dimensions(plant, &plants::Growth::sample(plant.seed));
            rows.push(serde_json::json!({"room_seed":seed,"object_id":plant.id,"species":plants::species_name(plant.variant),
                "support":label,"position_m":plant.position,"size_m":plant.size,"yaw_radians":plant.yaw,
                "pot_radius_m":radius,"pot_height_m":height,"pot_height_to_diameter":height/(2.*radius),
                "pot":plants::PotProfile::sample(plant.seed)}));
            if let Some(support) = support {
                assert!(scene.prop_clear(plant, support, 0.012));
            }
        }
        elevated_rooms += usize::from(elevated);
    }
    eprintln!("64 default rooms: {elevated_rooms} have elevated plants; supports={supports:?}");
    assert!(
        elevated_rooms >= 24,
        "too few elevated plants: {elevated_rooms}"
    );
    assert!(supports.len() >= 6, "too few support types: {supports:?}");
    assert!(
        supports.get("shelves").copied().unwrap_or(0) >= 3,
        "wall-shelf plants missing: {supports:?}"
    );
    qualification::save(&rows, &supports, elevated_rooms, &timings);
}
