//! Local, source-bound qualification. Consecutive requests; failures stay in the denominator.
use bevy_zeroverse::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest, ObjectKind},
    validation::{validate_geometry, validate_layout},
};
use clap::Parser;
use std::{
    collections::{BTreeMap, BTreeSet},
    hash::{Hash, Hasher},
    path::PathBuf,
    time::Instant,
};
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 64)]
    rooms: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    geometry: bool,
    #[arg(long, default_value_t = 0)]
    cameras: usize,
    #[arg(long, default_value_t = 0.)]
    humans: f32,
    #[arg(long, default_value_t = 0.65)]
    density: f32,
    /// Layouts to audit. Omit to qualify all public venues and the public suite.
    #[arg(long, value_enum, value_delimiter = ',')]
    profiles: Vec<IndoorLayout>,
    #[arg(long, default_value = "out/public_programs/audit.json")]
    output: PathBuf,
}
fn main() {
    let args = Args::parse();
    let mut profiles = Vec::new();
    let started = Instant::now();
    let requested = if args.profiles.is_empty() {
        IndoorLayout::PUBLIC_PROFILES
            .into_iter()
            .chain([IndoorLayout::PublicSuite])
            .collect()
    } else {
        args.profiles
    };
    for profile in requested {
        let t = Instant::now();
        let mut failures = Vec::new();
        let mut counts = BTreeMap::<String, usize>::new();
        let mut capacities = Vec::new();
        let mut vertices = 0u64;
        let mut timings = BTreeMap::<&str, Vec<f64>>::new();
        let mut shapes = BTreeSet::new();
        let mut parameters = Vec::new();
        let mut hallways = Vec::new();
        let mut stair_halls = Vec::new();
        let mut courtyards = Vec::new();
        let mut courts = Vec::new();
        let mut workouts = Vec::new();
        let mut dining = Vec::new();
        let mut markets = Vec::new();
        let mut checkouts = Vec::new();
        let mut apparel = Vec::new();
        for i in 0..args.rooms {
            let seed = args.seed.wrapping_add(i as u64);
            let result = (|| {
                let generation = Instant::now();
                let s = IndoorManifest::generate_with_humans(
                    seed,
                    profile,
                    args.density,
                    args.cameras,
                    args.humans,
                )?;
                timings
                    .entry("manifest_ms")
                    .or_default()
                    .push(generation.elapsed().as_secs_f64() * 1000.);
                let validation = Instant::now();
                validate_layout(&s)?;
                timings
                    .entry("layout_validation_ms")
                    .or_default()
                    .push(validation.elapsed().as_secs_f64() * 1000.);
                if args.geometry {
                    let geometry = Instant::now();
                    vertices += validate_geometry(&s)?.vertices as u64;
                    timings
                        .entry("actual_geometry_validation_ms")
                        .or_default()
                        .push(geometry.elapsed().as_secs_f64() * 1000.);
                }
                let mut hash = std::collections::hash_map::DefaultHasher::new();
                for o in s.objects.iter().filter(|o| !o.neighbor) {
                    format!("{:?}", o.kind).hash(&mut hash);
                    o.position.to_array().map(f32::to_bits).hash(&mut hash);
                    o.size.to_array().map(f32::to_bits).hash(&mut hash);
                    o.yaw.to_bits().hash(&mut hash);
                }
                shapes.insert(hash.finish());
                for o in s.objects.iter().filter(|o| !o.neighbor) {
                    *counts.entry(format!("{:?}", o.kind)).or_default() += 1;
                }
                capacities.extend(s.functional_programs().iter().map(|p| p.accepted_capacity));
                parameters.extend(s.functional_programs().iter().map(|p| serde_json::json!({"seed":seed,"activity":p.activity,"parameters":p.parameters,"requested_capacity":p.requested_capacity,"accepted_capacity":p.accepted_capacity,"capacity_unit":p.capacity_unit,"rejected_candidates":p.rejected_candidates,"reservations":p.reservations.len(),"services":p.services.len()})));
                for (zone, p) in s
                    .functional_programs()
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.activity == IndoorLayout::GroceryStore)
                {
                    use bevy_zeroverse::scene::procedural_indoor::objects::public::refrigerated_case;
                    let stock: Vec<_> = p
                        .members
                        .iter()
                        .map(|&id| &s.objects[id])
                        .filter(|o| o.kind == ObjectKind::GroceryDisplay)
                        .collect();
                    let cases: Vec<_> = stock
                        .iter()
                        .filter_map(|o| {
                            refrigerated_case(o).map(|c| serde_json::json!({
                        "id":o.id,"position":o.position,"yaw":o.yaw,"size":o.size,"construction":c
                    }))
                        })
                        .collect();
                    let produce = stock.iter().filter(|o| o.size.y < 1.15).count();
                    markets.push(serde_json::json!({"seed":seed,"zone":zone,"produce":produce,"ambient":stock.len()-produce-cases.len(),"refrigerated":cases}));
                }
                for (zone, p) in s
                    .functional_programs()
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.activity == IndoorLayout::ClothingStore)
                {
                    use bevy_zeroverse::scene::procedural_indoor::objects::public::fitting_room;
                    let booths: Vec<_> = p.members.iter().filter_map(|&id| {
                        let o = &s.objects[id];
                        fitting_room(o).map(|spec| serde_json::json!({
                            "id":id,"position":o.position,"yaw":o.yaw,"size":o.size,"construction":spec
                        }))
                    }).collect();
                    let cameras: Vec<_> = s.cameras.iter().map(|camera| {
                        let pose = camera.transform_at(0.);
                        serde_json::json!({"eye":pose.translation,"forward":*pose.forward(),"fov_degrees":camera.fov_degrees})
                    }).collect();
                    apparel.push(serde_json::json!({"seed":seed,"zone":zone,"merchandise":p.accepted_capacity,"fitting_rooms":booths,"initial_camera_poses":cameras}));
                }
                for (zone, p) in s
                    .functional_programs()
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.activity.is_retail())
                {
                    use bevy_zeroverse::scene::procedural_indoor::objects::public::checkout_station;
                    let stations:Vec<_>=p.members.iter().filter_map(|&id|{
                        let o=&s.objects[id];
                        checkout_station(o).map(|station| serde_json::json!({
                            "id":id,"position":o.position,"yaw":o.yaw,"size":o.size,"construction":station
                        }))
                    }).collect();
                    let cameras:Vec<_>=s.cameras.iter().map(|c|{
                        let pose=c.transform_at(0.);
                        serde_json::json!({"eye":pose.translation,"forward":*pose.forward(),"fov_degrees":c.fov_degrees,"fisheye":c.fisheye})
                    }).collect();
                    checkouts.push(serde_json::json!({"seed":seed,"zone":zone,"activity":p.activity,"merchandise":p.accepted_capacity,"checkouts":stations,"initial_camera_poses":cameras}));
                }
                if let Some(court) = s.program.as_ref().and_then(|p| p.volleyball) {
                    let nets:Vec<_>=s.objects.iter().filter(|o|o.kind==ObjectKind::VolleyballNet && !o.neighbor).map(|o|serde_json::json!({"center":o.position,"yaw":o.yaw,"size":o.size,"construction":bevy_zeroverse::scene::procedural_indoor::objects::public::volleyball::Net::for_object(o)})).collect();
                    let benches = s
                        .objects
                        .iter()
                        .filter(|o| o.kind == ObjectKind::Bench && !o.neighbor)
                        .count();
                    let stands: Vec<_> = s.objects.iter()
                        .filter(|o| !o.neighbor && o.kind == ObjectKind::BleacherBank)
                        .map(|o| {
                            let bank=bevy_zeroverse::scene::procedural_indoor::objects::public::bleachers::BleacherBank::for_object(o);
                            serde_json::json!({"id":o.id,"position":o.position,"yaw":o.yaw,"size":o.size,"spectator_places":bank.seats(),"construction":bank})
                        }).collect();
                    courts.push(serde_json::json!({"seed":seed,"court":court,"room_size":s.room_size,"nets":nets,"sideline_benches":benches,"bleachers":stands}));
                }
                if s.layout.is_exercise() {
                    let p = &s.functional_programs()[0];
                    let count = |kind| {
                        p.members
                            .iter()
                            .filter(|&&id| s.objects[id].kind == kind)
                            .count()
                    };
                    let instructor = p.cycling_instructor(&s.objects);
                    workouts.push(serde_json::json!({
                        "seed": seed, "bikes": count(ObjectKind::StationaryBike),
                        "treadmills": count(ObjectKind::Treadmill), "benches":count(ObjectKind::WeightBench),
                        "mats":count(ObjectKind::ExerciseMat), "racks":count(ObjectKind::DumbbellRack),
                        "power_racks":p.members.iter().map(|&i| &s.objects[i]).filter(|o|o.kind==ObjectKind::PowerRack).map(|o|serde_json::json!({
                            "object":o, "construction":bevy_zeroverse::scene::procedural_indoor::objects::public::strength::PowerRack::for_object(o)
                        })).collect::<Vec<_>>(),
                        "instructor_id":instructor.map(|o|o.id), "instructor_yaw":instructor.map(|o|o.yaw),
                        "aisle_width_m":p.reservations[0].max.x-p.reservations[0].min.x,
                    }));
                }
                for (zone, p) in s.functional_programs().iter().enumerate().filter(|(_, p)| {
                    matches!(
                        p.activity,
                        IndoorLayout::Restaurant | IndoorLayout::Cafe | IndoorLayout::Cafeteria
                    )
                }) {
                    use bevy_zeroverse::scene::procedural_indoor::objects::tables::dining_linen;
                    let tables: Vec<_> = p
                        .members
                        .iter()
                        .map(|&i| &s.objects[i])
                        .filter(|o| o.kind == ObjectKind::Table)
                        .collect();
                    let dressed: Vec<_> = tables
                        .iter()
                        .filter_map(|o| {
                            dining_linen(o)
                                .map(|cloth| serde_json::json!({"id":o.id,"linen":cloth}))
                        })
                        .collect();
                    let ids: BTreeSet<_> = tables.iter().map(|t| t.id).collect();
                    let wells: Vec<_> = p.members.iter().map(|&i| &s.objects[i]).filter(|o|o.kind==ObjectKind::HotFoodWell).map(|o|serde_json::json!({
                        "object":o,"counter":s.objects[o.support.unwrap()], "installation":bevy_zeroverse::scene::procedural_indoor::objects::public::hot_food::HotFoodWell::for_object(o)
                    })).collect();
                    dining.push(serde_json::json!({"seed":seed, "zone":zone, "activity":p.activity, "tables":tables.len(), "dressed":dressed,
                        "hot_food_wells":wells, "seats":p.accepted_capacity, "served_items":s.objects.iter()
                        .filter(|o| o.support.is_some_and(|id|ids.contains(&id))).count()}));
                }
                if s.layout == IndoorLayout::Hallway {
                    let topology = s.room_topology();
                    let (lo, hi) = s.primary_room_bounds();
                    let span = hi - lo;
                    let axis = usize::from(span.x > span.y);
                    let mut banks = [false; 2];
                    let mut fronts = 0;
                    let mut glass_fronts = 0;
                    for portal in topology.portals.iter().filter(|p| p.regions.contains(&0)) {
                        let wall = &s.program.as_ref().unwrap().partitions[portal.partition];
                        banks[usize::from(wall.coordinate > (lo[axis] + hi[axis]) * 0.5)] = true;
                        fronts += 1;
                        glass_fronts += usize::from(wall.glazing_fraction > 0.);
                    }
                    hallways.push(serde_json::json!({"seed": seed, "length_m": span.max_element(),
                        "width_m": span.min_element(), "long_axis": if axis == 0 {"Z"} else {"X"},
                        "banks": banks.into_iter().filter(|v| *v).count(), "suites":topology.regions.len()-1,
                        "front_doors":fronts, "glazed_fronts":glass_fronts, "communicating_portals":topology.portals.len()-fronts,
                        "furnished_suites":s.functional_programs()[1..].iter().filter(|p|p.accepted_capacity>0).count(),
                        "waiting_stations":s.functional_programs()[0].accepted_capacity}));
                }
                if s.layout == IndoorLayout::Courtyard {
                    use bevy_zeroverse::scene::procedural_indoor::{
                        architecture::fixture_positions, layout::ObjectKind, materials::Surface,
                    };
                    let envelope = s.envelope.as_ref().ok_or("missing courtyard envelope")?;
                    let roof = envelope.courtyard.ok_or("missing open roof")?;
                    let floor = &s.program.as_ref().unwrap().materials[Surface::Floor as usize];
                    let p = &s.functional_programs()[0];
                    let species: Vec<_> = p
                        .members
                        .iter()
                        .map(|&i| &s.objects[i])
                        .filter(|o| o.kind == ObjectKind::Plant)
                        .map(|o| o.variant)
                        .collect();
                    let meter = s
                        .domain()
                        .and_then(|d| d.photometry.meter)
                        .ok_or("missing courtyard meter")?;
                    courtyards.push(serde_json::json!({
                        "seed": seed, "room_m": s.room_size, "open_area_fraction":roof.open_area_m2()/(s.room_size.x*s.room_size.z),
                        "opening_min":roof.min,"opening_max":roof.max,"slab_m":roof.slab_m,"soffit":roof.soffit,
                        "columns":envelope.pillars.len(),"plant_species":species,"seats":p.accepted_capacity,
                        "planting_beds":p.members.iter().filter(|&&id|s.objects[id].kind==ObjectKind::PlantingBed).count(),
                        "paving":floor.flooring.as_ref().unwrap().layout,"roughness":floor.roughness,
                        "fixture_count":fixture_positions(&s).iter().filter(|p|p.z<s.room_size.z*0.5).count(),
                        "diffuse_daylight_lux":meter.diffuse_daylight_lux,"direct_daylight_lux":meter.direct_daylight_lux
                    }));
                }
                if s.layout == IndoorLayout::StairHall {
                    let m = s
                        .envelope
                        .as_ref()
                        .and_then(|e| e.mezzanine.as_ref())
                        .ok_or("missing stair hall flight")?;
                    let c = m
                        .construction
                        .as_ref()
                        .ok_or("missing stair hall construction")?;
                    let p = &s.functional_programs()[0];
                    stair_halls.push(serde_json::json!({
                        "seed":seed, "axis":m.stair_axis, "ascending_positive":m.ascending_positive(),
                        "gallery_m":m.deck.height, "width_m":m.stair_width(), "risers":m.steps,
                        "riser_m":m.riser_height(), "tread_m":m.tread_depth(), "structure":c.stair_structure,
                        "guard":c.guard.infill, "frame":c.frame_surface, "tread_surface":c.tread_surface,
                        "mid_landing":c.landing.is_some(), "upper_seats":p.accepted_capacity, "upper_contents":p.members.iter().map(|&i|&s.objects[i])
                            .filter(|o| o.solid && (o.position.y-m.deck.height).abs()<0.001).map(|o|o.kind).collect::<Vec<_>>()
                    }));
                }
                Ok::<_, String>(())
            })();
            if let Err(error) = result {
                failures.push(serde_json::json!({"seed":seed,"error":error}));
            }
        }
        eprintln!(
            "{profile:?}: {}/{} passed in {:.2}s",
            args.rooms - failures.len(),
            args.rooms,
            t.elapsed().as_secs_f64()
        );
        let timing_summary: BTreeMap<_,_> = timings.into_iter().map(|(name,mut values)| {
            values.sort_by(f64::total_cmp);
            let n=values.len();
            (name,serde_json::json!({"count":n,"mean":values.iter().sum::<f64>()/n.max(1) as f64,"p50":values.get(n/2),"p95":values.get(n*95/100),"p99":values.get(n*99/100),"max":values.last()}))
        }).collect();
        let mut summary = serde_json::json!({"profile":profile,"requested":args.rooms,"failures":failures,"counts":counts,"capacities":capacities,"programs":parameters,"unique_structural_fingerprints":shapes.len(),"timings":timing_summary,"vertices":vertices,"seconds":t.elapsed().as_secs_f64()});
        if !courts.is_empty() {
            summary["gymnasiums"] = serde_json::json!(courts);
        }
        if !apparel.is_empty() {
            summary["apparel"] = serde_json::json!(apparel);
        }
        if !checkouts.is_empty() {
            summary["checkouts"] = serde_json::json!(checkouts);
        }
        if !markets.is_empty() {
            summary["grocery"] = serde_json::json!(markets);
        }
        if !dining.is_empty() {
            summary["dining"] = serde_json::json!(dining);
        }
        if !workouts.is_empty() {
            summary["workouts"] = serde_json::json!(workouts);
        }
        if !hallways.is_empty() {
            summary["hallways"] = serde_json::json!(hallways);
        }
        if !courtyards.is_empty() {
            summary["courtyards"] = serde_json::json!(courtyards);
        }
        if !stair_halls.is_empty() {
            summary["stair_halls"] = serde_json::json!(stair_halls);
        }
        profiles.push(summary);
    }
    std::fs::create_dir_all(args.output.parent().unwrap()).unwrap();
    let failed = profiles
        .iter()
        .any(|p| !p["failures"].as_array().unwrap().is_empty());
    std::fs::write(&args.output,serde_json::to_vec_pretty(&serde_json::json!({"policy":"Consecutive seeds; no quality selection; all failures retained. Structural fingerprints hash exact accepted object kinds/transforms/dimensions, not appearance or a claim of perceptual uniqueness.","provenance":bevy_zeroverse::provenance::capture_provenance(),"first_seed":args.seed,"geometry":args.geometry,"cameras":args.cameras,"humans":args.humans,"density":args.density,"seconds":started.elapsed().as_secs_f64(),"profiles":profiles})).unwrap()).unwrap();
    if failed {
        std::process::exit(1);
    }
}
