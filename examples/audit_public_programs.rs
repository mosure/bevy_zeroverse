//! Local, source-bound qualification. Consecutive requests; failures stay in the denominator.
use bevy_zeroverse::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest, ObjectKind},
    validation::{validate_geometry, validate_layout},
};
use clap::Parser;
#[path = "audit_public_programs/presenter.rs"]
mod presenter;
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
        let mut floor_guides = Vec::new();
        let mut courtyards = Vec::new();
        let mut courts = Vec::new();
        let mut workouts = Vec::new();
        let mut dining = Vec::new();
        let mut hydration = Vec::new();
        let mut fountains = Vec::new();
        let mut baskets = Vec::new();
        let mut laundries = Vec::new();
        let mut laundry_carts = Vec::new();
        let mut transit_seating = Vec::new();
        let mut changing_rooms = Vec::new();
        let mut markets = Vec::new();
        let mut checkouts = Vec::new();
        let mut apparel = Vec::new();
        let mut sculptures = Vec::new();
        let mut libraries = Vec::new();
        let mut classrooms = Vec::new();
        let mut bakery_cases = Vec::new();
        let mut lecterns = Vec::new();
        let mut ensembles = Vec::new();
        let mut workshops = Vec::new();
        let mut studios = Vec::new();
        let mut receptions = Vec::new();
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
                    if o.kind == ObjectKind::DisplayCase && o.variant < 16 {
                        let cabinet=bevy_zeroverse::scene::procedural_indoor::objects::public::casework::parameters(o);
                        bakery_cases.push(serde_json::json!({"seed":seed,"object":o,"batches":cabinet.bakery_batches(o,s.generator_version)}));
                    }
                    if o.kind == ObjectKind::Lectern {
                        lecterns.push(presenter::observe(&s, o));
                    }
                    if o.kind == ObjectKind::InformationStand {
                        use bevy_zeroverse::scene::procedural_indoor::objects::{
                            audit, public::signage::floor_guide::FloorGuide,
                        };
                        if FloorGuide::for_object(o, s.generator_version).is_some() {
                            let stock = audit::program(o);
                            floor_guides.push(serde_json::json!({"seed":seed,"object":o,"sheet":stock["floor_guide"],
                                "support_style":stock["support"]}));
                        }
                    }
                    if o.kind == ObjectKind::Sculpture {
                        let stock = bevy_zeroverse::scene::procedural_indoor::objects::public::sculpture::parameters_at(o, s.generator_version);
                        sculptures.push(
                            serde_json::json!({"seed":seed,"id":o.id,"support":o.support,
                            "form":stock.form,"surface":stock.surface,"carving":stock.carving}),
                        );
                    }
                    if o.kind == ObjectKind::BasketStack {
                        use bevy_zeroverse::scene::procedural_indoor::objects::public::baskets::BasketStack;
                        baskets.push(serde_json::json!({"seed":seed,"object":o,"stock":BasketStack::for_object(o)}));
                    }
                    if o.kind == ObjectKind::LaundryCart {
                        use bevy_zeroverse::scene::procedural_indoor::objects::public::laundry::cart::LaundryCart;
                        laundry_carts.push(serde_json::json!({"seed":seed,"object":o,"stock":LaundryCart::for_object(o),
                            "station":o.interaction_target.map(|id|s.objects[id].kind)}));
                    }
                    if o.kind == ObjectKind::DrinkingFountain {
                        use bevy_zeroverse::scene::procedural_indoor::objects::public::fountain::Fountain;
                        let views: Vec<_> = s.cameras.iter().map(|c| {
                            let pose = c.transform_at(0.);
                            serde_json::json!({"eye":pose.translation,"forward":*pose.forward(),"fisheye":c.fisheye.is_some()})
                        }).collect();
                        fountains.push(serde_json::json!({"seed":seed,"object":o,"stock":Fountain::for_object(o),"initial_views":views}));
                    }
                    if o.kind == ObjectKind::WaterCooler {
                        let zone = s
                            .assembly_diagnostics
                            .programs
                            .iter()
                            .find(|p| p.members.contains(&o.id));
                        let stock=bevy_zeroverse::scene::procedural_indoor::objects::public::cooler::Cooler::for_object(o);
                        hydration.push(serde_json::json!({"seed":seed,"activity":zone.map(|p|p.activity),"object":o,"stock":stock}));
                    }
                }
                if s.layout == IndoorLayout::LockerRoom {
                    use bevy_zeroverse::scene::procedural_indoor::{
                        materials::Surface,
                        objects::{public::education, services::bench},
                    };
                    let program = &s.functional_programs()[0];
                    let members = || program.members.iter().map(|&id| &s.objects[id]);
                    let banks: Vec<_> = members().filter(|o| o.kind == ObjectKind::Locker).map(|o| {
                        let stock = education::parameters(o);
                        let doors = stock.compartments();
                        serde_json::json!({"id":o.id,"installation":education::installation(o),"peer":o.interaction_target,
                            "compartments":doors,"yaw":o.yaw,"size_m":o.size,"finish":stock.finish})
                    }).collect();
                    let benches: Vec<_> = members().filter(|o| o.kind == ObjectKind::Bench).map(|o| {
                        let b = bench::parameters(o);
                        serde_json::json!({"bank":o.interaction_target,"size_m":o.size,"slatted":b.slats,"stock":b.seat,
                            "end_towel":members().any(|t| t.kind == ObjectKind::KitchenTowel && t.support == Some(o.id)),
                            "bag":members().any(|t| t.kind == ObjectKind::Bag && t.support == Some(o.id))})
                    }).collect();
                    let floor = &s.program.as_ref().unwrap().materials[Surface::Floor as usize];
                    changing_rooms.push(serde_json::json!({"seed":seed,"target_banks":program.requested_capacity,"banks":banks,
                        "benches":benches,"aisle_m":program.parameters.aisle_metres,"sill_m":s.window_sill,
                        "floor_layout":floor.flooring.as_ref().unwrap().layout,"floor_roughness":floor.roughness}));
                }
                for p in s
                    .functional_programs()
                    .iter()
                    .filter(|p| p.activity == IndoorLayout::Laundromat)
                {
                    use bevy_zeroverse::scene::procedural_indoor::objects::public::laundry::{
                        FoldingTable, Machine,
                    };
                    let members = || p.members.iter().map(|&id| &s.objects[id]);
                    let cabinets: Vec<_> = members()
                        .filter(|o| {
                            matches!(o.kind, ObjectKind::WashingMachine | ObjectKind::TumbleDryer)
                        })
                        .map(|o| serde_json::json!({"object":o,"machine":Machine::for_object(o)}))
                        .collect();
                    let tables: Vec<_> = members()
                        .filter(|o| o.kind == ObjectKind::LaundryFoldingTable)
                        .map(|o| serde_json::json!({"object":o,"table":FoldingTable::for_object(o),
                            "towels":members().filter(|t| t.kind == ObjectKind::KitchenTowel && t.support == Some(o.id)).count()}))
                        .collect();
                    laundries.push(serde_json::json!({"seed":seed,"zone":p.zone,"requested_cabinets":p.requested_capacity,
                        "cabinets":cabinets,"folding_tables":tables,"reservations":p.reservations}));
                }
                for p in s
                    .functional_programs()
                    .iter()
                    .filter(|p| p.activity == IndoorLayout::Classroom)
                {
                    use bevy_zeroverse::scene::procedural_indoor::objects::tables::{
                        classroom_mates, classroom_table,
                    };
                    let members = || p.members.iter().map(|&i| &s.objects[i]);
                    let tables: Vec<_> = members().filter_map(|table| {
                        let stock = classroom_table(table)?;
                        let mate = members().find(|other| classroom_mates(table, other)).map(|o| o.id);
                        let seats: Vec<_> = members().filter(|o|o.kind==ObjectKind::Chair && o.interaction_target==Some(table.id))
                            .map(|o| serde_json::json!({"id":o.id,"position_m":o.position,"yaw_rad":o.yaw})).collect();
                        let contents: Vec<_> = s.objects.iter().filter(|o|o.support==Some(table.id))
                            .map(|o| serde_json::json!({"item":o.kind,"position_m":o.position,"size_m":o.size,"yaw_rad":o.yaw})).collect();
                        Some(serde_json::json!({"id":table.id,"mate":mate,"position_m":table.position,"size_m":table.size,
                            "yaw_rad":table.yaw,"stock":stock,"seats":seats,"contents":contents}))
                    }).collect();
                    classrooms.push(serde_json::json!({"seed":seed,"zone":p.zone,"min_m":p.min,"max_m":p.max,
                        "aisle_m":p.parameters.aisle_metres,"tables":tables,
                        "individual_desks":members().filter(|o|o.kind==ObjectKind::Desk).count(),
                        "tablet_seats":members().filter(|o|o.kind==ObjectKind::Chair && o.variant==10).count()}));
                }
                for p in s
                    .functional_programs()
                    .iter()
                    .filter(|p| p.activity == IndoorLayout::RehearsalRoom)
                {
                    use bevy_zeroverse::scene::procedural_indoor::objects::public::music;
                    let stations: Vec<_> = p.members.iter().map(|&i| &s.objects[i]).filter_map(|o| {
                        let construction = match o.kind {
                            ObjectKind::MusicStand => serde_json::json!(music::Stand::sample(o)),
                            ObjectKind::StagePiano => serde_json::json!(music::Keyboard::sample(o)),
                            _ => return None,
                        };
                        let chair = &s.objects[o.interaction_target?];
                        Some(serde_json::json!({"kind":o.kind,"id":o.id,"position_m":o.position,"yaw_rad":o.yaw,
                            "player":chair.id,"player_position_m":chair.position,"player_yaw_rad":chair.yaw,"construction":construction}))
                    }).collect();
                    let percussion: Vec<_> = p.members.iter().map(|&i| &s.objects[i])
                        .filter(|o| o.kind == ObjectKind::Timpani)
                        .map(|o| serde_json::json!({"id":o.id,"position_m":o.position,"yaw_rad":o.yaw,"size_m":o.size,"finish_seed":o.finish_seed,"construction":music::Timpani::sample(o)})).collect();
                    ensembles.push(serde_json::json!({"seed":seed,"seats":p.accepted_capacity,"focus_m":p.focus,"reservations":p.reservations,"stations":stations,"percussion":percussion}));
                }
                if s.layout == IndoorLayout::Workshop {
                    use bevy_zeroverse::scene::procedural_indoor::objects::public::workbench::Bench;
                    let stations: Vec<_> = s.objects.iter().filter(|o| o.kind == ObjectKind::Workbench && !o.neighbor).map(|bench| {
                        let contents: Vec<_> = s.objects.iter().filter(|o| o.support == Some(bench.id)).map(|o| serde_json::json!({"id":o.id,"item":o.kind,"contact":s.support_plane(o),"size_m":o.size,"yaw_rad":o.yaw})).collect();
                        serde_json::json!({"id":bench.id,"position_m":bench.position,"size_m":bench.size,"yaw_rad":bench.yaw,"bench":Bench::sample(bench),"contents":contents})
                    }).collect();
                    workshops.push(serde_json::json!({"seed":seed,"stations":stations}));
                }
                if s.layout == IndoorLayout::Studio {
                    use bevy_zeroverse::scene::procedural_indoor::objects::public::easel::Easel;
                    let stations:Vec<_>=s.objects.iter().filter(|o|o.kind==ObjectKind::ArtistEasel && !o.neighbor)
                        .map(|o|serde_json::json!({"id":o.id,"size_m":o.size,"position_m":o.position,"yaw_rad":o.yaw,"subject":o.interaction_target,"easel":Easel::sample(o)})).collect();
                    studios.push(serde_json::json!({"seed":seed,"stations":stations}));
                }
                if s.layout == IndoorLayout::Reception {
                    use bevy_zeroverse::scene::procedural_indoor::objects::public::reception::Counter;
                    let station = s.objects.iter().find(|o| !o.neighbor && o.kind == ObjectKind::ReceptionDesk).map(|o| {
                        let children: Vec<_> = s.objects.iter().filter(|item| item.support == Some(o.id))
                            .map(|item| (item.kind, s.support_plane(item))).collect();
                        let staff_chair = s.objects.iter().find(|seat| seat.kind == ObjectKind::Chair && seat.interaction_target == Some(o.id));
                        serde_json::json!({"millwork":Counter::sample(o),"size_m":o.size,"position_m":o.position,"yaw_rad":o.yaw,
                            "supported_items":children,"staff_chair":staff_chair.map(|seat| seat.id),
                            "staff_seated":staff_chair.is_some_and(|seat| s.humans.iter().any(|h| h.chair == Some(seat.id)))})
                    });
                    receptions.push(serde_json::json!({"seed": seed, "station": station}));
                }
                if s.layout == IndoorLayout::Library {
                    use bevy_zeroverse::scene::procedural_indoor::objects::storage::library_stack::Stack;
                    let stacks: Vec<_> = s.objects.iter().filter(|o| !o.neighbor && o.kind == ObjectKind::LibraryStack)
                        .map(|o| {
                            let shelving = Stack::sample(o);
                            serde_json::json!({"object":o.id, "size_m":o.size, "yaw_rad":o.yaw,
                                "featured_books":shelving.featured_books(o,s.generator_version), "shelving":shelving})
                        }).collect();
                    let seats = s
                        .objects
                        .iter()
                        .filter(|o| {
                            !o.neighbor
                                && matches!(
                                    o.kind,
                                    ObjectKind::Chair | ObjectKind::Sofa | ObjectKind::Bench
                                )
                        })
                        .count();
                    libraries.push(serde_json::json!({"seed": seed, "stacks": stacks, "seating_objects":seats}));
                }
                capacities.extend(s.functional_programs().iter().map(|p| p.accepted_capacity));
                parameters.extend(s.functional_programs().iter().map(|p| serde_json::json!({"seed":seed,"activity":p.activity,"parameters":p.parameters,"requested_capacity":p.requested_capacity,"accepted_capacity":p.accepted_capacity,"capacity_unit":p.capacity_unit,"rejected_candidates":p.rejected_candidates,"reservations":p.reservations.len(),"services":p.services.len()})));
                for (zone, p) in s
                    .functional_programs()
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.activity == IndoorLayout::GroceryStore)
                {
                    use bevy_zeroverse::scene::procedural_indoor::objects::public::{
                        gondola_stock, produce_stand, refrigerated_case,
                    };
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
                    let produce_stands: Vec<_> = stock
                        .iter()
                        .filter_map(|o| {
                            produce_stand(o, s.generator_version).map(|stand| serde_json::json!({
                            "id":o.id,"position":o.position,"yaw":o.yaw,"size":o.size,"stand":stand
                        }))
                        })
                        .collect();
                    let shelves: Vec<_> = stock
                        .iter()
                        .filter_map(|o| {
                            gondola_stock(o, s.generator_version).map(|shelves| {
                                serde_json::json!({
                                    "id":o.id,"merchandise":shelves
                                })
                            })
                        })
                        .collect();
                    markets.push(serde_json::json!({"seed":seed,"zone":zone,"produce":produce,"produce_stands":produce_stands,"ambient":stock.len()-produce-cases.len(),"gondolas":shelves,"refrigerated":cases}));
                }
                for (zone, p) in s
                    .functional_programs()
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.activity == IndoorLayout::ClothingStore)
                {
                    use bevy_zeroverse::scene::procedural_indoor::objects::public::{
                        apparel_rail, fitting_room, folded_apparel,
                    };
                    let rails: Vec<_> = p.members.iter().filter_map(|&id| {
                        let o = &s.objects[id];
                        let rail = apparel_rail(o, s.generator_version)?;
                        Some(serde_json::json!({"id": id, "yaw": o.yaw, "form": rail.form, "arms": rail.arms.len(), "collections": rail.collections.len(), "hanger_positions": rail.hangers}))
                    }).collect();
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
                    let folded_tables: Vec<_> = p
                        .members
                        .iter()
                        .filter_map(|&id| {
                            folded_apparel(&s.objects[id], s.generator_version)
                                .map(|stock| serde_json::json!({"id":id,"stock":stock}))
                        })
                        .collect();
                    apparel.push(serde_json::json!({"seed":seed,"zone":zone,"merchandise":p.accepted_capacity,"fitting_rooms":booths,"rails":rails,"folded_tables":folded_tables,"initial_camera_poses":cameras}));
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
                    let goals: Vec<_> = s.objects.iter()
                        .filter(|o| !o.neighbor && o.kind == ObjectKind::BasketballGoal)
                        .map(|o| {
                            let goal = bevy_zeroverse::scene::procedural_indoor::objects::public::basketball::BasketballGoal::for_object(o);
                            serde_json::json!({"id":o.id,"position":o.position,"yaw":o.yaw,"size":o.size,"rim_height_m":o.position.y+goal.rim_height(),"construction":goal})
                        }).collect();
                    courts.push(serde_json::json!({"seed":seed,"court":court,"room_size":s.room_size,"nets":nets,"sideline_benches":benches,"bleachers":stands,"basketball_goals":goals,"free_throw_marks":s.program.as_ref().unwrap().free_throw_marks}));
                }
                if s.layout.is_exercise() {
                    let p = &s.functional_programs()[0];
                    let count = |kind| {
                        p.members
                            .iter()
                            .filter(|&&id| s.objects[id].kind == kind)
                            .count()
                    };
                    let instructor = p.class_instructor(&s.objects);
                    let practice_props: Vec<_> = p.members.iter().map(|&i| &s.objects[i])
                        .filter(|o| matches!(o.kind, ObjectKind::YogaBlock | ObjectKind::YogaBolster))
                        .map(|o| serde_json::json!({"object":o,
                            "construction":bevy_zeroverse::scene::procedural_indoor::objects::audit::program(o)})).collect();
                    let mut mat_programs = BTreeMap::<String, usize>::new();
                    let mut folding = 0usize;
                    for mat in p
                        .members
                        .iter()
                        .map(|&i| &s.objects[i])
                        .filter(|o| o.kind == ObjectKind::ExerciseMat)
                    {
                        let product = bevy_zeroverse::scene::procedural_indoor::objects::public::mats::PracticeMat::for_object(mat);
                        *mat_programs
                            .entry(format!("{:?}", product.facing))
                            .or_default() += 1;
                        folding += usize::from(product.panels > 1);
                    }
                    workouts.push(serde_json::json!({
                        "seed": seed, "bikes": count(ObjectKind::StationaryBike),
                        "treadmills": count(ObjectKind::Treadmill), "benches":count(ObjectKind::WeightBench),
                        "mats":count(ObjectKind::ExerciseMat), "folding_mats":folding, "mat_facings":mat_programs, "floor_style":s.floor_style, "mat_racks":count(ObjectKind::ExerciseMatRack), "racks":count(ObjectKind::DumbbellRack),
                        "practice_props":practice_props,
                        "dumbbell_racks":p.members.iter().map(|&i| &s.objects[i]).filter(|o|o.kind==ObjectKind::DumbbellRack).map(|o|serde_json::json!({
                            "object":o, "construction":bevy_zeroverse::scene::procedural_indoor::objects::public::dumbbells::FreeWeightRack::for_object(o)
                        })).collect::<Vec<_>>(),
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
                    let tabletop = || {
                        s.objects
                            .iter()
                            .filter(|o| o.support.is_some_and(|id| ids.contains(&id)))
                    };
                    let place_settings = tabletop()
                        .filter(|o| {
                            o.kind == ObjectKind::Dishware && o.interaction_target.is_some()
                        })
                        .count();
                    let glassware: Vec<_> = tabletop().filter(|o| o.kind == ObjectKind::DrinkingGlass).map(|o| serde_json::json!({
                        "object": o, "glass": bevy_zeroverse::scene::procedural_indoor::objects::audit::program(o)
                    })).collect();
                    let napkins: Vec<_> = tabletop().filter(|o| o.kind == ObjectKind::TableNapkin).map(|o| serde_json::json!({
                        "object": o, "linen": bevy_zeroverse::scene::procedural_indoor::objects::audit::program(o)
                    })).collect();
                    let wells: Vec<_> = p.members.iter().map(|&i| &s.objects[i]).filter(|o|o.kind==ObjectKind::HotFoodWell).map(|o|serde_json::json!({
                        "object":o,"counter":s.objects[o.support.unwrap()], "installation":bevy_zeroverse::scene::procedural_indoor::objects::public::hot_food::HotFoodWell::for_object(o)
                    })).collect();
                    use bevy_zeroverse::scene::procedural_indoor::objects::public::espresso::{
                        Grinder, Machine,
                    };
                    let espresso_bars: Vec<_> = p.members.iter().map(|&i|&s.objects[i]).filter(|o|o.kind==ObjectKind::CoffeeGrinder).map(|g| {
                        let m=&s.objects[g.interaction_target.expect("validated grinder has a brewing station")];
                        let c=&s.objects[g.support.expect("validated grinder has a counter")];
                        let cups=s.objects.iter().filter(|o| o.support==Some(c.id) && o.kind==ObjectKind::CoffeeCup).count();
                        serde_json::json!({"counter": c,"machine":m, "grinder":g,"brewing":Machine::for_object(m),"grinding":Grinder::for_object(g),"pickup_cups":cups})
                    }).collect();
                    dining.push(serde_json::json!({"seed":seed, "zone":zone, "activity":p.activity, "tables":tables.len(), "dressed":dressed,
                        "glassware":glassware, "napkins":napkins, "place_settings":place_settings, "hot_food_wells":wells, "espresso_bars":espresso_bars, "seats":p.accepted_capacity, "served_items":tabletop().count()}));
                }
                for p in s.functional_programs().iter().filter(|p| {
                    matches!(
                        p.activity,
                        IndoorLayout::Hallway | IndoorLayout::ElevatorLanding
                    )
                }) {
                    use bevy_zeroverse::scene::procedural_indoor::objects::public::beam_seating::BeamSeating;
                    for o in p.members.iter().map(|&id| &s.objects[id]).filter(|o| {
                        matches!(o.kind, ObjectKind::BeamSeating | ObjectKind::Banquette)
                    }) {
                        let stock =
                            (o.kind == ObjectKind::BeamSeating).then(|| BeamSeating::for_object(o));
                        let passengers: Vec<_> = s.humans.iter().filter(|h| h.chair == Some(o.id)).map(|h|
                            serde_json::json!({"id":h.id,"seat":h.waiting_seat(),"pose":h.pose,"position_m":h.position,"yaw_radians":h.yaw})).collect();
                        transit_seating.push(serde_json::json!({"seed":seed,"activity":p.activity,"kind":o.kind,"id":o.id,"position":o.position,"yaw_radians":o.yaw,
                            "size_m":o.size,"stock":stock,"seats_on_beam":stock.map(BeamSeating::seat_count),"passengers":passengers}));
                    }
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
                    let garden_beds: Vec<_> = p.members.iter().map(|&i| &s.objects[i])
                        .filter(|o| o.kind == ObjectKind::PlantingBed)
                        .map(|o| serde_json::json!({
                            "object": o, "planting": bevy_zeroverse::scene::procedural_indoor::objects::audit::program(o)
                        })).collect();
                    let meter = s
                        .domain()
                        .and_then(|d| d.photometry.meter)
                        .ok_or("missing courtyard meter")?;
                    courtyards.push(serde_json::json!({
                        "seed": seed, "room_m": s.room_size, "open_area_fraction":roof.open_area_m2()/(s.room_size.x*s.room_size.z),
                        "opening_min":roof.min,"opening_max":roof.max,"slab_m":roof.slab_m,"soffit":roof.soffit,
                        "columns":envelope.pillars.len(),"plant_species":species,"seats":p.accepted_capacity,
                        "planting_beds":garden_beds.len(),"garden_beds":garden_beds,
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
        if !floor_guides.is_empty() {
            summary["floor_guides"] = serde_json::json!(floor_guides);
        }
        if !ensembles.is_empty() {
            summary["ensembles"] = serde_json::json!(ensembles);
        }
        if !classrooms.is_empty() {
            summary["classrooms"] = serde_json::json!(classrooms);
        }
        if !courts.is_empty() {
            summary["gymnasiums"] = serde_json::json!(courts);
        }
        if !workshops.is_empty() {
            summary["workshops"] = serde_json::json!(workshops);
        }
        if !studios.is_empty() {
            summary["studios"] = serde_json::json!(studios);
        }
        if !receptions.is_empty() {
            summary["receptions"] = serde_json::json!(receptions);
        }
        if !libraries.is_empty() {
            summary["libraries"] = serde_json::json!(libraries);
        }
        if !sculptures.is_empty() {
            summary["sculptures"] = serde_json::json!(sculptures);
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
        if !baskets.is_empty() {
            summary["basket_stacks"] = serde_json::json!(baskets);
        }
        if !changing_rooms.is_empty() {
            summary["changing_rooms"] = serde_json::json!(changing_rooms);
        }
        if !transit_seating.is_empty() {
            summary["waiting_seats"] = serde_json::json!(transit_seating);
        }
        if !laundry_carts.is_empty() {
            summary["laundry_carts"] = serde_json::json!(laundry_carts);
        }
        if !laundries.is_empty() {
            summary["laundries"] = serde_json::json!(laundries);
        }
        if !fountains.is_empty() {
            summary["drinking_fountains"] = serde_json::json!(fountains);
        }
        if !bakery_cases.is_empty() {
            summary["bakery_cases"] = serde_json::json!(bakery_cases);
        }
        if !lecterns.is_empty() {
            summary["lecterns"] = serde_json::json!(lecterns);
        }
        if !hydration.is_empty() {
            summary["hydration"] = serde_json::json!(hydration);
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
