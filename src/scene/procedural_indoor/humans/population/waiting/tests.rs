use super::*;
use crate::scene::procedural_indoor::{geometry::Geometry, humans, objects};

#[test]
fn waiting_passengers_preserve_the_population_and_replay_the_selected_place() {
    let mut occupied = [0; 2];
    let mut stand = 0;
    for seed in 70800..70848 {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::ElevatorLanding, 0.65, 0, 0.)
                .unwrap();
        scene.human_density = 1.;
        let mut reference = scene.clone();
        reference.generator_version = VERSION - 1;
        populate(&mut reference, 1.);
        populate(&mut scene, 1.);
        assert_eq!(reference.humans.len(), scene.humans.len());
        let mut repeat = reference.clone();
        repeat.humans.clear();
        repeat.generator_version = VERSION;
        populate(&mut repeat, 1.);
        assert_eq!(scene.humans, repeat.humans);
        for (h, previous) in scene.humans.iter().zip(&reference.humans) {
            assert_eq!(
                (h.id, h.seed, &h.appearance),
                (previous.id, previous.seed, &previous.appearance)
            );
            let Some(place) = h.waiting_seat() else {
                assert_eq!(h, previous);
                stand += usize::from(!h.pose.seated());
                continue;
            };
            let bank = &scene.objects[h.chair.unwrap()];
            assert!(!previous.pose.seated());
            assert!(valid(&scene, bank, h, place));
            occupied[usize::from(bank.kind == ObjectKind::BeamSeating)] += 1;
            let mut wrong = h.clone();
            wrong.yaw += PI;
            assert!(!valid(&scene, bank, &wrong, place));
            assert!(!valid(
                &scene,
                bank,
                h,
                poses::WaitingSeat {
                    seat_height_m: place.seat_height_m + 0.01,
                    ..place
                }
            ));
            assert!(!valid(
                &scene,
                bank,
                h,
                poses::WaitingSeat {
                    bay: usize::MAX,
                    ..place
                }
            ));
            if bank.kind == ObjectKind::BeamSeating {
                if let Some(bay) = BeamSeating::for_object(bank).table_bay {
                    assert!(!valid(&scene, bank, h, poses::WaitingSeat { bay, ..place }));
                }
            }
            let standing = humans::standing_at(h, Vec3::ZERO, 0., false);
            assert!(standing.chair.is_none() && standing.waiting_seat().is_none());
            let archived: IndoorHuman =
                serde_json::from_value(serde_json::to_value(h).unwrap()).unwrap();
            assert_eq!(archived.waiting_seat(), Some(place));
        }
        humans::validate(&scene).unwrap();
    }
    assert!(
        occupied.into_iter().all(|n| n >= 3) && stand >= 12,
        "{occupied:?}, {stand}"
    );
}

// Intersect the rendered trousers and both seat types independently of the
// pose solver, including skinning, shoe grounding, cushion shape and yaw.
fn vertical_hits(g: &Geometry, xz: Vec2) -> impl Iterator<Item = f32> + '_ {
    g.indices.as_chunks::<3>().0.iter().filter_map(move |tri| {
        let [a, b, c] = tri.map(|i| Vec3::from(g.positions[i as usize]));
        let ab = b.xz() - a.xz();
        let ac = c.xz() - a.xz();
        let q = xz - a.xz();
        let det = ab.perp_dot(ac);
        let u = q.perp_dot(ac) / det;
        let v = ab.perp_dot(q) / det;
        (u >= -1e-5 && v >= -1e-5 && u + v <= 1.00001)
            .then_some(a.y + u * (b.y - a.y) + v * (c.y - a.y))
    })
}

#[test]
fn waiting_passengers_bear_on_the_rendered_seat_with_grounded_shoes() {
    let mut count = 0;
    let mut materials = std::collections::BTreeSet::new();
    let mut contacts = Vec::new();
    for seed in 70800..70960 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::ElevatorLanding, 0.65, 0, 1.)
                .unwrap();
        for h in &scene.humans {
            let Some(place) = h.waiting_seat() else {
                continue;
            };
            let bank = &scene.objects[h.chair.unwrap()];
            let material = if bank.kind == ObjectKind::BeamSeating {
                BeamSeating::for_object(bank).facing
            } else {
                banquette::parameters(bank).upholstery
            };
            if count >= 32 && materials.contains(&(bank.kind == ObjectKind::BeamSeating, material))
            {
                continue;
            }
            let mesh = humans::build_human(h);
            let seat = objects::build_manifest_object(&scene, bank);
            let local =
                bank.transform().compute_affine().inverse() * h.transform().compute_affine();
            let mut closest = f32::INFINITY;
            for x in [-0.10, -0.05, 0., 0.05, 0.10] {
                for z in [-0.11, -0.05, 0., 0.05, 0.09] {
                    let q = h.joints[0].xz() + Vec2::new(x, z);
                    let bank_xz = local.transform_point3(Vec3::new(q.x, 0., q.y)).xz();
                    let top = seat
                        .parts
                        .values()
                        .flat_map(|g| vertical_hits(g, bank_xz))
                        .filter(|y| (y - place.seat_height_m).abs() < 0.045)
                        .max_by(f32::total_cmp);
                    let pants = vertical_hits(&mesh.parts[&HumanSurface::Trousers], q)
                        .filter(|y| *y > place.seat_height_m - 0.20)
                        .min_by(f32::total_cmp);
                    if let Some((pants, top)) = pants.zip(top) {
                        closest = closest.min(pants - top);
                    }
                }
            }
            let sole = mesh.parts[&HumanSurface::Sole]
                .positions
                .iter()
                .map(|p| p[1])
                .fold(f32::INFINITY, f32::min);
            assert!(sole.abs() < 0.003, "{seed} sole {sole}");
            for (hip, knee, ankle) in [(13, 14, 15), (17, 18, 19)] {
                let j = &mesh.local_joints;
                let original = &h.joints;
                for (a, b) in [(hip, knee), (knee, ankle)] {
                    assert!(
                        (j[a].distance(j[b]) - original[a].distance(original[b])).abs() < 0.001
                    );
                }
                assert!(j[ankle].xz().distance(original[ankle].xz()) < 0.001);
            }
            if bank.kind == ObjectKind::BeamSeating {
                let stock = BeamSeating::for_object(bank);
                let arms = [place.bay, place.bay + 1].map(|index| stock.arm_x(bank, index));
                let y = place.seat_height_m + 0.193;
                for point in mesh.parts.values().flat_map(|g| &g.positions) {
                    let p = local.transform_point3(Vec3::from(*point));
                    if (-0.058..bank.size.z * 0.24).contains(&p.z) {
                        assert!(
                            arms.iter()
                                .flatten()
                                .all(|x| (p.x - x).hypot(p.y - y) > 0.010),
                            "seed {seed} body penetrates an occupied armrest"
                        );
                    }
                }
            }
            contacts.push((seed, bank.kind, material, h.stature, h.build, closest));
            materials.insert((bank.kind == ObjectKind::BeamSeating, material));
            count += 1;
        }
        if count >= 32 && materials.len() >= 5 {
            break;
        }
    }
    eprintln!(
        "waiting mesh contact: {} passengers, range {:.2} to {:.2} mm",
        contacts.len(),
        contacts.iter().map(|c| c.5).fold(f32::INFINITY, f32::min) * 1000.,
        contacts
            .iter()
            .map(|c| c.5)
            .fold(f32::NEG_INFINITY, f32::max)
            * 1000.
    );
    assert!(
        contacts.iter().all(|c| (-0.025..=0.025).contains(&c.5)),
        "{contacts:#?}"
    );
    assert!(count >= 32 && materials.len() >= 5, "{count} {materials:?}");
}

#[cfg(feature = "human_motion")]
#[test]
fn waiting_passengers_use_their_own_bay_for_motion_waypoints() {
    use crate::human_motion::{planning, HumanMotionConfig, HumanTrajectory};
    let mut motions = 0;
    for seed in 70800..70832 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::ElevatorLanding, 0.65, 0, 1.)
                .unwrap();
        crate::scene::procedural_indoor::validation::validate_layout(&scene).unwrap();
        let mut config = HumanMotionConfig {
            fraction: 1.,
            ..Default::default()
        };
        for plan in planning::plan(&scene, &config).unwrap().0 {
            let h = scene.humans.iter().find(|h| h.id == plan.actor_id).unwrap();
            if h.waiting_seat().is_none() {
                continue;
            }
            assert_eq!(plan.support_chair, h.chair);
            assert!(plan.request.waypoints.iter().any(|w| w
                .position
                .xz()
                .distance(h.position.xz())
                < 0.001));
            config.trajectories = vec![HumanTrajectory {
                actor_id: h.id,
                prompt: plan.request.prompt,
                waypoints: plan.request.waypoints,
                support_chair: plan.support_chair,
            }];
            config.fraction = 0.;
            let (explicit, _) = planning::plan(&scene, &config).unwrap();
            assert_eq!(explicit[0].support_chair, h.chair);
            motions += 1;
        }
        if motions >= 3 {
            break;
        }
    }
    assert!(motions >= 3);
}
