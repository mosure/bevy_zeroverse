use super::*;
use crate::scene::procedural_indoor::{
    materials::Surface,
    objects::{audit, build_object, part_label, public::volleyball::Net},
    program::court::VolleyballCourt,
    validation::validate_layout,
};
use std::collections::BTreeSet;

#[test]
fn basketball_stations_have_wall_support_opposed_pairs_and_clear_approaches() {
    use crate::scene::procedural_indoor::objects::public::basketball::BasketballGoal;
    let mut counts = BTreeSet::new();
    let mut heights = BTreeSet::new();
    let mut directions = BTreeSet::new();
    let mut tampered = false;
    for seed in 0..192 {
        let room = IndoorManifest::generate_with_humans(seed, IndoorLayout::Gymnasium, 0.65, 0, 0.)
            .unwrap();
        validate_layout(&room).unwrap();
        let c = room.program.as_ref().unwrap().volleyball.unwrap();
        let goals: Vec<_> = room
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::BasketballGoal)
            .collect();
        counts.insert(goals.len());
        for o in &goals {
            assert!(furnishing::gymnasium::valid_goal(&room, o, c));
            let spec = BasketballGoal::for_object(o);
            let rim = o.position.y + spec.rim_height();
            heights.insert(rim < 3.);
            let dir = o.transform().rotation * Vec3::Z;
            directions.insert(dir.x.abs() > 0.99);
            let access = furnishing::gymnasium::goal_access(o);
            assert!(room.functional_programs()[0].reservations.contains(&access));
            assert!(!room.camera_clear(o.position + Vec3::Y * (0.5 * o.size.y)));
            if !tampered {
                let mut bad = room.clone();
                bad.objects[o.id].position += dir * 0.07;
                assert!(bad.validate_public_programs().is_err());
                tampered = true;
            }
        }
        if let [a, b] = goals.as_slice() {
            let axis = a.transform().rotation * Vec3::Z;
            assert!(axis.dot(b.transform().rotation * Vec3::Z) < -0.99999);
            assert!((b.position - a.position).dot(axis.cross(Vec3::Y)).abs() < 0.001);
            assert_eq!(a.finish_seed, b.finish_seed);
            assert_eq!(a.size, b.size);
            assert!((a.position.y - b.position.y).abs() < 1e-6);
        }
    }
    assert_eq!(counts, BTreeSet::from([0, 1, 2]));
    assert_eq!((heights.len(), directions.len()), (2, 2));
}

#[test]
fn gymnasium_courts_replay_across_densities_and_reserve_real_clearance() {
    let mut types = BTreeSet::new();
    let mut nets = BTreeSet::new();
    let mut bench_counts = BTreeSet::new();
    for seed in 58100..58132 {
        let density = [0., 0.65, 1.][seed as usize % 3];
        let s =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Gymnasium, density, 3, 0.25)
                .unwrap();
        validate_layout(&s).unwrap();
        assert!(s.accepts_camera_visibility());
        let program = s.program.as_ref().unwrap();
        let c = program.volleyball.unwrap();
        let (lo, hi) = c.protected_bounds();
        assert!(lo.cmpgt(-s.room_size.xz() * 0.5).all() && hi.cmplt(s.room_size.xz() * 0.5).all());
        let floor = &program.materials[Surface::Floor as usize];
        let finish = floor.flooring.as_ref().unwrap();
        finish.validate().unwrap();
        assert!(
            floor.wood.is_some()
                && (0.057..=0.086).contains(&(floor.period_m / finish.cells[0] as f32))
        );
        assert_eq!(
            serde_json::from_value::<VolleyballCourt>(serde_json::to_value(c).unwrap()).unwrap(),
            c
        );
        types.insert((c.full_size, c.long_x));
        if c.full_size {
            assert_eq!(
                (c.width, c.length, c.attack_distance, c.free_zone),
                (9., 18., 3., 3.)
            );
            nets.insert((c.net_height * 100.).round() as u32);
        }
        assert!(s.minimum_ceiling_height(lo, hi) >= if c.full_size { 7. } else { 5.5 });
        for pos in crate::scene::procedural_indoor::architecture::fixture_positions(&s) {
            let half = crate::scene::procedural_indoor::architecture::fixture_half_extents(&s, pos);
            if let Some(h) = crate::scene::procedural_indoor::program::court::clear_height_at(
                &s,
                (pos - half).xz(),
                (pos + half).xz(),
            ) {
                assert!(
                    pos.y - half.y >= h,
                    "fixture in playing volume, seed {seed}"
                );
            }
        }
        let p = &s.functional_programs()[0];
        assert_eq!(
            (p.capacity_unit, p.accepted_capacity),
            (CapacityUnit::PlayingCourts, 1)
        );
        let route = &p.reservations[0];
        assert_eq!((route.min, route.max), (lo, hi));
        let mut benches = 0;
        for o in s.objects.iter().filter(|o| o.solid && !o.neighbor) {
            let (a, b) = o.bounds();
            if o.kind == ObjectKind::VolleyballNet {
                assert!(furnishing::gymnasium::matches_net(o, c));
                if c.full_size {
                    let net = Net::for_object(o);
                    assert!((9.5..=10.0).contains(&(net.half_web * 2.)));
                }
                let tip = o.transform().transform_point(Vec3::X * (c.width * 0.5));
                assert!(
                    (tip.xz() - c.world(Vec2::new(c.width * 0.5, 0.))).length() < 0.002
                        || (tip.xz() - c.world(Vec2::new(-c.width * 0.5, 0.))).length() < 0.002
                );
            } else {
                assert!(!route.overlaps(a, b));
                benches += usize::from(o.kind == ObjectKind::Bench);
            }
        }
        bench_counts.insert(benches);
    }
    assert_eq!(types.len(), 4);
    assert_eq!(nets, BTreeSet::from([224, 243]));
    assert!(bench_counts.len() >= 3);
}

#[test]
fn gymnasium_rejects_a_displaced_net_and_an_obstruction_on_its_court() {
    let mut s =
        IndoorManifest::generate_with_humans(58101, IndoorLayout::Gymnasium, 0.65, 0, 0.).unwrap();
    s.validate_public_programs().unwrap();
    let index = s
        .objects
        .iter()
        .position(|o| o.kind == ObjectKind::VolleyballNet)
        .unwrap();
    let installed = s.objects[index].clone();
    s.objects[index].position.x += 0.20;
    assert!(s.validate_public_programs().is_err());
    s.objects[index] = installed.clone();
    let mut obstruction = installed;
    obstruction.kind = ObjectKind::Bench;
    obstruction.id = s.objects.len();
    s.objects.push(obstruction);
    assert!(s
        .validate_public_programs()
        .unwrap_err()
        .contains("reserved circulation obstructed"));
}

#[test]
fn volleyball_net_has_actual_open_cells_and_resolved_padded_posts() {
    for seed in 0..20 {
        let o = audit::fixture(ObjectKind::VolleyballNet, seed);
        let spec = Net::for_object(&o);
        let mesh = build_object(&o);
        let stats = audit::check(&o, &mesh).unwrap();
        assert!(stats.triangles < 5200 && stats.batches <= 4);
        assert!(mesh
            .parts
            .keys()
            .all(|(_, label)| part_label(label) == "sports_equipment"));
        let pitch = 2. * spec.half_web / spec.columns as f32;
        assert!((0.097..0.103).contains(&pitch));
        // Project the actual indexed triangles along the normal to the net.
        let hit = |xy: Vec2| {
            mesh.parts.values().any(|g| {
                g.indices.as_chunks::<3>().0.iter().any(|triangle| {
                    let [a, b, c] =
                        triangle.map(|i| Vec3::from_array(g.positions[i as usize]).truncate());
                    let (u, v, w) = (b - a, c - a, xy - a);
                    let det = u.perp_dot(v);
                    if det.abs() < 1e-10 {
                        return false;
                    }
                    let (s, t) = (w.perp_dot(v) / det, u.perp_dot(w) / det);
                    s >= 0. && t >= 0. && s + t <= 1.
                })
            })
        };
        let x = (spec.columns / 2) as f32 * pitch - spec.half_web;
        let y = spec.center_height - 0.12;
        let proposal_hit = |xy: Vec2| {
            let mut covered = false;
            spec.view_boxes(|center, half| {
                covered |= (xy - center.truncate()).abs().cmple(half.truncate()).all();
            });
            covered
        };
        assert!(!proposal_hit(Vec2::new(x + 0.45 * pitch, y)) && !proposal_hit(Vec2::new(x, 0.95)));
        assert!(
            proposal_hit(Vec2::new(x, spec.center_height - 0.025))
                && proposal_hit(Vec2::new(spec.post_center, 0.8))
        );
        assert!(hit(Vec2::new(x + 0.0003, y)));
        assert!(!hit(Vec2::new(x + 0.45 * pitch, y)));
        assert!(hit(Vec2::new(x + 0.45 * pitch, spec.center_height - 0.025)));
        assert!(!hit(Vec2::new(x + 0.45 * pitch, 0.95)));
        assert!(hit(Vec2::new(spec.post_center + 0.015, 0.8)));
    }
}

#[test]
fn gymnasium_hangs_cable_trays_above_the_run_off_on_seed_58100() {
    let s =
        IndoorManifest::generate_with_humans(58100, IndoorLayout::Gymnasium, 0.65, 0, 0.).unwrap();
    let court = s.program.as_ref().unwrap().volleyball.unwrap();
    let r = &s.functional_programs()[0].reservations[0];
    let system = s.ceiling_installation.as_ref().unwrap();
    assert!(system.modules.iter().any(|m| m.system
        == crate::scene::procedural_indoor::architecture::ceiling::CeilingSystem::CableTray));
    assert!(system
        .obstacles()
        .all(|(lo, hi)| !r.overlaps(lo, hi) || lo.y >= court.clear_height()));
    validate_layout(&s).unwrap();
}

#[test]
fn bleacher_installations_face_the_court_and_preserve_sideline_and_portal_clearance() {
    let mut axes = BTreeSet::new();
    let mut clear_sidelines = 0;
    let mut invalid_checked = false;
    for seed in 61120..61160 {
        let s = IndoorManifest::generate_with_humans(seed, IndoorLayout::Gymnasium, 0.65, 0, 0.)
            .unwrap();
        let c = s.program.as_ref().unwrap().volleyball.unwrap();
        let banks: Vec<_> = s
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::BleacherBank)
            .collect();
        assert!(banks.len() <= 2);
        if banks.is_empty() {
            clear_sidelines += 1;
        }
        for o in banks {
            assert!(furnishing::gymnasium::valid_bank(o, c));
            // Fit the interval before sampling. Clamping a sampled length
            // previously concentrated over a quarter of all banks at 8.30 m.
            assert!(o.size.x < 8.30);
            let (lo, hi) = o.bounds();
            assert!(!s.entry_approach_overlaps(lo, hi));
            let access = furnishing::gymnasium::bank_access(o);
            assert!(s.functional_programs()[0].reservations.contains(&access));
            assert!(s
                .objects
                .iter()
                .filter(|p| p.solid && !p.neighbor)
                .all(|p| {
                    let (lo, hi) = p.bounds();
                    !access.overlaps(lo, hi)
                }));

            assert!(!s.functional_programs()[0].reservations[0].overlaps(lo, hi));
            let axis = if c.long_x { 1 } else { 0 };
            axes.insert((c.long_x, o.position.xz()[axis] > c.center[axis]));
            if !invalid_checked {
                let mut bad = s.clone();
                bad.objects[o.id].yaw += std::f32::consts::PI;
                assert!(bad
                    .validate_public_programs()
                    .unwrap_err()
                    .contains("spectator bank"));
                bad.objects[o.id] = o.clone();
                bad.objects[o.id].position.x = c.center.x;
                bad.objects[o.id].position.z = c.center.y;
                assert!(bad.validate_public_programs().is_err());
                invalid_checked = true;
            }
        }
    }
    assert!(invalid_checked && clear_sidelines > 0);
    assert_eq!(axes.len(), 4);
}

#[test]
fn gymnasium_free_throw_marks_follow_real_boards_and_survive_archive_replay() {
    use crate::scene::procedural_indoor::{
        objects::public::basketball::BasketballGoal,
        program::{court::FreeThrowMark, IndoorProgram},
    };
    let mut forms = BTreeSet::new();
    let mut directions = BTreeSet::new();
    let mut heights = BTreeSet::new();
    let mut unpainted_goals = 0;
    let mut corrupted = false;
    for seed in 72100..72210 {
        let s = IndoorManifest::generate_with_humans(seed, IndoorLayout::Gymnasium, 0.65, 0, 0.)
            .unwrap();
        let p = s.program.as_ref().unwrap();
        let marks = &p.free_throw_marks;
        let goals = s
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::BasketballGoal && !o.neighbor)
            .count();
        unpainted_goals += goals - marks.len();
        validate_layout(&s).unwrap();
        for m in marks {
            let goal = &s.objects[m.goal_id];
            let kit = BasketballGoal::for_object(goal);
            let board = goal
                .transform()
                .transform_point(Vec3::Z * kit.face_z(goal.size))
                .xz();
            let far = m.center + m.forward * 0.025;
            assert!(far.abs_diff_eq(board + m.forward * m.backboard_distance_m, 0.00005));
            let adult = goal.position.y + kit.rim_height() > 3.;
            heights.insert(adult);
            assert_eq!(m.backboard_distance_m == 4.60, adult);
            forms.insert((m.semicircle, m.line_length_m == 3.60));
            directions.insert((m.forward.x.abs() as u32, m.forward.y.abs() as u32));
        }
        if !marks.is_empty() && !corrupted {
            let json = serde_json::to_value(p).unwrap();
            let replay: IndoorProgram = serde_json::from_value(json.clone()).unwrap();
            assert_eq!(replay.free_throw_marks, *marks);
            let mut legacy = json;
            legacy.as_object_mut().unwrap().remove("free_throw_marks");
            assert!(serde_json::from_value::<IndoorProgram>(legacy)
                .unwrap()
                .free_throw_marks
                .is_empty());
            let mut bad = marks[0];
            bad.center.y += 0.1;
            assert!(FreeThrowMark::validate_set(&s, &[bad]).is_err());
            bad = marks[0];
            bad.goal_id = s.objects.len();
            assert!(FreeThrowMark::validate_set(&s, &[bad]).is_err());
            assert!(FreeThrowMark::validate_set(&s, &[marks[0], marks[0]]).is_err());
            let mut old = s.clone();
            old.generator_version = 319;
            old.program.as_mut().unwrap().free_throw_marks.clear();
            FreeThrowMark::install(&mut old);
            assert!(old.program.as_ref().unwrap().free_throw_marks.is_empty());
            corrupted = true;
        }
    }
    assert_eq!(
        forms,
        BTreeSet::from([(false, false), (false, true), (true, true)])
    );
    assert_eq!(heights.len(), 2);
    assert_eq!(directions.len(), 2);
    assert!(unpainted_goals > 5 && corrupted);
}
