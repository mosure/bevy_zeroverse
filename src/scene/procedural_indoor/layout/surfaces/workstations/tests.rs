use super::*;
use crate::scene::procedural_indoor::{
    layout::{assemblies::AssemblyRole, equipment::cable_route_clear},
    objects::{desktop::ports, equipment::wires, tables},
    validation::validate_layout,
};

#[test]
fn installations_have_usable_operator_positions_and_matching_devices() {
    let (mut raised, mut shared, mut opposed_screens, mut dual, mut portrait) = (0, 0, 0, 0, 0);
    let mut cases = [0; 2];
    let mut case_angles = [0; 2];
    let mut inputs = [0; 2];
    for profile in [IndoorLayout::OpenOffice, IndoorLayout::Coworking] {
        for seed in 81500..81512 {
            let s = IndoorManifest::generate_with_humans(seed, profile, 0.65, 0, 0.25).unwrap();
            validate_layout(&s).unwrap_or_else(|e| panic!("{profile:?} {seed}: {e}"));
            for g in &s.assembly_diagnostics.accepted {
                let desk = &s.objects[g.surface];
                if g.role == AssemblyRole::StandingWorkstation {
                    assert!((0.94..1.19).contains(&desk.size.y));
                    assert!(tables::parameters(desk).workstation.is_some());
                    assert!(!s
                        .objects
                        .iter()
                        .any(|o| o.kind == ObjectKind::Chair
                            && o.interaction_target == Some(desk.id)));
                    raised += 1;
                }
                if g.role != AssemblyRole::Workstation || g.seats.len() != 2 {
                    continue;
                }
                let forward = desk.transform().rotation * Vec3::Z;
                let sides = g
                    .seats
                    .iter()
                    .map(|&i| (s.objects[i].position - desk.position).dot(forward))
                    .collect::<Vec<_>>();
                assert!(sides[0] * sides[1] < 0.);
                opposed_screens += usize::from(g.seats.iter().all(|&i| {
                    s.objects.iter().any(|o| {
                        matches!(o.kind, ObjectKind::Monitor | ObjectKind::Laptop)
                            && o.support == Some(desk.id)
                            && o.interaction_target == Some(i)
                    })
                }));
                shared += 1;
            }
            for o in s.objects.iter().filter(|o| !o.neighbor) {
                match o.kind {
                    ObjectKind::PcTower => {
                        let monitor = &s.objects[o.interaction_target.unwrap()];
                        assert_eq!(monitor.kind, ObjectKind::Monitor);
                        let desk = &s.objects[monitor.support.unwrap()];
                        let angle = (o.yaw - desk.yaw).sin().abs().asin().to_degrees();
                        case_angles[0] += usize::from(angle < 4.);
                        case_angles[1] += usize::from(angle > 9.);
                        if let Some(id) = o.support {
                            assert_eq!(monitor.support, Some(id));
                            assert!(
                                (o.position.y - s.objects[id].position.y - s.objects[id].size.y)
                                    .abs()
                                    < 0.002
                            );
                        } else {
                            assert!((o.position.y - s.floor_height(o.position.xz())).abs() < 0.002);
                        }
                        cases[usize::from(o.support.is_some())] += 1;
                    }
                    ObjectKind::Keyboard | ObjectKind::Mouse => {
                        // Other desks, including public-room furniture, may also carry loose props.
                        let Some(screen) = o.interaction_target.map(|id| &s.objects[id]) else {
                            continue;
                        };
                        assert_eq!(screen.kind, ObjectKind::Monitor);
                        assert_eq!(screen.support, o.support);
                        inputs[usize::from(o.kind == ObjectKind::Mouse)] += 1;
                    }
                    ObjectKind::Monitor => {
                        portrait += usize::from(o.size.y > o.size.x * 1.2);
                        dual += usize::from(s.objects.iter().any(|p| {
                            p.id < o.id
                                && p.kind == o.kind
                                && p.support == o.support
                                && p.interaction_target == o.interaction_target
                        }));
                    }
                    _ => {}
                }
            }
        }
    }
    assert!(raised >= 6 && shared >= 3 && opposed_screens > 0 && dual >= 2 && portrait > 0,
        "standing={raised}, shared={shared}, equipped on both sides={opposed_screens}, dual={dual}, portrait={portrait}");
    assert!(case_angles.into_iter().all(|n| n > 1), "{case_angles:?}");
    assert!(
        cases.iter().all(|&n| n >= 3) && inputs.iter().all(|&n| n >= 12),
        "{cases:?} {inputs:?}"
    );
}

#[test]
fn installed_data_leads_replay_at_their_sockets_and_reject_edited_connections() {
    let mut functions = [0; 2];
    let (mut hanging, mut desk_level, mut wireless) = (0, 0, 0);
    for seed in 81500..81509 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::OpenOffice, 0.65, 0, 0.25)
                .unwrap();
        for o in scene.objects.iter().filter(|o| !o.neighbor) {
            if matches!(o.kind, ObjectKind::Mouse | ObjectKind::Keyboard)
                && o.interaction_target.is_some()
                && !scene
                    .objects
                    .iter()
                    .any(|c| c.kind == ObjectKind::Cable && c.interaction_target == Some(o.id))
            {
                wireless += 1;
            }
            if o.kind != ObjectKind::Cable
                || !matches!(o.variant, wires::USB_LEAD | wires::DISPLAY_LEAD)
            {
                continue;
            }
            let src = &scene.objects[o.interaction_target.unwrap()];
            let dst = &scene.objects[o.cable_target.unwrap()];
            let [(a, _), (b, _)] = ports::data_ports(src, dst).unwrap();
            let points = o.cable_path.as_ref().unwrap();
            assert!(o.transform().transform_point(points[0]).distance(a) < 0.001);
            assert!(
                o.transform()
                    .transform_point(*points.last().unwrap())
                    .distance(b)
                    < 0.001
            );
            let replay = serde_json::from_slice(&serde_json::to_vec(o).unwrap()).unwrap();
            assert!(cable_route_clear(&scene, &replay));
            let mut invalid = o.clone();
            invalid.variant = 0;
            assert!(!cable_route_clear(&scene, &invalid));
            invalid = o.clone();
            invalid.cable_path.as_mut().unwrap()[0].z += 0.009;
            assert!(!cable_route_clear(&scene, &invalid));
            if dst.support.is_none() {
                assert_eq!(dst.kind, ObjectKind::PcTower);
                assert!(a.y - b.y > 0.35);
                hanging += 1;
            } else {
                desk_level += 1;
            }
            functions[usize::from(o.variant == wires::DISPLAY_LEAD)] += 1;
        }
    }
    assert!(
        functions.iter().all(|&n| n > 0) && hanging > 0 && desk_level > 0 && wireless > 0,
        "{functions:?}, hanging={hanging}, desk={desk_level}, untethered={wireless}"
    );
}
