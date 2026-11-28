use super::*;
use crate::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest},
    objects::{audit, public::elevator::Levels},
};

#[test]
fn actual_car_service_and_stair_elevations_determine_landing_identity() {
    let mut cases = [0; 3];
    for layout in [IndoorLayout::ElevatorLanding, IndoorLayout::StairHall] {
        for seed in 74000..74032 {
            let s = IndoorManifest::generate_with_humans(seed, layout, 0.65, 0, 0.35).unwrap();
            for p in s.functional_programs() {
                let members = || p.members.iter().map(|&id| &s.objects[id]);
                for o in members().filter(|o| o.kind == ObjectKind::InformationStand) {
                    let g = FloorGuide::for_object(o, s.generator_version).unwrap();
                    match layout {
                        IndoorLayout::ElevatorLanding => {
                            assert_eq!(g.connection, Connection::Lift);
                            let mut cars = 0;
                            for car in members().filter(|o| o.kind == ObjectKind::Elevator) {
                                let landing = Levels::decode(car.variant).unwrap();
                                assert_eq!((g.level, g.highest), (landing.floor, landing.highest));
                                cars += 1;
                            }
                            assert!(cars > 0);
                            cases[0] += 1;
                        }
                        IndoorLayout::StairHall => {
                            let deck = &s
                                .envelope
                                .as_ref()
                                .unwrap()
                                .mezzanine
                                .as_ref()
                                .unwrap()
                                .deck;
                            assert_eq!(g.connection, Connection::Stairs);
                            assert_eq!(g.highest, 1);
                            let h = if g.level == 1 {
                                deck.height
                            } else {
                                s.floor_height(o.position.xz())
                            };
                            assert!((h - o.position.y).abs() < 1e-4);
                            cases[1 + usize::from(g.level)] += 1;
                        }
                        _ => unreachable!(),
                    }
                    let archived: IndoorObject =
                        serde_json::from_slice(&serde_json::to_vec(o).unwrap()).unwrap();
                    assert_eq!(FloorGuide::for_object(&archived, INTRODUCED), Some(g));
                    assert_eq!(FloorGuide::for_object(&archived, INTRODUCED - 1), None);
                }
            }
        }
    }
    assert!(cases.into_iter().all(|n| n > 0), "{cases:?}");
    assert!(FloorGuide::stair(0.5, 0., 2.25).is_none());
}

#[test]
fn edition_keys_roundtrip_and_preserve_ordinary_posters() {
    let mut o = audit::fixture(ObjectKind::InformationStand, 5);
    let variant = o.variant;
    for highest in 1..=29 {
        for level in 0..=highest {
            let g = FloorGuide::lift(level, highest);
            g.identify(&mut o);
            assert_eq!(o.variant & 255, variant & 255);
            assert_eq!(FloorGuide::for_object(&o, INTRODUCED), Some(g));
            for landscape in [false, true] {
                for stock in 0..4 {
                    let e = Edition {
                        guide: g,
                        landscape,
                        stock,
                    };
                    assert_eq!(Edition::from_slot(e.slot()), Some(e));
                }
            }
        }
    }
    for slot in [0, FIRST_SLOT - 1, FIRST_SLOT, FIRST_SLOT + 0x3ff, END_SLOT] {
        assert!(Edition::from_slot(slot).is_none());
    }
    o.variant |= 1 << 21;
    assert!(FloorGuide::for_object(&o, INTRODUCED).is_none());
    let mut poster = Assembly::default();
    super::super::stand(&mut poster, &o, INTRODUCED);
    assert!(poster.parts.keys().any(|(s, _)| *s == Surface::Art));
    assert!(!poster
        .parts
        .keys()
        .any(|(s, _)| *s == Surface::PrintedPaper));
}

#[test]
fn sheets_fill_the_aperture_with_outward_ink_on_both_faces() {
    let mut formats = [0; 2];
    let mut folding = 0;
    for seed in 0..160 {
        let mut o = audit::fixture(ObjectKind::InformationStand, seed);
        FloorGuide::lift(3, 11).identify(&mut o);
        let params = super::super::parameters(&o);
        let sheet = Sheet::for_stand(&o, INTRODUCED, &params).unwrap();
        let w = o.size.x * params.width_fraction;
        let h = o.size.y * params.height_fraction;
        let d = (o.size.z * 0.045).min(o.size.y * 0.018);
        assert!((sheet.size_m.x / sheet.size_m.y - sheet.edition.aspect()).abs() < 1e-5);
        formats[usize::from(sheet.edition.landscape)] += 1;
        let mut panel = Assembly::default();
        sheet.panel(&mut panel, params.finish, "sign", Vec3::new(w, h, d));
        let frame = &panel.parts[&(params.finish, "sign".to_string())];
        for (tri, normal) in frame.indices.as_chunks::<3>().0.iter().map(|tri| {
            let ps = tri.map(|i| Vec3::from_array(frame.positions[i as usize]));
            (ps, Vec3::from_array(frame.normals[tri[0] as usize]))
        }) {
            assert!((tri[1] - tri[0]).cross(tri[2] - tri[0]).dot(normal) > 1e-9);
            if normal.z > 0.01 {
                // Every front-facing frame triangle stays on one side of the aperture.
                let local = tri.map(|v| v.truncate() - sheet.center_m);
                assert!((0..2).any(|axis| [-1., 1.].iter().any(|&s| local
                    .iter()
                    .all(|p| p[axis] * s >= sheet.size_m[axis] * 0.5 - 1e-6))));
            }
        }
        let mut sign = Assembly::default();
        super::super::stand(&mut sign, &o, INTRODUCED);
        let papers: Vec<_> = sign
            .parts
            .iter()
            .filter(|((s, _), _)| *s == Surface::PrintedPaper)
            .collect();
        assert_eq!(papers.len(), 1);
        let ((_, label), mesh) = papers[0];
        assert_eq!(
            label,
            &format!("information_stand#finish{}#uv_local", sheet.edition.slot())
        );
        let sides = if params.support == 1 {
            folding += 1;
            2
        } else {
            1
        };
        assert_eq!(mesh.positions.len(), sides * 4);
        for (side, quad) in mesh.positions.as_chunks::<4>().0.iter().enumerate() {
            let p = quad.map(Vec3::from_array);
            let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize();
            let sign = if sides == 2 && side == 0 { -1. } else { 1. };
            assert!(n.z * sign > 0.7);
            assert!((p[1] - p[0]).x * sign > 0. && p[2].y > p[1].y);
            assert_eq!(
                &mesh.uvs[side * 4..side * 4 + 4],
                &[[0., 1.], [1., 1.], [1., 0.], [0., 0.]]
            );
        }
    }
    assert!(
        folding > 15 && formats.into_iter().all(|n| n > 10),
        "{folding} {formats:?}"
    );
}
