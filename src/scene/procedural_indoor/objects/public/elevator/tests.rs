use super::*;
use crate::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest},
    materials::{finish_slot, registered_finish},
    objects::{audit, build_object_at_version, part_label},
};
use station::{Direction, ReadoutMount};

#[test]
fn cars_on_a_landing_share_service_directions_even_after_manifest_roundtrip() {
    let mut multiple = 0;
    let mut terminal = 0;
    for seed in 95500..95520 {
        let scene = IndoorManifest::generate_with_humans(
            seed,
            IndoorLayout::ElevatorLanding,
            0.65,
            3,
            0.25,
        )
        .unwrap();
        let cars: Vec<_> = scene
            .objects
            .iter()
            .filter(|o| !o.neighbor && o.kind == ObjectKind::Elevator)
            .collect();
        assert!(!cars.is_empty());
        multiple += usize::from(cars.len() > 1);
        let expected = Levels::sample(seed);
        for o in cars {
            let copy: IndoorObject =
                serde_json::from_slice(&serde_json::to_vec(o).unwrap()).unwrap();
            let (s, _) = installation(&copy);
            assert_eq!(Levels::decode(copy.variant), Some(expected));
            assert_eq!(s.levels, expected);
            assert!(s.displayed_floor <= expected.highest);
            assert_eq!(
                s.call_directions.contains(&Some(Direction::Down)),
                expected.floor > 0
            );
            assert_eq!(
                s.call_directions.contains(&Some(Direction::Up)),
                expected.floor < expected.highest
            );
            assert!(s.registered.is_none() || s.call_directions.contains(&s.registered));
            let p = parameters(o);
            assert_eq!(p.floor_number, u32::from(s.displayed_floor));
            if p.opening > 0. {
                assert_eq!(s.displayed_floor, expected.floor);
                assert!(s.registered.is_none());
            }
            if s.call_directions[1].is_none() {
                terminal += 1;
            }
        }
    }
    assert!(multiple >= 4 && terminal >= 4);
}

#[test]
fn pressings_fit_the_actual_jamb_and_readouts_fit_their_mount() {
    let mut mounts = [0; 3];
    let mut horizontal = 0;
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::Elevator, seed);
        if seed % 2 == 0 {
            o.variant = Levels::sample(seed).core_variant();
            o.size.y = 2.66 + (seed % 19) as f32 * 0.12;
        }
        let car = car_size(&o);
        let p = sample(&o);
        let (s, _) = installation(&o);
        let margin = if is_core(&o) { o.size.x * 0.016 } else { 0. };
        let left = car.x * p.portal_left - margin;
        let right = car.x * p.portal_right + margin;
        let on_jamb = |xy: Vec2, size: Vec2| {
            let lo = xy.x - size.x * 0.5;
            let hi = xy.x + size.x * 0.5;
            lo >= -car.x * 0.5 && hi <= car.x * 0.5 && (hi <= left || lo >= right)
        };
        assert!(on_jamb(s.plate_center_m, s.plate_size_m));
        match s.mount {
            ReadoutMount::Header => {
                mounts[0] += 1;
                assert!(
                    s.readout_center_m.y - s.readout_size_m.y * 0.5
                        > p.portal_top * car.y + if is_core(&o) { 0.016 } else { 0. }
                );
                assert!(s.readout_center_m.y + s.readout_size_m.y * 0.5 < o.size.y);
            }
            ReadoutMount::Jamb => {
                mounts[1] += 1;
                assert!(on_jamb(s.readout_center_m, s.readout_size_m));
            }
            ReadoutMount::Integrated => {
                mounts[2] += 1;
                assert!(
                    ((s.readout_center_m - s.plate_center_m).abs() + s.readout_size_m * 0.5)
                        .cmple(s.plate_size_m * 0.5)
                        .all()
                );
            }
        }
        horizontal += usize::from(s.horizontal);
        assert!((0.032..=0.048).contains(&s.button_diameter_m));
        let a = build_object_at_version(&o, None, LANDING_VERSION);
        audit::check(&o, &a).unwrap();
        let mut film_faces = 0;
        for ((surface, label), g) in a
            .parts
            .iter()
            .filter(|((_, l), _)| part_label(l) == "control_panel")
        {
            assert!(g
                .positions
                .iter()
                .all(|v| (s.face_z_m - 1e-6..=o.size.z * 0.5).contains(&v[2])));
            if let Some(slot) = finish_slot(*surface, label) {
                assert!(registered_finish(*surface, slot));
            }
            if *surface == Surface::Ink && label.contains("#uv_local") {
                film_faces += 1;
                assert!(g.uvs.iter().flatten().all(|u| (0.0..=1.0).contains(u)));
            }
            for tri in g.indices.as_chunks::<3>().0 {
                let pos = |i: u32| Vec3::from_array(g.positions[i as usize]);
                let n = Vec3::from_array(g.normals[tri[0] as usize]);
                assert!(
                    (pos(tri[1]) - pos(tri[0]))
                        .cross(pos(tri[2]) - pos(tri[0]))
                        .dot(n)
                        > 0.
                );
            }
        }
        assert_eq!(film_faces, 2);
        o.finish_seed = Some(seed ^ 0xaffe);
        let (recolored, _) = installation(&o);
        assert_eq!(s.levels, recolored.levels);
        assert_eq!(s.plate_center_m, recolored.plate_center_m);
        assert_eq!(s.readout_center_m, recolored.readout_center_m);
        assert_eq!(s.button_diameter_m, recolored.button_diameter_m);
    }
    assert!(mounts.into_iter().all(|v| v > 15) && horizontal > 1);
}

#[test]
fn sill_sections_have_negative_guides_below_each_door_and_a_separate_transfer_joint() {
    for seed in 0..96 {
        let mut o = audit::fixture(ObjectKind::Elevator, seed);
        if seed % 2 == 0 {
            o.variant = 20;
        }
        let (station, sill) = installation(&o);
        let mut a = Assembly::default();
        sill.build(&mut a);
        let top_at = |z: f32| {
            let x = sill.left_m + (sill.right_m - sill.left_m) * 0.312;
            let mut hits = Vec::new();
            for ((surface, _), g) in &a.parts {
                for tri in g.indices.as_chunks::<3>().0 {
                    let p =
                        [tri[0], tri[1], tri[2]].map(|i| Vec3::from_array(g.positions[i as usize]));
                    if (p[1] - p[0]).cross(p[2] - p[0]).y <= 0. {
                        continue;
                    }
                    let [u, v, w] = p.map(|v| v.xz());
                    let q = Vec2::new(x, z) - u;
                    let det = (v - u).perp_dot(w - u);
                    let b = q.perp_dot(w - u) / det;
                    let c = (v - u).perp_dot(q) / det;
                    if b >= 0. && c >= 0. && b + c <= 1. {
                        hits.push((
                            *surface,
                            p[0].y + (p[1].y - p[0].y) * b + (p[2].y - p[0].y) * c,
                        ));
                    }
                }
            }
            assert_eq!(hits.len(), 1, "seed {seed}, z={z}");
            hits[0]
        };
        let car = car_size(&o);
        let p = parameters(&o);
        for (pos, _) in leaf_bounds(&p) {
            let center = pos[2] * car.z;
            let (mat, height) = top_at(center);
            assert_eq!(mat, Surface::Metal);
            assert!((height - (sill.height_m - sill.guide_depth_m)).abs() < 1e-6);
            assert!((top_at(center + sill.guide_width_m * 0.66).1 - sill.height_m).abs() < 1e-6);
        }
        let (bed, h) = top_at(sill.joint_center_m);
        assert_eq!(bed, Surface::Rubber);
        assert!(h > 0. && h < sill.height_m);
        assert!(sill.front_m > station.face_z_m);
        let legacy = build_object_at_version(&o, None, LANDING_VERSION - 1);
        assert!(!legacy
            .parts
            .keys()
            .any(|(_, l)| part_label(l) == "control_panel" && l.contains("#uv_local")));
    }
}
