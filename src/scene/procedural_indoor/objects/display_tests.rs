use super::*;
use crate::scene::procedural_indoor::placement::{rotated_half, Footprint};

#[test]
fn phone_covers_and_camera_islands_follow_the_rounded_case() {
    for seed in 0..256 {
        let o = audit::fixture(ObjectKind::Phone, seed);
        let p = super::super::clutter::tabletop::phone::parameters(&o);
        let body = p.body_height;
        for g in build_object(&o).parts.values() {
            for v in &g.positions {
                if v[1] > body + 1e-6 {
                    let boundary = (v[0].abs() / (p.span.x * 0.49)).powf(p.roundness)
                        + (v[2].abs() / (p.span.y * 0.49)).powf(p.roundness);
                    assert!(boundary <= 1.0001, "phone {seed}: {v:?}; {boundary}");
                }
            }
        }
    }
}

#[test]
fn a_frame_posters_stay_on_their_own_side_of_the_hinge() {
    let mut samples = 0;
    for seed in 0..256 {
        let o = audit::fixture(ObjectKind::InformationStand, seed);
        if public::signage::parameters(&o).support != 1 {
            continue;
        }
        samples += 1;
        for ((surface, _), g) in build_object(&o).parts {
            if surface != Surface::Art {
                continue;
            }
            for tri in g.indices.as_chunks::<3>().0 {
                let z = tri.map(|i| g.positions[i as usize][2]);
                assert!(
                    z.iter().all(|&v| v > o.size.z * 0.01)
                        || z.iter().all(|&v| v < -o.size.z * 0.01)
                );
            }
        }
    }
    assert!(samples > 32);
}

#[test]
fn display_exhibit_meshes_replay_at_metric_extremes() {
    use ObjectKind::*;
    let mut errors = Vec::new();
    for kind in [
        Elevator,
        Sculpture,
        InformationStand,
        ExhibitionPanel,
        NoticeBoard,
        Whiteboard,
        WallArt,
        Monitor,
        Display,
        Phone,
        Laptop,
    ] {
        for seed in 0..128 {
            let mut o = audit::fixture(kind, seed);
            if kind == WallArt {
                o.variant = (seed % 6) as u32;
            }
            o.size *= match seed % 3 {
                0 => Vec3::new(0.80, 1.15, 0.85),
                1 => Vec3::new(1.15, 0.80, 1.20),
                _ => Vec3::ONE,
            };
            let a = build_object(&o);
            if let Err(e) = audit::check(&o, &a) {
                errors.push(format!(
                    "{e}; size={:?}; program={}",
                    o.size,
                    audit::program(&o)
                ));
                continue;
            }
            let replay = build_object(&o);
            assert_eq!(a.parts.len(), replay.parts.len());
            for (key, g) in &a.parts {
                let h = &replay.parts[key];
                assert_eq!(g.positions, h.positions, "{kind:?}/{seed}");
                assert_eq!(g.normals, h.normals);
                assert_eq!(g.uvs, h.uvs);
                assert_eq!(g.indices, h.indices);
            }
        }
    }
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

#[test]
fn elevator_panels_fit_pockets_and_never_intersect_during_travel() {
    for seed in 0..1024 {
        let o = audit::fixture(ObjectKind::Elevator, seed);
        let mut p = public::elevator::parameters(&o);
        assert!((0.003..0.012).contains(&(p.threshold * o.size.y)));
        for step in 0..=20 {
            p.opening = step as f32 / 20.;
            let leaves = public::elevator::leaf_bounds(&p);
            for (i, &(center, size)) in leaves.iter().enumerate() {
                let c = Vec3::from_array(center);
                let h = Vec3::from_array(size) * 0.5;
                assert!(c.x - h.x >= -0.46 && c.x + h.x <= 0.46);
                assert!(c.y - h.y > p.threshold && c.y + h.y < p.portal_top);
                if step == 20 {
                    assert!(c.x + h.x <= p.portal_left || c.x - h.x >= p.portal_right);
                }
                for &(other, dimensions) in &leaves[..i] {
                    let delta = (c - Vec3::from_array(other)).abs();
                    assert!(delta.cmpge(h + Vec3::from_array(dimensions) * 0.5).any());
                }
            }
        }
    }
}

#[test]
fn pinups_keep_rotated_clearance_and_exposed_backing() {
    for seed in 0..1024 {
        let o = audit::fixture(ObjectKind::NoticeBoard, seed);
        let p = equipment::parameters(&o);
        let frame = o.size.x.min(o.size.y) * p.board_frame_fraction;
        let lo = Vec2::new(-o.size.x * 0.5 + frame * 1.8, frame * 1.8);
        let hi = Vec2::new(o.size.x * 0.5 - frame * 1.8, o.size.y - frame * 1.8);
        let notes = equipment::pinups::placements(&o);
        assert!(!notes.is_empty(), "seed {seed}");
        assert_eq!(
            serde_json::to_value(&notes).unwrap(),
            serde_json::to_value(equipment::pinups::placements(&o)).unwrap()
        );
        let mut area = 0.;
        let mut occupied: Vec<Footprint> = Vec::new();
        for note in notes {
            let f = Footprint {
                center: note.center,
                half: rotated_half(note.size, note.yaw),
                yaw: note.yaw,
            };
            assert!(f.inside(lo, hi, frame * 0.15 - 1e-6));
            assert!(occupied.iter().all(|&old| !old.overlaps(f, frame * 0.15)));
            if let Some(divider) = p.board_divider_fraction {
                let x = (divider - 0.5) * (o.size.x - frame * 2.);
                assert!((f.center.x - x).abs() >= f.half.x + frame * 0.32);
            }
            area += note.size.element_product();
            occupied.push(f);
        }
        assert!(area <= (hi - lo).element_product() * 0.48 + 1e-6);
    }
}

#[test]
fn elevator_controls_are_in_front_of_surround_and_floor_surfaces_do_not_overlap() {
    use crate::scene::procedural_indoor::architecture::overlap_tests::intersection_area;
    for seed in 0..64 {
        let o = audit::fixture(ObjectKind::Elevator, seed);
        let a = build_object(&o);
        let mut floors = Vec::new();
        let top = public::elevator::parameters(&o).threshold * o.size.y;
        for ((_, label), g) in &a.parts {
            if part_label(label) == "control_panel" {
                // The surround's front is z=.49; backing is flush, controls forward.
                assert!(g.positions.iter().all(|p| p[2] >= o.size.z * 0.49 - 1e-5));
            }
            for tri in g.indices.as_chunks::<3>().0 {
                let p = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                if p.iter().all(|v| (v.y - top).abs() < 1e-5)
                    && (p[1] - p[0]).cross(p[2] - p[0]).y > 1e-8
                {
                    floors.push(p.map(|p| p.xz()));
                }
            }
        }
        for (i, &a) in floors.iter().enumerate() {
            for &b in &floors[..i] {
                assert!(intersection_area(a, b) < 1e-5, "seed {seed}");
            }
        }
    }
}
