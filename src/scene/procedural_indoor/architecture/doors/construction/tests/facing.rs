use super::*;
use crate::scene::procedural_indoor::motion::{IndoorMotionConfig, MotionKind};

fn protected() -> DoorInstallation {
    let mut d = door(19, DoorMechanism::Paired);
    d.program.panel = DoorPanel::VisionLite;
    d.program.finish = Surface::Wood;
    d.program.vision_light = Some(VisionLight {
        shape: VisionShape::Porthole { diameter_m: 0.42 },
        center_height_m: 1.48,
        latch_bias: 0.48,
        metal_bead: true,
    });
    d.program.veneer = Some(DoorVeneer {
        cut_seed: 411,
        matching: VeneerMatch::Book,
        grain_angle_radians: 0.032,
    });
    d.program.kick_plate = Some(KickPlateStock {
        faces: ProtectedFaces::Both,
        finish: Surface::Plastic,
        height_m: 0.46,
        edge_inset_m: 0.024,
        bottom_inset_m: 0.011,
        gauge_m: 0.0025,
        corner_clip_m: 0.012,
    });
    d
}

fn skin(d: &DoorInstallation, index: usize, local: bool) -> (Assembly, Leaf, Fit) {
    let mut leaf = d.leaves()[index];
    if local {
        leaf.transform = Transform::IDENTITY;
    }
    let j = Joinery::sample(&d.program);
    let fit = Fit::new(d, leaf, index, &j);
    let mut a = Assembly::default();
    panel::build(&mut a, &d.program, leaf, &fit, &j, index, &labels(&j));
    (a, leaf, fit)
}

#[test]
fn plate_replaces_only_solid_face_stock_leaving_the_lite_and_clipped_corners_visible() {
    let d = protected();
    assert_eq!(d.kick_plates().len(), 4);
    for (i, _) in d.leaves().iter().enumerate() {
        let (a, leaf, fit) = skin(&d, i, true);
        for side in [-1., 1.] {
            let p = fit
                .protective_face(&d.program, leaf, i, side, &Joinery::sample(&d.program))
                .unwrap();
            let hole = fit.opening.unwrap();
            assert!(p.bounds.max.y + 0.01 < hole.min.y);
            let b = p.bounds;
            for xy in [
                b.center(),
                b.min + Vec2::new(0.043, 0.008),
                b.max - Vec2::new(0.006, 0.034),
            ] {
                let (depth, surface, label) = hit(&a, xy, side, |_, _| true).unwrap();
                assert_eq!(surface, p.finish);
                assert!(label.starts_with("door_hardware#"));
                assert!((depth - p.base_z - side * p.gauge_m).abs() < 2e-6);
                assert!(hit_at_depth(
                    &a,
                    xy,
                    side,
                    |_, l| l.starts_with("door#"),
                    |z| z * side > 0.
                )
                .is_none());
            }
            for xy in [
                b.min + Vec2::splat(p.corner_clip_m * 0.19),
                Vec2::new(b.max.x + 0.004, b.center().y),
                Vec2::new(b.center().x, b.min.y - 0.003),
                Vec2::new(b.center().x, b.max.y + 0.006),
            ] {
                let (depth, surface, label) = hit(&a, xy, side, |_, _| true).unwrap();
                assert_eq!(surface, d.program.finish);
                assert!(label.starts_with("door#") && (depth - p.base_z).abs() < 2e-6);
            }
            assert_eq!(
                hit(&a, hole.center(), side, |_, _| true).unwrap().1,
                Surface::GlassInterior
            );
            let hardware = &a.parts[&(p.finish, labels(&Joinery::sample(&d.program)).hardware)];
            for idx in hardware.indices.as_chunks::<3>().0 {
                let [u, v, w] = idx.map(|n| Vec3::from_array(hardware.positions[n as usize]));
                let n = (v - u).cross(w - u);
                assert!(n.length_squared() > 1e-16);
                for j in idx {
                    assert!(n.dot(Vec3::from_array(hardware.normals[*j as usize])) > 0.);
                }
            }
        }
    }
}

#[test]
fn paired_sheets_have_metric_grain_independent_backs_and_deliberately_matched_latch_edges() {
    let mut d = protected();
    for matching in [
        VeneerMatch::Independent,
        VeneerMatch::Continuous,
        VeneerMatch::Book,
    ] {
        d.program.veneer.as_mut().unwrap().matching = matching;
        let leaves = d.leaves();
        let uv = |i: usize, side: f32, dx: f32, y: f32| {
            let leaf = leaves[i];
            let latch = if i == 0 { 0.5 } else { -0.5 };
            let chart =
                d.program.veneer.unwrap().charts(&d.program, leaf, i)[usize::from(side > 0.)];
            chart.transform_point2(Vec2::new((latch * leaf.size.x + dx) * side, y))
        };
        for side in [-1., 1.] {
            let origin = uv(0, side, 0., 0.23);
            let left = uv(0, side, -0.11, 0.23);
            assert!((left.distance(origin) - 0.11).abs() < 1e-5);
            assert!(origin.distance(uv(0, -side, 0., 0.23)) > 0.01);
            if matching == VeneerMatch::Independent {
                assert!(origin.distance(uv(1, side, 0., 0.23)) > 0.01);
            } else {
                assert!(origin.abs_diff_eq(uv(1, side, 0., 0.23), 1e-5));
                let right = uv(1, side, 0.11, 0.23);
                assert!(right.abs_diff_eq(
                    if matching == VeneerMatch::Book {
                        left
                    } else {
                        2. * origin - left
                    },
                    1e-5
                ));
            }
        }
    }
    let (a, leaf, fit) = skin(&d, 0, true);
    let charts = d.program.veneer.unwrap().charts(&d.program, leaf, 0);
    let g = &a.parts[&(d.program.finish, labels(&Joinery::sample(&d.program)).leaf)];
    let mut checked = 0;
    for ((coords, n), uv) in g.positions.iter().zip(&g.normals).zip(&g.uvs) {
        let v = Vec3::from(*coords);
        let side = n[2];
        if side.abs() != 1. || (v.z * side - leaf.size.z * 0.5).abs() > 1e-5 {
            continue;
        }
        let expected = charts[usize::from(side > 0.)].transform_point2(Vec2::new(v.x * side, v.y));
        assert!(expected.abs_diff_eq(Vec2::from(*uv), 1e-5));
        if fit.opening.unwrap().contains(v.truncate()) {
            checked += 1;
        }
    }
    assert!(checked > 16); // Includes curved and flat shoulders of the porthole.
}

#[test]
fn facing_tracks_its_own_hinge_and_surface_coordinates_survive_articulation() {
    let mut d = protected();
    d.center = Vec3::new(3., 0.16, -2.);
    d.yaw = 0.68;
    for opening in [0., 0.47, 1.] {
        d.program.opening = opening;
        let mut articulated = Assembly {
            motion_policy: Some(IndoorMotionConfig {
                door_fraction: 1.,
                ..Default::default()
            }),
            ..Default::default()
        };
        d.build(&mut articulated, true);
        assert_eq!(articulated.joints.len(), 2);
        for (i, joint) in articulated.joints.iter().enumerate() {
            assert_eq!(joint.kind, MotionKind::Door);
            let (local, _, _) = skin(&d, i, true);
            let (world, leaf, _) = skin(&d, i, false);
            assert_eq!(joint.frame, leaf.transform);
            let key = joint
                .parts
                .iter()
                .find(|(s, l)| *s == Surface::Plastic && l.starts_with("door_hardware#"))
                .unwrap();
            let submitted = &articulated.parts[key];
            for ((s, l), g) in &world.parts {
                let unposed = &local.parts[&(*s, l.clone())];
                assert_eq!((&g.indices, &g.uvs), (&unposed.indices, &unposed.uvs));
                for (u, v) in unposed.positions.iter().zip(&g.positions) {
                    assert!(leaf
                        .transform
                        .transform_point(Vec3::from(*u))
                        .abs_diff_eq(Vec3::from(*v), 2e-5));
                }
                if l.starts_with("door_hardware#") {
                    assert!(submitted.positions.starts_with(&g.positions));
                }
            }
        }
    }
}

#[test]
fn face_stock_replays_and_missing_archive_fields_keep_the_original_uvs() {
    let mut faces = [0; 3];
    let mut matching = [0; 3];
    let mut cross = 0;
    for i in 0..768 {
        let mechanism = DoorMechanism::ALL[i % 5];
        let p = DoorProgram::sample(i as u64, mechanism);
        assert!(p.validate());
        assert_eq!(p, DoorProgram::sample(i as u64, mechanism));
        if let Some(k) = p.kick_plate {
            assert!(matches!(
                mechanism,
                DoorMechanism::Hinged | DoorMechanism::Paired
            ));
            assert!(matches!(p.panel, DoorPanel::Flush | DoorPanel::VisionLite));
            faces[match k.faces {
                ProtectedFaces::Push => 0,
                ProtectedFaces::Pull => 1,
                ProtectedFaces::Both => 2,
            }] += 1;
        }
        if let Some(v) = p.veneer {
            matching[match v.matching {
                VeneerMatch::Independent => 0,
                VeneerMatch::Continuous => 1,
                VeneerMatch::Book => 2,
            }] += 1;
            cross += usize::from(v.grain_angle_radians > 1.);
        }
    }
    assert!(faces.into_iter().all(|n| n > 1) && matching.into_iter().all(|n| n > 1) && cross > 3);
    let mut d = protected();
    let mut archived = serde_json::to_value(&d.program).unwrap();
    for key in ["kick_plate", "veneer"] {
        archived.as_object_mut().unwrap().remove(key);
    }
    d.program = serde_json::from_value(archived.clone()).unwrap();
    assert_eq!(serde_json::to_value(&d.program).unwrap(), archived);
    assert!(d.kick_plates().is_empty() && d.program.validate());
    let (a, _, _) = skin(&d, 0, true);
    let g = &a.parts[&(d.program.finish, labels(&Joinery::sample(&d.program)).leaf)];
    for ((v, n), uv) in g.positions.iter().zip(&g.normals).zip(&g.uvs) {
        if n[2].abs() == 1. {
            assert_eq!(*uv, [v[0] * n[2], v[1]]);
        }
    }
}
