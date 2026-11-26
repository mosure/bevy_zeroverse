use super::*;
use crate::scene::procedural_indoor::{
    objects::{audit, build_object, part_label},
    placement::{rotated_half, Footprint},
};

// Test the flat underside of the submitted runner at the rubber bearing.
fn supported_under(stand: &Assembly, finish: Surface, point: Vec3) -> bool {
    stand
        .parts
        .iter()
        .filter(|((surface, label), _)| *surface == finish && part_label(label) == "whiteboard")
        .any(|(_, g)| {
            g.indices.as_chunks::<3>().0.iter().any(|t| {
                let [a, b, c] = t.map(|i| Vec3::from(g.positions[i as usize]));
                if [a, b, c].iter().any(|v| (v.y - point.y).abs() > 2e-6) {
                    return false;
                }
                let ab = b.xz() - a.xz();
                let ac = c.xz() - a.xz();
                let ap = point.xz() - a.xz();
                let area = ab.perp_dot(ac);
                area > 0.
                    && ap.perp_dot(ac) >= -1e-8
                    && ab.perp_dot(ap) >= -1e-8
                    && ap.perp_dot(ac) + ab.perp_dot(ap) <= area + 1e-8
            })
        })
}

#[test]
fn leaves_swing_within_the_stand_and_their_load_stays_between_its_feet() {
    let mut arrangements = [0; 3];
    let mut swung = 0;
    let mut parked = 0;
    for seed in 0..192 {
        let mut o = audit::fixture(ObjectKind::TeachingBoard, seed);
        if seed % 3 == 1 {
            o.size *= Vec3::new(0.83, 1.08, 0.80);
        }
        let p = parameters(&o);
        let (size, pose) = board_section(&o, &p);
        let board = p.writing_board.as_ref().unwrap();
        arrangements[board.arrangement as usize] += 1;
        assert_eq!(p.panels, board.leaves.len());
        let mut mass_center = Vec3::ZERO;
        let mut mass = 0.;
        for leaf in &board.leaves {
            let center = board
                .swing(leaf.hinge, size.z)
                .transform_point(Vec3::X * leaf.flat_x_m);
            mass_center += pose.transform_point(center) * leaf.width_m;
            mass += leaf.width_m;
            let Some(hinge) = leaf.hinge else { continue };
            if hinge.angle_rad.abs() > 0.1 {
                swung += 1;
            } else if hinge.angle_rad == 0. {
                parked += 1;
            }
            let side = (leaf.flat_x_m - hinge.axis_x_m).signum();
            for t in 0..=8 {
                let h = Hinge {
                    angle_rad: hinge.stop_rad * t as f32 / 8.,
                    ..hinge
                };
                let rotation = board.swing(Some(h), size.z);
                let pin = board.pivot(hinge, size.z);
                assert!(rotation.transform_point(pin).distance(pin) < 2e-6);
                for x in [-0.5, 0.5] {
                    for y in [-0.5, 0.5] {
                        for z in [-0.5, 0.5] {
                            let corner =
                                Vec3::new(leaf.flat_x_m + x * leaf.width_m, y * size.y, z * size.z);
                            let rotated = rotation.transform_point(corner);
                            assert!(
                                (rotated.x - hinge.axis_x_m) * side > 0.,
                                "wing crosses its neighbor in product {seed}"
                            );
                            let v = pose.transform_point(rotated);
                            assert!(
                                v.x.abs() <= o.size.x * 0.5 + 1e-6
                                    && v.z.abs() <= o.size.z * 0.5 + 1e-6
                                    && (0.0..=o.size.y).contains(&v.y),
                                "swing escapes product {seed}: {v:?}"
                            );
                        }
                    }
                }
            }
        }
        mass_center /= mass;
        let feet = board.carrier_width_m * (p.post_span * 0.5 + p.post_splay);
        assert!(
            (mass_center.x - board.post_center_x_m).abs() < feet
                && mass_center.z.abs() < o.size.z * p.foot_depth * 0.5,
            "product {seed} is top-heavy"
        );
        // Each wing can be parked independently at either mechanical stop.
        for mask in 0..1 << board.leaves.len() {
            let mut center = Vec3::ZERO;
            for (index, leaf) in board.leaves.iter().enumerate() {
                let hinge = leaf.hinge.map(|h| Hinge {
                    angle_rad: if mask & (1 << index) == 0 {
                        0.
                    } else {
                        h.stop_rad
                    },
                    ..h
                });
                let c = board
                    .swing(hinge, size.z)
                    .transform_point(Vec3::X * leaf.flat_x_m);
                center += pose.transform_point(c) * (leaf.width_m / mass);
            }
            assert!(
                (center.x - board.post_center_x_m).abs() < feet
                    && center.z.abs() < o.size.z * p.foot_depth * 0.5,
                "wing stops destabilize stand {seed}"
            );
        }
        let built = build_object(&o);
        audit::check(&o, &built).unwrap();
        if let Some(pad) = board.foot_pads {
            let (_, pads) = built
                .parts
                .iter()
                .find(|((surface, label), _)| {
                    *surface == Surface::Rubber && part_label(label) == "whiteboard"
                })
                .unwrap();
            assert_eq!(pads.indices.len() / 3, 4 * 72);
            for side in [-1., 1.] {
                let x = board.post_center_x_m + side * feet;
                for z in [-pad.center_z_m, pad.center_z_m] {
                    for (height, normal) in [(0., Vec3::NEG_Y), (pad.height_m, Vec3::Y)] {
                        let pt = Vec3::new(x, height, z);
                        assert!(pads
                            .positions
                            .iter()
                            .zip(&pads.normals)
                            .any(|(&pos, &n)| Vec3::from(pos).distance(pt) < 2e-6
                                && Vec3::from(n).dot(normal) > 0.99999));
                    }
                    assert!(
                        supported_under(&built, p.finish, Vec3::new(x, pad.height_m, z)),
                        "unseated runner {seed}"
                    );
                }
            }
        }
        assert!(parameters_at(&o, INTRODUCED - 1).writing_board.is_none());
    }
    assert!(arrangements.iter().all(|&count| count > 15) && swung > 30 && parked > 3);
}

#[test]
fn installed_faces_are_closed_and_read_upright_from_either_side() {
    let mut independent = 0;
    let mut shared = 0;
    let mut plain = 0;
    for seed in 0..72 {
        let o = audit::fixture(ObjectKind::TeachingBoard, seed);
        let p = parameters(&o);
        let (size, pose) = board_section(&o, &p);
        let b = p.writing_board.as_ref().unwrap();
        let built = build_object(&o);
        let mut last_u = 0.;
        let mut stock = std::collections::BTreeSet::new();
        for leaf in &b.leaves {
            let mut shell = Assembly::default();
            b.sheet(&mut shell, p.finish, leaf, size);
            let mut edges = std::collections::BTreeMap::new();
            for g in shell.parts.values() {
                for tri in g.indices.as_chunks::<3>().0 {
                    let [a, b, c] = tri.map(|i| g.positions[i as usize].map(f32::to_bits));
                    for (a, b) in [(a, b), (b, c), (c, a)] {
                        edges.entry((a.min(b), a.max(b))).or_insert([0; 2])[usize::from(a > b)] +=
                            1;
                    }
                }
            }
            assert!(edges.values().all(|&e| e == [1, 1]));
            if b.continuous_front {
                assert_eq!(leaf.front_u[0], last_u);
                last_u = leaf.front_u[1];
                assert_eq!(leaf.front_slot, b.leaves[0].front_slot);
            } else {
                assert_eq!(leaf.front_u, [0., 1.]);
                assert!(stock.insert(leaf.front_slot));
            }
            let installed = pose
                .mul_transform(b.swing(leaf.hinge, size.z))
                .mul_transform(Transform::from_xyz(leaf.flat_x_m, 0., 0.));
            for sign in [1., -1.] {
                let slot = if sign > 0. {
                    Some(leaf.front_slot)
                } else {
                    leaf.reverse_slot
                };
                let Some(slot) = slot else {
                    plain += 1;
                    let (_, g) = built
                        .parts
                        .iter()
                        .find(|((stock, label), _)| {
                            *stock == b.rear_stock && part_label(label) == "whiteboard"
                        })
                        .expect("the finished reverse face must use its sampled stock");
                    let normal = installed.rotation * Vec3::NEG_Z;
                    let center =
                        installed.transform_point(Vec3::NEG_Z * (size.z * 0.5 - b.recess_m));
                    let target_area = (leaf.width_m - 2. * b.border_m) * (size.y - 2. * b.border_m);
                    let reverse_area: f32 = g
                        .indices
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .filter_map(|tri| {
                            let vertices = tri.map(|i| Vec3::from(g.positions[i as usize]));
                            let [a, c, d] = vertices;
                            let cross = (c - a).cross(d - a);
                            let centroid = (a + c + d) / 3.;
                            (cross.dot(normal) > 0.
                                && vertices
                                    .iter()
                                    .all(|v| (*v - center).dot(normal).abs() < 2e-6)
                                && (centroid - center).dot(installed.rotation * Vec3::X).abs()
                                    < leaf.width_m * 0.5 - b.border_m)
                                .then_some(cross.length() * 0.5)
                        })
                        .sum();
                    assert!(
                        (reverse_area - target_area).abs() < 3e-5,
                        "rear skin of {seed}"
                    );
                    continue;
                };
                let g = &built.parts[&(
                    Surface::Whiteboard,
                    format!("whiteboard#finish{slot}#uv_local"),
                )];
                let u = if sign > 0. { leaf.front_u } else { [0., 1.] };
                let normal = installed.rotation * (sign * Vec3::Z);
                for (x, ux) in [(-1., u[0]), (1., u[1])] {
                    for (y, v) in [(-1., 1.), (1., 0.)] {
                        let corner = Vec3::new(
                            x * sign * (leaf.width_m * 0.5 - b.border_m),
                            y * (size.y * 0.5 - b.border_m),
                            sign * (size.z * 0.5 - b.recess_m),
                        );
                        let point = installed.transform_point(corner);
                        assert!(
                            g.positions
                                .iter()
                                .zip(&g.normals)
                                .zip(&g.uvs)
                                .any(|((&q, &n), uv)| Vec3::from(q).distance(point) < 2e-6
                                    && Vec3::from(n).dot(normal) > 0.99999
                                    && *uv == [ux, v]),
                            "missing or mirrored skin at {seed}"
                        );
                    }
                }
            }
        }
        if b.continuous_front {
            assert_eq!(last_u, 1.);
            shared += 1;
        } else if b.leaves.len() > 1 {
            independent += 1;
        }
    }
    assert!(shared > 4 && independent > 4 && plain > 18);
}

#[test]
fn ledge_tools_rest_on_the_level_tray_clear_of_its_lip_and_each_other() {
    let mut stock = [0; 2];
    let mut empty = 0;
    for seed in 0..96 {
        let o = audit::fixture(ObjectKind::TeachingBoard, seed);
        let p = parameters(&o);
        let Some(t) = p.writing_board.unwrap().ledge else {
            continue;
        };
        empty += usize::from(t.contents.is_empty());
        let mut fitted = Vec::new();
        for item in &t.contents {
            stock[usize::from(item.eraser)] += 1;
            let footprint = Footprint {
                center: item.center,
                half: rotated_half(item.size.xz(), item.yaw),
                yaw: item.yaw,
            };
            assert!(fitted.iter().all(|f| !footprint.overlaps(*f, 0.0039)));
            fitted.push(footprint);
            let mut tool = Assembly::default();
            presentation::contents::build_ledge_item(&mut tool, t.top_m, item);
            let mut low = f32::INFINITY;
            let mut contact_x = [f32::INFINITY, f32::NEG_INFINITY];
            let along = Vec2::new(item.yaw.cos(), -item.yaw.sin());
            for ((_, label), g) in tool.parts {
                assert_eq!(part_label(&label), "stationery");
                for [x, y, z] in g.positions {
                    assert!(x > t.lo_m.x && x < t.hi_m.x && z > t.lo_m.y && z < t.hi_m.y - 0.003);
                    low = low.min(y);
                    if (y - t.top_m).abs() < 4e-6 {
                        let x = (Vec2::new(x, z) - item.center).dot(along);
                        contact_x = [contact_x[0].min(x), contact_x[1].max(x)];
                    }
                }
            }
            assert!(
                contact_x[0] < -0.35 * item.size.x && contact_x[1] > 0.35 * item.size.x,
                "unbalanced tool {seed}"
            );
            assert!(
                (low - t.top_m).abs() < 4e-6,
                "unsupported tool {seed}: {low} vs {}",
                t.top_m
            );
        }
    }
    assert!(stock[0] > 60 && stock[1] > 6 && empty > 0);
}
