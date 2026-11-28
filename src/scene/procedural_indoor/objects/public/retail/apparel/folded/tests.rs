use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object_at_version, part_label};

#[test]
fn stocked_layers_stay_on_their_support_inside_distinct_table_bays() {
    let mut layouts = [0; 2];
    let mut cuts = [[0; 2]; 4];
    let mut first_shorter = 0;
    let mut first_vacant = 0;
    let mut contrast = 0;
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::ApparelDisplay, seed);
        o.size = if seed % 5 == 0 {
            Vec3::new(1., 0.83, 0.64)
        } else {
            Vec3::new(1.9, 1.05, 0.9)
        };
        let display = folded_apparel(&o, FOLDED_APPAREL_VERSION).unwrap();
        let Installation::Folded { piles, layers, .. } = parameters(&o) else {
            panic!()
        };
        assert!(!display.piles.is_empty());
        assert!(display.piles.iter().map(|p| p.layers.len()).sum::<usize>() <= piles * layers);
        let cell = (o.size.xz() - Vec2::splat(0.055))
            / Vec2::new(display.columns as f32, display.rows as f32);
        let mut height = display.deck_y_m;
        layouts[display.rows - 1] += 1;
        let first = display.piles.iter().find(|p| p.column == 0 && p.row == 0);
        first_vacant += usize::from(first.is_none());
        first_shorter += usize::from(first.is_some_and(|f| {
            display
                .piles
                .iter()
                .any(|p| p.layers.len() > f.layers.len())
        }));
        for p in &display.piles {
            let item = &p.garment;
            cuts[item.cut as usize][usize::from(item.surface == Surface::FabricAlt)] += 1;
            contrast += usize::from(item.dye_slot != item.binding_slot);
            assert!(p.column < display.columns && p.row < display.rows);
            let c = cell
                * (Vec2::new(p.column as f32 + 0.5, p.row as f32 + 0.5)
                    - Vec2::new(display.columns as f32, display.rows as f32) * 0.5);
            let mut next = display.deck_y_m;
            let mut previous: Option<&FoldedLayer> = None;
            let mut buried = Assembly::default();
            if p.layers.len() > 1 {
                garment::build(&mut buried, item, false);
            }
            for l in &p.layers {
                let rot = Quat::from_rotation_y(l.yaw_rad);
                assert_eq!((rot * Vec3::Z).z > 0., display.rows == 1 || p.row == 1);
                assert!((l.base_m.y - next).abs() < 2e-6);
                for x in [-1., 1.] {
                    for z in [-1., 1.] {
                        let corner = Vec3::new(x * item.size_m.x, 0., z * item.size_m.z)
                            * 0.5
                            * l.size_grade;
                        let corner_w = l.base_m + rot * corner;
                        assert!(
                            (corner_w.xz() - c).abs().cmplt(cell * 0.5).all(),
                            "{seed} {corner_w:?}"
                        );
                        if let Some(lower) = previous {
                            // A rectangle in the load-bearing centre of the upper piece
                            // must rest on the continuous flat crown of the lower one.
                            let q = Quat::from_rotation_y(-lower.yaw_rad)
                                * (l.base_m + rot * (corner * 0.5) - lower.base_m)
                                / lower.size_grade;
                            assert!(
                                (vertical_extent(&buried, q.xz()).1 - item.size_m.y).abs() < 1e-6,
                                "{seed} {q:?}"
                            );
                        }
                    }
                }
                next += item.size_m.y * l.size_grade;
                previous = Some(l);
            }
            height = height.max(next + item.detail_rise_m * previous.unwrap().size_grade);
        }
        assert!((height - o.size.y).abs() < 2e-6);
        if seed < 24 {
            let mesh = build_object_at_version(&o, None, FOLDED_APPAREL_VERSION);
            let stats = audit::check(&o, &mesh).unwrap();
            assert!(stats.min_y_m.abs() < 1e-5 && stats.max_envelope_overrun_m < 1e-5);
            assert!(mesh
                .parts
                .keys()
                .all(|(_, label)| matches!(part_label(label), "clothes" | "retail_fixture")));
        }
    }
    assert!(layouts.iter().all(|&n| n > 60) && cuts.iter().flatten().all(|&n| n > 5));
    assert!(first_shorter > 20 && first_vacant > 12 && contrast > 15);
}

fn vertical_extent(mesh: &Assembly, point: Vec2) -> (f32, f32) {
    let mut lowest = f32::INFINITY;
    let mut highest = f32::NEG_INFINITY;
    for g in mesh.parts.values() {
        for indices in g.indices.as_chunks::<3>().0 {
            let [a, b, c] = indices.map(|i| Vec3::from(g.positions[i as usize]));
            let ab = (b - a).xz();
            let ac = (c - a).xz();
            let ap = point - a.xz();
            let det = ab.perp_dot(ac);
            if det.abs() < 1e-10 {
                continue;
            }
            let u = ap.perp_dot(ac) / det;
            let v = ab.perp_dot(ap) / det;
            if u >= -1e-5 && v >= -1e-5 && u + v <= 1. + 1e-5 {
                let h = a.y + (b.y - a.y) * u + (c.y - a.y) * v;
                highest = highest.max(h);
                lowest = lowest.min(h);
            }
        }
    }
    (lowest, highest)
}

#[test]
fn collars_are_open_pockets_are_raised_and_all_cloth_panels_face_out() {
    for seed in 0..64 {
        let mut rng = stream(seed, 47);
        for cut in [
            FoldedCut::Tee,
            FoldedCut::Pullover,
            FoldedCut::ButtonDown,
            FoldedCut::Trousers,
        ] {
            let item = FoldedGarment::sample(cut, &mut rng);
            let mut top = Assembly::default();
            garment::build(&mut top, &item, true);
            for ((surface, label), g) in &top.parts {
                assert_eq!(part_label(label), "clothes");
                assert!(g.uvs.iter().flatten().all(|v| v.is_finite()));
                if *surface == Surface::Plastic {
                    continue;
                }
                for ids in g.indices.as_chunks::<3>().0 {
                    let [a, b, c] = ids.map(|i| Vec3::from(g.positions[i as usize]));
                    let normal = (b - a).cross(c - a);
                    let n = Vec3::from(g.normals[ids[0] as usize]);
                    assert!(
                        (n.length() - 1.).abs() < 1e-4 && normal.dot(n) > 0.,
                        "{seed} {cut:?} {ids:?}"
                    );
                    if [a, b, c].iter().all(|v| v.y == item.size_m.y) {
                        assert!(normal.y > 0.);
                    }
                }
            }
            let sample = item.feature_center_m + item.feature_half_size_m * Vec2::new(0.011, 0.027);
            let buried = {
                let mut m = Assembly::default();
                garment::build(&mut m, &item, false);
                m
            };
            assert!((vertical_extent(&buried, sample).1 - item.size_m.y).abs() < 1e-6);
            let visible_y = if cut == FoldedCut::Trousers {
                item.size_m.y + item.detail_rise_m
            } else {
                item.size_m.y * 0.54
            };
            assert!(
                (vertical_extent(&top, sample).1 - visible_y).abs() < 1e-6,
                "{seed} {cut:?}"
            );
            let max_y = top
                .parts
                .values()
                .flat_map(|g| g.positions.iter().map(|p| p[1]))
                .fold(f32::NEG_INFINITY, f32::max);
            assert!((max_y - item.size_m.y - item.detail_rise_m).abs() < 1e-6);
        }
    }
}

#[test]
fn folded_replay_is_local_to_the_fixture_and_preserves_older_meshes() {
    for kind in [ObjectKind::ApparelDisplay, ObjectKind::GroceryDisplay] {
        for seed in 0..24 {
            let mut o = audit::fixture(kind, seed);
            assert!(folded_apparel(&o, FOLDED_APPAREL_VERSION - 1).is_none());
            if let Some(stock) = folded_apparel(&o, FOLDED_APPAREL_VERSION) {
                if seed == 2 {
                    let m = build_object_at_version(&o, None, FOLDED_APPAREL_VERSION - 1);
                    let v = audit::check(&o, &m).unwrap();
                    assert_eq!((v.triangles, v.batches), (2220, 5));
                }
                o.position += Vec3::new(3., 0., 4.);
                o.yaw += 0.3;
                assert_eq!(
                    serde_json::to_value(stock).unwrap(),
                    serde_json::to_value(folded_apparel(&o, FOLDED_APPAREL_VERSION).unwrap())
                        .unwrap()
                );
            } else {
                let prev = build_object_at_version(&o, None, FOLDED_APPAREL_VERSION - 1);
                let now = build_object_at_version(&o, None, FOLDED_APPAREL_VERSION);
                assert_eq!(prev.parts.len(), now.parts.len());
                for (key, g) in prev.parts {
                    let h = &now.parts[&key];
                    assert_eq!(g.positions, h.positions);
                    assert_eq!(g.indices, h.indices);
                    assert_eq!(g.normals, h.normals);
                    assert_eq!(g.uvs, h.uvs);
                }
            }
        }
    }
}

#[test]
fn display_stands_have_floor_contact_under_the_deck_and_open_knee_space_or_toe_reveals() {
    let mut counts = [0; 3];
    for seed in 0..96 {
        let mut o = audit::fixture(ObjectKind::ApparelDisplay, seed);
        o.size = Vec3::new(1.4, 0.87, 0.76);
        let d = folded_apparel(&o, FOLDED_APPAREL_VERSION).unwrap();
        let mut base = Assembly::default();
        d.stand.build(&mut base, o.size, &d);
        let (floor, clear) = match d.stand {
            DisplayTableStand::Apron {
                leg_m,
                end_inset_m,
                side_inset_m,
                ..
            } => {
                counts[0] += 1;
                (
                    o.size.xz() * 0.5
                        - Vec2::new(end_inset_m, side_inset_m)
                        - Vec2::splat(leg_m * 0.5),
                    Vec2::ZERO,
                )
            }
            DisplayTableStand::PanelEnds {
                end_inset_m,
                panel_thickness_m,
                ..
            } => {
                counts[1] += 1;
                (
                    Vec2::new((o.size.x - panel_thickness_m) * 0.5 - end_inset_m, 0.),
                    Vec2::new(0., o.size.z * 0.26),
                )
            }
            DisplayTableStand::Plinth {
                body_xz_m,
                toe_height_m,
                toe_recess_m,
            } => {
                counts[2] += 1;
                let reveal = Vec2::new((body_xz_m.x - toe_recess_m) * 0.5, 0.);
                assert!((vertical_extent(&base, reveal).0 - toe_height_m).abs() < 1e-5);
                (Vec2::ZERO, Vec2::new((body_xz_m.x + o.size.x) * 0.25, 0.))
            }
        };
        let (lo, hi) = vertical_extent(&base, floor);
        assert!(lo.abs() < 1e-5 && (hi - d.deck_y_m).abs() < 1e-5);
        assert!((vertical_extent(&base, clear).0 - (d.deck_y_m - d.deck_thickness_m)).abs() < 1e-5);
    }
    assert!(counts.iter().all(|&n| n > 3));
}
