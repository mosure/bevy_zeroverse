use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object, build_object_at_version};

#[test]
fn apparel_rails_keep_sized_garments_inside_their_bays_and_hooks_seated() {
    let mut forms = [0; 3];
    let mut checked_wire = 0;
    let mut swivel = 0;
    let mut bare_stops = 0;
    let mut long_gaps = 0;
    for seed in 0..144 {
        let mut o = audit::fixture(ObjectKind::ApparelDisplay, seed);
        // Include the narrowed fallback allowed by placement.
        if seed % 5 == 0 {
            o.size = Vec3::new(1.0, 1.43, 0.58);
        }
        let Some(rack) = apparel_rail(&o, DISPLAY_RAIL_VERSION) else {
            continue;
        };
        forms[rack.form as usize] += 1;
        let side = rack.form == RailForm::SideHang;
        let mut bounds: Vec<(Vec2, Vec2)> = Vec::new();
        let mut stock_meshes = Vec::new();
        for collection in &rack.collections {
            let mut shape = Assembly::default();
            collection.build(&mut shape);
            stock_meshes.push(shape);
        }
        for hanger in &rack.hangers {
            let arm = &rack.arms[hanger.arm];
            let hook = rack.hook(hanger);
            let rail = arm.point(hanger.along_m);
            let (_, up) = arm.hook_frame();
            assert!(hook[4].distance(rail + up * (ARM_RADIUS + WIRE)) < 1e-6);
            if rack.form == RailForm::Waterfall {
                assert!(
                    arm.end_m.y < arm.start_m.y
                        && arm.stops_m.iter().any(|&d| (arm.point(d).distance(rail)
                            - hardware::STOP_RADIUS
                            - WIRE)
                            .abs()
                            < 1e-6)
                );
            }
            let yaw = Quat::from_rotation_y(hanger.yaw_rad);
            let normal = yaw * Vec3::X;
            if side {
                assert!(normal.x.abs() > 0.86);
            } else {
                assert!(normal.z > 0.94);
            }
            swivel += usize::from(normal.x.abs().min(normal.z.abs()) > 0.12);
            let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
            for g in stock_meshes[hanger.collection].parts.values() {
                for &p in &g.positions {
                    let q = hook[0] + yaw * (Vec3::from_array(p) * hanger.size_grade);
                    lo = lo.min(q);
                    hi = hi.max(q);
                }
            }
            assert!(lo.y > 0.060 && hi.y < arm.start_m.y);
            let half = o.size.xz() * 0.5;
            assert!(
                lo.xz().cmpge(-half).all() && hi.xz().cmple(half).all(),
                "{seed} {lo:?} {hi:?} {:?}",
                o.size
            );
            for (other_lo, other_hi) in &bounds {
                assert!(
                    !lo.xz().cmplt(*other_hi + 0.001).all()
                        || !hi.xz().cmpgt(*other_lo - 0.001).all(),
                    "garments touch on {seed}"
                );
            }
            bounds.push((lo.xz(), hi.xz()));
        }
        assert!(!bounds.is_empty());
        if rack.form == RailForm::Waterfall {
            bare_stops += usize::from(
                rack.arms.iter().map(|a| a.stops_m.len()).sum::<usize>() > rack.hangers.len(),
            );
        }
        long_gaps += usize::from(
            rack.hangers
                .windows(2)
                .any(|p| p[0].arm == p[1].arm && p[1].along_m - p[0].along_m > 0.28),
        );
        if seed < 24 || (rack.hanger_stock == Surface::Metal && checked_wire < 20) {
            checked_wire += usize::from(rack.hanger_stock == Surface::Metal);
            let audit = audit::check(&o, &build_object(&o)).unwrap();
            assert!(
                audit.min_y_m.abs() < 0.001
                    && audit.max_envelope_overrun_m < 0.001
                    && audit.batches <= 16
            );
        }
    }
    assert!(checked_wire >= 15);
    assert!(forms.into_iter().all(|n| n >= 10) && swivel > 25 && bare_stops > 3 && long_gaps > 8);
}

#[test]
fn display_rail_replay_preserves_old_racks_folded_stock_and_grocery_shelves() {
    for kind in [ObjectKind::GroceryDisplay, ObjectKind::ApparelDisplay] {
        for seed in 0..24 {
            let o = audit::fixture(kind, seed);
            assert!(apparel_rail(&o, DISPLAY_RAIL_VERSION - 1).is_none());
            if let Some(rack) = apparel_rail(&o, DISPLAY_RAIL_VERSION) {
                let copy: IndoorObject =
                    serde_json::from_slice(&serde_json::to_vec(&o).unwrap()).unwrap();
                let again = apparel_rail(&copy, DISPLAY_RAIL_VERSION).unwrap();
                assert_eq!(
                    serde_json::to_value(rack).unwrap(),
                    serde_json::to_value(again).unwrap()
                );
                let Installation::Rail {
                    hangers,
                    length,
                    frame,
                } = parameters(&o)
                else {
                    panic!()
                };
                let mut reference = Assembly::default();
                legacy_rail(
                    &mut reference,
                    o.size,
                    hangers,
                    length,
                    frame,
                    &mut stream(seed, u64::from_le_bytes(*b"ShopLoad")),
                );
                let mut old = Assembly::default();
                crate::scene::procedural_indoor::objects::public::retail::build(
                    &mut old,
                    &o,
                    DISPLAY_RAIL_VERSION - 1,
                );
                assert_eq!(reference.parts.len(), old.parts.len());
                for (k, g) in &old.parts {
                    assert_eq!(g.positions, reference.parts[k].positions);
                    assert_eq!(g.indices, reference.parts[k].indices);
                }
            } else {
                let old = build_object_at_version(&o, None, DISPLAY_RAIL_VERSION - 1);
                let rail_release = build_object_at_version(&o, None, DISPLAY_RAIL_VERSION);
                assert_eq!(rail_release.parts.len(), old.parts.len());
                for (k, g) in &rail_release.parts {
                    let prev = &old.parts[k];
                    assert_eq!(g.positions, prev.positions);
                    assert_eq!(g.normals, prev.normals);
                    assert_eq!(g.uvs, prev.uvs);
                    assert_eq!(g.indices, prev.indices);
                }
            }
        }
    }
}

#[test]
fn sleeve_dyes_reassign_existing_sewn_faces_including_lining_and_cuffs() {
    fn triangles(mesh: &Assembly) -> Vec<[[u32; 8]; 3]> {
        let mut out = Vec::new();
        for ((_, name), g) in &mesh.parts {
            assert_eq!(
                crate::scene::procedural_indoor::objects::part_label(name),
                "clothes"
            );
            for f in g.indices.as_chunks::<3>().0 {
                out.push(f.map(|i| {
                    let i = i as usize;
                    let [x, y, z] = g.positions[i];
                    let [nx, ny, nz] = g.normals[i];
                    let [u, v] = g.uvs[i];
                    [x, y, z, nx, ny, nz, u, v].map(f32::to_bits)
                }));
            }
        }
        out.sort_unstable();
        out
    }
    let mut tested = 0;
    for seed in 0..64 {
        let o = audit::fixture(ObjectKind::ApparelDisplay, seed);
        let Some(rack) = apparel_rail(&o, DISPLAY_RAIL_VERSION) else {
            continue;
        };
        for line in rack.collections {
            let Some(slot) = line.sleeve_finish_slot else {
                continue;
            };
            assert_ne!(slot, line.cut.finish_slot);
            let mut solid = Assembly::default();
            line.cut.build(&mut solid);
            let mut contrast = Assembly::default();
            line.build(&mut contrast);
            assert_eq!(triangles(&solid), triangles(&contrast));
            let sleeves = &contrast.parts[&(line.cut.surface, format!("clothes#finish{slot}"))];
            assert!(sleeves
                .positions
                .iter()
                .all(|p| p[2].abs() > line.cut.shoulder_width_m * 0.33));
            tested += 1;
        }
    }
    assert!(tested >= 8);
}
