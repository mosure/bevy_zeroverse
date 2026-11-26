use super::*;
use crate::scene::procedural_indoor::{
    objects::{audit, build_object, part_label},
    placement::{rotated_half, Footprint},
};
use std::collections::BTreeSet;

#[test]
fn washing_stations_fit_actual_cutouts_support_props_and_vary_independently() {
    let mut widths = BTreeSet::new();
    let mut shapes = BTreeSet::new();
    let mut finishes = BTreeSet::new();
    let mut positions = BTreeSet::new();
    let mut unequal = 0;
    let mut accepted = 0;
    let mut faucet_positions = BTreeSet::new();
    let mut records = Vec::new();
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::KitchenSink, seed);
        o.finish_seed = Some(seed ^ 0x414243);
        let p = parameters(&o);
        let q = program::parameters(&o, &p);
        let mesh = build_object(&o);
        audit::check(&o, &mesh).unwrap_or_else(|e| panic!("sink {seed}: {e}"));
        assert_eq!(
            serde_json::to_value(&q).unwrap(),
            serde_json::to_value(program::parameters(&o, &parameters(&o))).unwrap()
        );
        finishes.insert(format!("{:?}/{:?}", q.basin_finish, q.faucet_finish));
        let mut worktop = Assembly::default();
        for ((surface, label), g) in &mesh.parts {
            if part_label(label) == "counter" {
                worktop.parts.insert((*surface, label.clone()), g.clone());
            }
        }
        let (d, z) = body(&o, &p);
        let mut occupied = Vec::new();
        for prop in &q.accessories {
            let half = rotated_half(prop.size.xz(), prop.yaw);
            let fp = Footprint {
                center: prop.position.xz(),
                half,
                yaw: 0.,
            };
            assert!(fp.inside(
                Vec2::new(-o.size.x * 0.5, z - d * 0.5),
                Vec2::new(o.size.x * 0.5, z + d * 0.5),
                0.011
            ));
            assert!(occupied.iter().all(|&other| !fp.overlaps(other, 0.0059)));
            occupied.push(fp);
            for sign in [
                Vec2::ONE,
                Vec2::new(1., -1.),
                Vec2::new(-1., 1.),
                -Vec2::ONE,
            ] {
                let point = prop.position.xz() + half * sign;
                let hit = super::super::tests::top_at(&worktop, point);
                assert!(
                    (hit - prop.position.y).abs() < 0.001,
                    "prop support {seed}/{:?}: {hit}, expected {}",
                    prop.kind,
                    prop.position.y
                );
            }
            positions.insert((
                (prop.position.x * 100.) as i32,
                (prop.position.z * 100.) as i32,
                (prop.yaw * 10.) as i32,
            ));
            accepted += 1;
        }
        for basin in &q.basins {
            widths.insert((basin.size.x * 100.) as i32);
            shapes.insert((
                (basin.roundness * 10.) as i32,
                (basin.taper.x * 100.) as i32,
            ));
            let mut a = Assembly::default();
            basin::build(&mut a, basin, q.worktop_m, q.basin_finish, [false; 2]);
            for sign in [
                Vec2::ONE,
                Vec2::new(-1., 1.),
                Vec2::NEG_ONE,
                Vec2::new(1., -1.),
            ] {
                let corner = basin.center + sign * (basin.size * 0.5 - Vec2::splat(0.001));
                let rim_height = super::super::tests::top_at(&a, corner);
                assert!(
                    (rim_height - q.worktop_m - 0.006).abs() < 0.0002,
                    "open rim corner in seed {seed}"
                );
            }
            for v in [Vec2::ZERO, Vec2::new(0.35, 0.25), Vec2::new(-0.3, 0.15)] {
                let point = basin.center + basin.floor_half() * v;
                if point.distance(basin.drain) <= basin.drain_radius {
                    continue;
                }
                let hit = super::super::tests::top_at(&a, point);
                assert!(
                    (hit - basin.floor_m).abs() < 0.001,
                    "basin closed/unsupported: {seed}: {hit}"
                );
            }
            assert!(basin.floor_m > 0.4 && q.worktop_m - basin.floor_m > 0.12);
        }
        if q.basins.len() == 2 {
            unequal += usize::from((q.basins[0].size.x - q.basins[1].size.x).abs() > 0.02);
        }
        let tip = q.faucet_points.last().unwrap();
        assert!(q
            .basins
            .iter()
            .any(|b| (tip.xz() - b.center).abs().cmplt(b.inner_half()).all()));
        faucet_positions.insert((
            (q.mount.x * 100.) as i32,
            ((tip.z - q.mount.z) * 100.) as i32,
        ));
        for dish in &q.dishes {
            let b = q
                .basins
                .iter()
                .find(|b| (dish.position.y - b.floor_m - 0.001).abs() < 1e-5)
                .unwrap();
            assert!(
                ((dish.position.xz() - b.center).abs() + dish.dish.half_extents(dish.yaw))
                    .cmple(b.floor_half() * 0.681)
                    .all()
            );
            assert!(dish.position.y + dish.dish.stack_height() < q.worktop_m);
        }
        records.push(serde_json::json!({"object":o,"program":q}));
    }
    assert!(widths.len() > 65 && shapes.len() > 200 && finishes.len() == 9 && unequal > 60);
    assert!(positions.len() > 220 && accepted > 250 && faucet_positions.len() > 150);
    println!("sink sampling: width_bins={} shape_bins={} finish_pairs={} unequal_pairs={unequal} accepted_props={accepted} pose_bins={} faucet_bins={}",widths.len(),shapes.len(),finishes.len(),positions.len(),faucet_positions.len());
    if let Some(path) = std::env::var_os("ZEROVERSE_KITCHEN_REPORT") {
        std::fs::write(
            std::path::Path::new(&path).join("washing-stations.json"),
            serde_json::to_vec(&records).unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn sink_finish_interventions_preserve_geometry_and_fitted_poses() {
    for seed in 0..32 {
        let mut o = audit::fixture(ObjectKind::KitchenSink, seed);
        let a = build_object(&o);
        let vertices = |a: &Assembly| {
            let mut v: Vec<_> = a
                .parts
                .values()
                .flat_map(|g| {
                    g.indices.iter().map(|&i| {
                        (
                            g.positions[i as usize].map(f32::to_bits),
                            g.normals[i as usize].map(f32::to_bits),
                        )
                    })
                })
                .collect();
            v.sort_unstable();
            v
        };
        o.finish_seed = Some(99999);
        let b = build_object(&o);
        assert!(vertices(&a) == vertices(&b), "finish moved sink {seed}");
    }
}

#[test]
fn waste_slots_and_aerators_are_recessed_above_closed_geometry() {
    use super::super::tests::top_at;
    let mut shapes = BTreeSet::new();
    let mut inserts = BTreeSet::new();
    let mut smallest = f32::INFINITY;
    let mut largest = 0_f32;
    for seed in 0..256 {
        let o = audit::fixture(ObjectKind::KitchenSink, seed);
        let cabinet = parameters(&o);
        let plan = program::parameters(&o, &cabinet);
        let tap = plan.tap.as_ref().unwrap();
        let outlet = tap.outlet(&plan);
        assert!(outlet.y > plan.worktop_m + 0.015, "outlet below rim {seed}");
        assert!(plan
            .faucet_points
            .iter()
            .all(|p| p.y + plan.faucet_radius_m < o.size.y));
        assert!(plan.basins.iter().any(|b| (outlet.xz() - b.center)
            .abs()
            .cmplt(b.inner_half() * 0.7)
            .all()));
        if matches!(tap.spout, tap::Spout::Gooseneck | tap::Spout::PullDown) {
            assert!(
                outlet.y > plan.worktop_m + 0.13,
                "tall tap clearance {seed}"
            );
        }
        let mut faucet = Assembly::default();
        tap.build(&mut faucet, &plan, cabinet.faucet_handles);
        // An upward ray through the mouth first hits the recessed insert. Reflecting
        // Y lets us reuse the vertical first-hit oracle used for worktops.
        for g in faucet.parts.values_mut() {
            for v in &mut g.positions {
                v[1] = -v[1];
            }
        }
        let recess = -top_at(&faucet, outlet.xz()) - outlet.y;
        assert!((0.001..0.008).contains(&recess), "{seed} aerator {recess}");
        assert!(faucet.bounds().1.y >= -plan.worktop_m - 1e-5);
        shapes.insert(format!("{:?}", tap.spout));

        let old = program::parameters_at(&o, &cabinet, FITTED_PLUMBING_VERSION - 1);
        assert!(old.tap.is_none() && old.basins.iter().all(|b| b.waste.is_none()));
        assert_eq!(old.mount, plan.mount);
        assert_eq!(old.handles, plan.handles);
        assert_eq!(
            serde_json::to_value(old.accessories).unwrap(),
            serde_json::to_value(&plan.accessories).unwrap()
        );
        let fitted = program::parameters_at(&o, &cabinet, FITTED_PLUMBING_VERSION);
        assert_eq!(
            serde_json::to_value(&old.dishes).unwrap(),
            serde_json::to_value(&fitted.dishes).unwrap()
        );
        // Later dishware revisions can reshape a bowl at the same basin location.
        let placement = |dish: &program::BasinDish| {
            (
                dish.seed,
                dish.position,
                dish.yaw,
                dish.dish.style,
                dish.dish.count,
            )
        };
        assert!(old
            .dishes
            .iter()
            .map(placement)
            .eq(plan.dishes.iter().map(placement)));
        for b in &plan.basins {
            let waste = b.waste.as_ref().unwrap();
            inserts.insert(format!("{:?}", waste.strainer));
            smallest = smallest.min(b.drain_radius * 2.);
            largest = largest.max(b.drain_radius * 2.);
            let mut shell = Assembly::default();
            basin::build(&mut shell, b, plan.worktop_m, plan.basin_finish, [false; 2]);
            let scan = |x| {
                let y = top_at(&shell, b.drain + x * b.drain_radius);
                assert!(
                    y.is_finite() && y <= b.floor_m + 1e-5,
                    "open shell {seed}: {y}"
                );
                (b.floor_m - y) / b.drain_radius
            };
            let mut deep = 0;
            let mut web = 0;
            for j in 0..64 {
                let angle = TAU * (j as f32 + 0.27) / 64.;
                let depth = scan(Vec2::new(angle.cos(), angle.sin()) * 0.45);
                deep += usize::from((depth - waste.depth_ratio).abs() < 0.001);
                web += usize::from(depth < 0.31);
            }
            assert!(deep > 4, "{seed}: sealed glass or metal over drain");
            assert_eq!(web > 0, waste.strainer != waste::Strainer::Open, "{seed}");
            // Every direction just outside the fitting still sees the supported bowl floor.
            for d in [Vec2::X, Vec2::NEG_X, Vec2::Y, Vec2::NEG_Y] {
                assert!(scan(d * 1.10).abs() < 0.001);
            }
        }
    }
    assert_eq!((shapes.len(), inserts.len()), (4, 3));
    assert!(smallest < 0.041 && largest > 0.096);
    println!("waste diameter {smallest:.3}..{largest:.3} m");
}

#[test]
fn joined_bowls_have_no_wall_inside_the_shared_lip() {
    for seed in 0..48 {
        let o = audit::fixture(ObjectKind::KitchenSink, seed);
        let p = parameters(&o);
        let plan = program::parameters(&o, &p);
        let [left, right] = &plan.basins[..] else {
            continue;
        };
        let seam = left.center.x + left.size.x * 0.5;
        assert!((right.center.x - right.size.x * 0.5 - seam).abs() < 1e-6);
        let mut shell = Assembly::default();
        for (b, shared) in [(left, [false, true]), (right, [true, false])] {
            basin::build(&mut shell, b, plan.worktop_m, plan.basin_finish, shared);
        }
        for g in shell.parts.values() {
            for tri in g.indices.as_chunks::<3>().0 {
                let v = tri.map(|i| Vec3::from(g.positions[i as usize]));
                assert!(!v.iter().all(|p| (p.x - seam).abs() < 1e-6));
            }
        }
        for t in [-0.99, -0.45, 0.0, 0.45, 0.99] {
            for dx in [-0.001, 0., 0.001] {
                let h = super::super::tests::top_at(
                    &shell,
                    Vec2::new(seam + dx, left.center.y + left.size.y * 0.5 * t),
                );
                assert!(
                    (h - plan.worktop_m - 0.006).abs() < 1e-5,
                    "gap at join {seed}"
                );
            }
        }
    }
}

#[test]
fn eccentric_circular_drains_stitch_to_wide_bowls_without_cracks_or_folded_faces() {
    use std::collections::BTreeMap;
    let fixture = audit::fixture(ObjectKind::KitchenSink, 9);
    let plan = program::parameters(&fixture, &parameters(&fixture));
    for aspect in [0.4, 1., 3., 8.] {
        for sx in [-1., 1.] {
            for sz in [-1., 1.] {
                let mut b = plan.basins[0].clone();
                b.size = Vec2::new(aspect * 0.4, 0.4);
                b.roundness = 7.75;
                b.drain = b.center + b.floor_half() * Vec2::new(sx * 0.45, sz * 0.44);
                b.drain_radius = b.floor_half().min_element() * 0.34;
                b.waste.as_mut().unwrap().strainer = waste::Strainer::Open;
                let mut mesh = Assembly::default();
                basin::build(&mut mesh, &b, plan.worktop_m, plan.basin_finish, [false; 2]);
                let mut edges = BTreeMap::new();
                let mut rim = BTreeSet::new();
                for g in mesh.parts.values() {
                    for tri in g.indices.as_chunks::<3>().0 {
                        let v = tri.map(|i| Vec3::from(g.positions[i as usize]));
                        let face = (v[1] - v[0]).cross(v[2] - v[0]);
                        let n: Vec3 = tri.iter().map(|&i| Vec3::from(g.normals[i as usize])).sum();
                        assert!(face.dot(n) > 0., "folded sector {aspect} {sx} {sz} {v:?}");
                        for p in v {
                            if (p.y - b.floor_m).abs() < 1e-6
                                && (p.xz().distance(b.drain) - b.drain_radius).abs() < 1e-6
                            {
                                rim.insert(p.to_array().map(f32::to_bits));
                            }
                        }
                        let v = v.map(|p| p.to_array().map(f32::to_bits));
                        for k in 0..3 {
                            let (u, v) = (v[k], v[(k + 1) % 3]);
                            let count = edges.entry((u.min(v), u.max(v))).or_insert((0_u32, 0_i32));
                            count.0 += 1;
                            count.1 += if u < v { 1 } else { -1 };
                        }
                    }
                }
                assert_eq!(rim.len(), 32);
                assert!(
                    edges.values().all(|&count| count == (2, 0)),
                    "unclosed bowl or fitting"
                );
            }
        }
    }
}
