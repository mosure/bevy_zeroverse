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
            basin::build(&mut a, basin, q.worktop_m, q.basin_finish);
            for v in [Vec2::ZERO, Vec2::new(0.35, 0.25), Vec2::new(-0.3, 0.15)] {
                let hit = super::super::tests::top_at(&a, basin.center + basin.floor_half() * v);
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
            assert!(
                dish.position.y
                    + dish.dish.height
                    + dish.dish.wall * ((dish.dish.count - 1) as f32)
                    < q.worktop_m
            );
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
