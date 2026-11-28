use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object};
use std::collections::BTreeSet;

#[test]
fn manufactured_mats_and_racks_fit_their_catalog_envelopes_and_replay() {
    let mut panels = BTreeSet::new();
    let mut finishes = BTreeSet::new();
    let mut fills = BTreeSet::new();
    for kind in [ObjectKind::ExerciseMat, ObjectKind::ExerciseMatRack] {
        for seed in 0..96 {
            let mut o = audit::fixture(kind, seed);
            o.finish_seed = Some(seed ^ 391);
            let a = build_object(&o);
            let check = audit::check(&o, &a).unwrap_or_else(|e| panic!("{kind:?}/{seed}: {e}"));
            assert!(check.min_y_m.abs() < 0.001 && check.max_envelope_overrun_m < 0.001);
            assert!(check.batches <= 8 && check.triangles < 11000);
            if kind == ObjectKind::ExerciseMat {
                let m = PracticeMat::for_object(&o);
                panels.insert(m.panels);
                finishes.insert(m.facing.key().1);
                let face = &a.parts[&(m.facing.key().0, m.facing.key().1.to_string())];
                assert!(face
                    .positions
                    .iter()
                    .all(|p| (p[1] - o.size.y).abs() < 1e-6));
                assert!(face.normals.iter().all(|n| n[1] > 0.999));
                if m.panels > 1 {
                    assert!((0.002..0.014).contains(&m.hinge_gap_m));
                }
            } else {
                let rack = MatRack::for_object(&o);
                fills.insert(rack.rolls.len());
                assert!(!rack.rolls.is_empty());
                for (i, r) in rack.rolls.iter().enumerate() {
                    let shelf =
                        (r.center.y - r.radius - 0.082) * rack.tiers as f32 / (o.size.y - 0.082);
                    assert!((shelf - shelf.round()).abs() < 0.0001 && r.bore < r.radius);
                    for other in &rack.rolls[..i] {
                        assert!(
                            r.center.xy().distance(other.center.xy()) >= r.radius + other.radius
                        );
                    }
                }
            }
            if seed < 3 {
                let replay = build_object(&o);
                for (k, g) in &a.parts {
                    let again = &replay.parts[k];
                    assert_eq!(
                        (&g.positions, &g.uvs, &g.indices),
                        (&again.positions, &again.uvs, &again.indices)
                    );
                }
            }
        }
    }
    assert_eq!((panels.len(), finishes.len()), (5, 5));
    assert!(fills.len() >= 8);
}

#[test]
fn practice_mat_hinge_has_one_recessed_contact_surface_and_archived_mats_replay() {
    use crate::scene::procedural_indoor::objects::build_object_at_version;
    let o = (0..100)
        .map(|seed| audit::fixture(ObjectKind::ExerciseMat, seed))
        .find(|o| PracticeMat::for_object(o).panels > 1)
        .unwrap();
    let spec = PracticeMat::for_object(&o);
    let panel = (o.size.z - (spec.panels - 1) as f32 * spec.hinge_gap_m) / spec.panels as f32;
    let z = -o.size.z * 0.5 + panel + spec.hinge_gap_m * 0.5;
    let sample = Vec2::new(0.041, z);
    let intersections = |a: &Assembly| {
        let mut h = Vec::new();
        for g in a.parts.values() {
            for tri in g.indices.as_chunks::<3>().0 {
                let [p, q, r] =
                    std::array::from_fn(|i| Vec3::from_array(g.positions[tri[i] as usize]));
                if (q - p).cross(r - p).y <= 0. {
                    continue;
                }
                let (a, b, t) = ((q - p).xz(), (r - p).xz(), sample - p.xz());
                let denom = a.perp_dot(b);
                let (u, v) = (t.perp_dot(b) / denom, a.perp_dot(t) / denom);
                if u >= 0. && v >= 0. && u + v <= 1. {
                    h.push(p.y + u * (q.y - p.y) + v * (r.y - p.y));
                }
            }
        }
        h
    };
    let h = intersections(&build_object(&o));
    assert_eq!(h.len(), 1);
    assert!((h[0] / o.size.y - 0.485).abs() < 0.002);
    let old = build_object_at_version(&o, None, VERSION - 1);
    assert!(!old.parts.keys().any(|(_, k)| k.contains("#uv_local")));
    assert_eq!(intersections(&old), [o.size.y]);
}

#[test]
fn practice_rack_open_bores_and_shared_edge_charts_follow_roll_dimensions() {
    use crate::scene::procedural_indoor::materials::mat_stock::CHART_TURNS;
    for seed in 0..32 {
        let o = audit::fixture(ObjectKind::ExerciseMatRack, seed);
        let rack = MatRack::for_object(&o);
        let a = build_object(&o);
        let mut used = 0;
        for roll in rack.rolls {
            let (surface, _) = roll.stock.key();
            let end = &a.parts[&(surface, roll.stock.edge_label().into())];
            let front = roll.center.z + roll.length * 0.5;
            let points = end.positions.iter().zip(&end.uvs).filter(|(p, _)| {
                (Vec2::new(p[0], p[1]) - roll.center.xy()).length() <= roll.radius + 1e-5
                    && p[2] >= front - roll.recess_m - 1e-5
                    && p[2] <= front + 1e-5
            });
            let mut bore = 0;
            for (p, uv) in points {
                let offset = Vec2::new(p[0], p[1]) - roll.center.xy();
                let radius = offset.length();
                assert!(radius >= roll.bore - 1e-5);
                bore += usize::from((radius - roll.bore).abs() < 1e-5);
                let from_chart = (Vec2::from_array(*uv) - 0.5) * (2. * CHART_TURNS * roll.sheet_m);
                assert!(
                    from_chart.abs_diff_eq(offset, 1e-5)
                        && uv.iter().all(|f| (0.0..=1.0).contains(f))
                );
                used += 1;
            }
            assert!(bore >= 16);
        }
        assert!(used > 0);
    }
}
