use super::*;
use crate::scene::procedural_indoor::{
    envelope::ray_triangle,
    materials,
    objects::{audit, build_object_at_version, part_label},
};

fn distance(a: &Assembly, eye: Vec3, ray: Vec3) -> Option<f32> {
    a.parts
        .values()
        .flat_map(|g| {
            g.indices.as_chunks::<3>().0.iter().filter_map(move |tri| {
                ray_triangle(
                    eye,
                    ray,
                    tri.map(|idx| Vec3::from_array(g.positions[idx as usize])),
                )
            })
        })
        .min_by(f32::total_cmp)
}

#[test]
fn produce_stock_and_crate_bearings_fit_narrow_and_full_size_stands() {
    let mut bases = [0; 3];
    let mut styles = [0; 3];
    let mut layouts = [0; 3];
    let (mut empty, mut full, mut shared, mut offset_rows) = (0, 0, 0, 0);
    for seed in 0..64 {
        let mut o = audit::fixture(ObjectKind::GroceryDisplay, seed);
        for size in [Vec3::new(1., 0.81, 0.68), Vec3::new(2.28, 1.07, 0.98)] {
            o.size = size;
            let stand = produce_stand(&o, VERSION).unwrap();
            bases[stand.base as usize] += 1;
            layouts
                [usize::from(stand.row_split_m.is_some()) + usize::from(stand.rear_rise_m > 0.)] +=
                1;
            let mut frame = Assembly::default();
            let mut cuts = TimberCuts::new(o.seed, 0);
            stand.frame(&mut frame, &o, Surface::Wood, &mut cuts);
            let (low, high) = frame.bounds();
            assert!(low.y.abs() < 2e-6 && high.y <= o.size.y);
            if let Some(split) = stand.row_split_m {
                let edges: Vec<_> = stand
                    .bins
                    .iter()
                    .filter(|b| b.base_m.z > split)
                    .map(|b| b.base_m.x + b.size_xz_m.x * 0.5)
                    .collect();
                offset_rows +=
                    usize::from(stand.bins.iter().filter(|b| b.base_m.z < split).any(|b| {
                        edges
                            .iter()
                            .all(|x| (x - b.base_m.x - b.size_xz_m.x * 0.5).abs() > 0.02)
                    }));
            }
            let lot_labels = format!("{FIXTURE}#finish{}", stand.resin_slot);
            for (i, bin) in stand.bins.iter().enumerate() {
                styles[bin.style as usize] += 1;
                empty += usize::from(bin.stock.is_empty());
                full += usize::from(bin.stock.len() >= 12);
                shared += usize::from(stand.bins[..i].iter().any(|other| other.lot == bin.lot));
                let half = bin.size_xz_m * 0.5;
                assert!(bin.size_xz_m.cmplt(Vec2::new(0.98, 0.85)).all());
                if bin.style == CrateStyle::VentedTote {
                    assert!(bin.size_xz_m.cmplt(Vec2::new(0.83, 0.65)).all());
                }
                assert!((bin.base_m.xz().abs() + half).cmplt(size.xz() * 0.5).all());
                assert!(bin.base_m.y + bin.rim_height_m < size.y);
                assert!(stand.bins[..i].iter().all(|other| {
                    (other.base_m.xz() - bin.base_m.xz())
                        .abs()
                        .cmpge((other.size_xz_m + bin.size_xz_m) * 0.5)
                        .any()
                }));
                for frac in [-0.95, 0.95] {
                    let eye = bin.base_m + Vec3::new(half.x * frac, 0.01, half.y * frac);
                    assert!((distance(&frame, eye, Vec3::NEG_Y).unwrap() - 0.01).abs() < 2e-5);
                }
                let mut shell = Assembly::default();
                bin.crate_mesh(&mut shell, &mut cuts, &lot_labels);
                for (surface, key) in shell.parts.keys() {
                    assert_eq!(part_label(key), FIXTURE);
                    if bin.style == CrateStyle::VentedTote {
                        assert_eq!(
                            materials::finish_slot(*surface, key),
                            Some(stand.resin_slot)
                        );
                        assert!(materials::registered_finish(*surface, stand.resin_slot));
                    }
                }
                for (j, piece) in bin.stock.iter().enumerate() {
                    let fruit = &stand.lots[bin.lot][piece.shape];
                    let radius = fruit.clearance_radius();
                    assert!(bin.floor_m() + fruit.total_height() <= size.y);
                    assert!(bin.stock[..j]
                        .iter()
                        .all(|q| piece.position_xz_m.distance(q.position_xz_m)
                            > radius + stand.lots[bin.lot][q.shape].clearance_radius()));
                    for dir in [Vec2::ZERO, Vec2::X, Vec2::NEG_X, Vec2::Y, Vec2::NEG_Y] {
                        let p = piece.position_xz_m + dir * radius;
                        let eye = bin.base_m + Vec3::new(p.x, 0.24, p.y);
                        let top = eye.y - distance(&shell, eye, Vec3::NEG_Y).unwrap();
                        assert!(
                            (top - bin.floor_m()).abs() < 2e-5,
                            "seed {seed}: {top}, {bin:?}"
                        );
                    }
                }
            }
        }
    }
    assert!(bases.iter().chain(&layouts).all(|&v| v >= 8) && styles.iter().all(|&n| n > 30));
    assert!(empty >= 10 && full > 25 && shared > 60 && offset_rows > 30);
}

#[test]
fn produce_vents_and_handles_are_open_and_other_displays_replay() {
    let mut styles = [0; 3];
    let mut other = [0; 2];
    for seed in 0..60 {
        let o = audit::fixture(ObjectKind::GroceryDisplay, seed);
        let old = build_object_at_version(&o, None, VERSION - 1);
        let current = build_object_at_version(&o, None, VERSION);
        audit::check(&o, &current).unwrap();
        let Some(stand) = produce_stand(&o, VERSION) else {
            other[usize::from(is_refrigerated(&o))] += 1;
            assert_eq!(old.parts.len(), current.parts.len());
            for (key, g) in old.parts {
                let c = &current.parts[&key];
                assert_eq!(g.positions, c.positions);
                assert_eq!(g.normals, c.normals);
                assert_eq!(g.indices, c.indices);
                assert_eq!(g.uvs, c.uvs);
            }
            continue;
        };
        assert!(produce_stand(&o, VERSION - 1).is_none());
        // Actual end-panel opening between the two panels/cleats. Above it is the
        // continuous handle or wooden slat. The formed sheet has an unperforated wall.
        for bin in &stand.bins {
            let mut mesh = Assembly::default();
            bin.crate_mesh(&mut mesh, &mut TimberCuts::new(seed, 0), FIXTURE);
            styles[bin.style as usize] += 1;
            let hit = |h: f32| {
                distance(
                    &mesh,
                    bin.base_m + Vec3::new(bin.size_xz_m.x * 0.5 + 0.02, h * bin.rim_height_m, 0.),
                    Vec3::NEG_X,
                )
                .is_some()
            };
            assert_eq!(hit(0.52), bin.style == CrateStyle::FormedTray);
            assert!(hit(0.85));
        }
    }
    assert!(styles.iter().all(|&n| n > 0) && other.iter().all(|&n| n > 0));
}

#[test]
fn produce_refinishing_preserves_stock_positions_and_geometry() {
    let mut hits = 0;
    for seed in 0..32 {
        let mut o = audit::fixture(ObjectKind::GroceryDisplay, seed);
        o.size = Vec3::new(1.72, 0.95, 0.85);
        let first = produce_stand(&o, VERSION).unwrap();
        o.finish_seed = Some(seed.wrapping_add(0xa91c));
        let changed = produce_stand(&o, VERSION).unwrap();
        assert_eq!(first.bins.len(), changed.bins.len());
        for (a, b) in first.bins.iter().zip(&changed.bins) {
            assert_eq!(
                serde_json::to_value(a).unwrap(),
                serde_json::to_value(b).unwrap()
            );
        }
        for (a, b) in first
            .lots
            .iter()
            .flatten()
            .zip(changed.lots.iter().flatten())
        {
            assert_eq!(
                (a.axes_m, a.height_m, a.yaw, a.lean_m),
                (b.axes_m, b.height_m, b.yaw, b.lean_m)
            );
        }
        hits += usize::from(first.resin_slot != changed.resin_slot);
    }
    assert!(hits >= 12);
}
