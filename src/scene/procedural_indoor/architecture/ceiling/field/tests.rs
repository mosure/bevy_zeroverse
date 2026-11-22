use super::*;
use crate::scene::procedural_indoor::{architecture, geometry::Geometry};

fn sample(seed: u64) -> IndoorManifest {
    IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap()
}

fn first_hit(g: &Geometry, p: Vec2) -> Option<f32> {
    g.indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(|t| {
            let [a, b, c] = t.map(|i| Vec3::from_array(g.positions[i as usize]));
            let (ab, ac, ap) = (b.xz() - a.xz(), c.xz() - a.xz(), p - a.xz());
            let d = ab.perp_dot(ac);
            if d.abs() < 1e-10 {
                return None;
            }
            let (u, v) = (ap.perp_dot(ac) / d, ab.perp_dot(ap) / d);
            (u >= 0. && v >= 0. && u + v <= 1.).then_some(a.y + u * (b.y - a.y) + v * (c.y - a.y))
        })
        .min_by(f32::total_cmp)
}

#[test]
fn cut_ceiling_closes_a_concave_footprint_and_keeps_equipment_pockets_open() {
    let mut s = sample(6);
    s.room_size = Vec3::new(4.8, 3.6, 4.8);
    let outline = vec![
        Vec2::splat(-2.4),
        Vec2::new(1.8, -2.4),
        Vec2::new(2.4, -1.8),
        Vec2::new(2.4, 0.3),
        Vec2::new(1.23, 0.3),
        Vec2::new(1.23, 2.4),
        Vec2::new(-2.4, 2.4),
    ];
    s.envelope.as_mut().unwrap().footprint = outline.clone();
    let grid = CeilingGrid {
        pitch_m: Vec2::splat(0.6),
        bays: UVec2::splat(8),
        first_joint_m: Vec2::splat(-1.8),
        tile: CeilingTile::sample(23, 3),
        troffer_depth_m: 0.06,
        troffers: vec![RecessedLight {
            fixture: 0,
            cell: UVec2::new(2, 3),
        }],
        air_terminals: [AirPattern::FourWaySupply, AirPattern::SlottedReturn]
            .into_iter()
            .enumerate()
            .map(|(i, pattern)| AirTerminal {
                cell: UVec2::new(3 + i as u32, 4),
                pattern,
                blades: 4,
                angle_rad: 0.51,
                across_x: true,
            })
            .collect(),
    };
    let mut a = Assembly::default();
    bevy::tasks::block_on(build(&mut a, &s, &grid, || std::future::ready(())));
    let h = s.room_size.y;
    let hit = |p| {
        a.parts
            .values()
            .filter_map(|g| first_hit(g, p))
            .min_by(f32::total_cmp)
    };
    for z in 0..31 {
        for x in 0..33 {
            let p = Vec2::new(x as f32 * 0.147 - 2.3371, z as f32 * 0.155 - 2.3241);
            let y = hit(p);
            assert_eq!(polygon::contains(&outline, p, 0.), y.is_some(), "{p}");
            assert!(
                y.is_none_or(|v| (h - 0.015..=h + 0.097).contains(&v)),
                "{p} {y:?}"
            );
        }
    }
    // The mounting datum (including the beveled shoulder) and the first
    // visible roof triangle agree across an entire row of ordinary stock.
    for i in 0..263 {
        let p = grid.center(UVec2::new(3, 2)) + Vec2::new(i as f32 * 0.0031 - 0.41, 0.081);
        assert!(
            (grid.mount_height(&s, p) - hit(p).unwrap()).abs() < 2e-5,
            "{p}"
        );
    }
    let c = grid.center(grid.troffers[0].cell);
    assert!((hit(c).unwrap() - h - 0.095).abs() < 1e-5);
    let board = grid.center(UVec2::new(3, 2));
    assert!((hit(board).unwrap() - (h + 0.0036 - grid.tile.drop_m)).abs() < 1e-5);
    assert!((hit(board + Vec2::X * 0.299).unwrap() - h).abs() < 1e-5);
    for air in &grid.air_terminals {
        let center = grid.center(air.cell);
        assert!(grid
            .air_reservations(h)
            .any(|(lo, hi)| center.cmpgt(lo.xz()).all()
                && center.cmplt(hi.xz()).all()
                && lo.y < h - 0.006
                && hi.y > h));
        let heights: Vec<_> = (0..61)
            .map(|i| hit(center + Vec2::new(i as f32 * 0.0081 - 0.24, 0.041)).unwrap() - h)
            .collect();
        assert!(
            heights.iter().any(|y| *y < 0.05) && heights.iter().any(|y| *y > 0.09),
            "{:?}",
            air.pattern
        );
    }
}

#[test]
fn fitted_manifest_replays_and_preserves_the_circuit_geometry_and_semantics() {
    let group = [
        75009, 75021, 75029, 75037, 75047, 75050, 75056, 75090, 75094, 75108, 75115, 75116, 75128,
        75133, 75135, 75145,
    ];
    let s = group
        .into_iter()
        .map(sample)
        .find(|s| CeilingGrid::for_scene(s).is_some_and(|g| !g.troffers.is_empty()))
        .expect("sample includes a recessed panel circuit");
    let grid = CeilingGrid::for_scene(&s).unwrap();
    let field = s.ceiling_installation.as_ref().unwrap();
    field.validate(&s).unwrap();
    let restored: CeilingInstallation =
        serde_json::from_slice(&serde_json::to_vec(field).unwrap()).unwrap();
    assert_eq!(restored, *field);
    assert_eq!(CeilingInstallation::sample(&s), *field);
    let raw = architecture::fixture_layout::positions(&s);
    let fitted = architecture::fixture_positions(&s);
    assert_eq!(fitted.len(), raw.len());
    for (i, &p) in fitted.iter().enumerate() {
        let Some(m) = grid.troffers.iter().find(|m| m.fixture == i) else {
            assert_eq!(raw[i], p);
            continue;
        };
        assert_eq!(p.xz(), grid.center(m.cell));
        let half = architecture::fixture_half_extents(&s, p);
        assert_eq!(half, grid.lamp_size() * 0.5);
        assert!((p.y - half.y - s.room_size.y + 0.004).abs() < 1e-5);
        assert_eq!(architecture::fixtures::direction(&s, p), Vec3::NEG_Y);
        assert!(architecture::fixtures::emitter(&s, p).y < p.y - half.y);
        let label = format!("lamp#{i}");
        let a = architecture::fixtures::geometry(&s, i, p);
        let g = &a.parts[&(Surface::Light, label.clone())];
        assert!(g
            .positions
            .iter()
            .all(|q| Vec3::from_array(*q).xz().distance(p.xz()) < half.xz().length()));
        assert!(first_hit(g, p.xz()).is_some());
        assert!(a
            .parts
            .keys()
            .all(|(_, l)| l == &label || l.starts_with(&format!("{label}#finish"))));
    }
    let mut bad = grid.clone();
    bad.troffers.push(bad.troffers[0].clone());
    assert!(bad.validate(&s).is_err());
    let mut bad = grid.clone();
    bad.troffers[0].cell = UVec2::MAX;
    assert!(bad.validate(&s).is_err());
    let mut bad = grid.clone();
    bad.troffers[0].fixture = usize::MAX;
    assert!(bad.validate(&s).is_err());
    let old: CeilingInstallation =
        serde_json::from_str(r#"{"modules":[],"exposed_slab":false}"#).unwrap();
    assert!(old.room_grid.is_none());
    let mut previous = s;
    previous.generator_version = 157;
    assert!(CeilingInstallation::sample(&previous).room_grid.is_none());
}
