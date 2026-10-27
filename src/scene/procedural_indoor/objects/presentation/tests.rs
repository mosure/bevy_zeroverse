use super::*;
use crate::scene::procedural_indoor::placement::{rotated_half, Footprint};
use std::collections::BTreeSet;
mod qualification;

const KINDS: [ObjectKind; 3] = [
    ObjectKind::Whiteboard,
    ObjectKind::WallShelf,
    ObjectKind::ProjectionScreen,
];

#[test]
fn presentation_meshes_fit_independent_metric_extremes_and_keep_wall_datum() {
    for kind in KINDS {
        let sizes = match kind {
            ObjectKind::Whiteboard => [
                Vec3::new(0.7, 0.65, 0.04),
                Vec3::new(2.1, 0.65, 0.09),
                Vec3::new(0.7, 1.3, 0.12),
                Vec3::new(2.1, 1.3, 0.04),
            ],
            ObjectKind::WallShelf => [
                Vec3::new(0.42, 0.13, 0.17),
                Vec3::new(1.70, 0.13, 0.40),
                Vec3::new(0.42, 0.38, 0.40),
                Vec3::new(1.70, 0.38, 0.17),
            ],
            _ => [
                Vec3::new(1.0, 0.70, 0.05),
                Vec3::new(2.70, 0.70, 0.12),
                Vec3::new(1.0, 1.70, 0.12),
                Vec3::new(2.70, 1.70, 0.05),
            ],
        };
        for seed in 0..96 {
            for size in sizes {
                let mut o = audit::fixture(kind, seed);
                o.size = size;
                let a = build_object(&o);
                audit::check(&o, &a).unwrap_or_else(|e| panic!("{kind:?}/{seed}/{size:?}: {e}"));
                let (lo, hi) = a.bounds();
                assert!(
                    lo.cmpge(Vec3::new(-size.x * 0.5, 0., -size.z * 0.5) - Vec3::splat(1e-5))
                        .all(),
                    "{kind:?}/{seed}/{size:?}: {lo:?}"
                );
                assert!(
                    hi.cmple(Vec3::new(size.x * 0.5, size.y, size.z * 0.5) + Vec3::splat(1e-5))
                        .all(),
                    "{kind:?}/{seed}/{size:?}: {hi:?}"
                );
                assert!(
                    (lo.z + size.z * 0.5).abs() < 1e-5,
                    "detached from wall datum"
                );
                assert_eq!(audit::program(&o), audit::program(&o), "seed replay");
            }
        }
    }
}

#[test]
fn tray_accessories_are_supported_separated_and_not_equal_spaced_rows() {
    let (mut count, mut rotated, mut centered, mut empty, mut erasers) = (0, 0, 0, 0, 0);
    let mut positions = BTreeSet::new();
    for seed in 0..192 {
        let o = audit::fixture(ObjectKind::Whiteboard, seed);
        let p = parameters(&o);
        let mut occupied: Vec<Footprint> = Vec::new();
        assert_eq!(
            p.marker_count,
            p.contents.iter().filter(|v| !v.eraser).count()
        );
        empty += usize::from(p.contents.is_empty());
        if p.tray {
            assert_eq!(p.tray_lo.y, p.face_z_m, "tray must meet board");
        }
        for item in &p.contents {
            let q = Footprint {
                center: item.center,
                half: rotated_half(item.size.xz(), item.yaw),
                yaw: item.yaw,
            };
            assert!(q.inside(p.tray_lo, p.tray_hi, 0.004));
            assert!(!occupied.iter().any(|&a| a.overlaps(q, 0.0039)));
            occupied.push(q);
            let mut mesh = Assembly::default();
            contents::build_item(&mut mesh, &o, &p, item);
            let (lo, hi) = mesh.bounds();
            assert!(lo.x >= p.tray_lo.x + 0.004 && hi.x <= p.tray_hi.x - 0.004);
            assert!(lo.z >= p.tray_lo.y + 0.004 && hi.z <= p.tray_hi.y - 0.004);
            // Polygonal cylinders can rest within 0.1 mm of their analytic radius.
            assert!(
                (lo.y - p.tray_base_top_m).abs() < 0.0001,
                "floating item {seed}: {lo:?}"
            );
            positions.insert((
                (item.center.x * 1000.) as i32,
                (item.center.y * 1000.) as i32,
            ));
            let angle = (2. * item.yaw).sin().atan2((2. * item.yaw).cos()).abs() * 0.5;
            rotated += usize::from(angle > 3_f32.to_radians());
            centered +=
                usize::from((item.center.x - (p.tray_lo.x + p.tray_hi.x) * 0.5).abs() < 0.005);
            erasers += usize::from(item.eraser);
            count += 1;
        }
        for (i, m) in p.magnets.iter().enumerate() {
            assert!(m.center.x.abs() + m.radius_m + p.border_m < o.size.x * 0.5);
            assert!(
                m.center.y - m.radius_m > p.border_m
                    && m.center.y + m.radius_m < o.size.y - p.border_m
            );
            assert!(p.magnets[i + 1..]
                .iter()
                .all(|b| m.center.distance(b.center) > m.radius_m + b.radius_m));
        }
    }
    assert!(count > 200 && positions.len() > 150 && erasers > 10 && empty > 10);
    assert!(
        rotated * 3 > count && centered * 10 < count,
        "count={count} rotated={rotated} centered={centered}"
    );
}

#[test]
fn shelves_keep_full_top_support_and_vary_mounting_construction() {
    let mut styles = BTreeSet::new();
    let mut positions = BTreeSet::new();
    for seed in 0..128 {
        let mut o = audit::fixture(ObjectKind::WallShelf, seed);
        // Include long shelves to exercise three/four-support fields.
        o.size.x = if seed % 2 == 0 { 1.6 } else { 0.6 };
        let p = equipment::wall_shelf::parameters(&o);
        styles.insert(p.style);
        let a = build_object(&o);
        for x in [-0.48, 0., 0.48] {
            for z in [-0.48, 0., 0.48] {
                let origin = Vec3::new(x * o.size.x, o.size.y + 0.05, z * o.size.z);
                let (_, distance) = ray_hit(&a, origin, Vec3::NEG_Y).unwrap();
                assert!(
                    (distance - 0.05).abs() < 0.001,
                    "unsupported shelf top {seed}"
                );
            }
        }
        assert!(p.brace_radius_m <= p.strap_thickness_m * 0.5);
        assert!(p
            .bracket_x_m
            .windows(2)
            .all(|v| v[1] - v[0] > p.strap_width_m));
        for &x in &p.bracket_x_m {
            positions.insert((x / o.size.x * 1000.) as i32);
            assert!(x.abs() + p.strap_width_m * 0.5 < o.size.x * 0.5);
        }
    }
    assert_eq!(styles.len(), 4);
    assert!(positions.len() > 100);
}

#[test]
fn optical_targets_hit_the_actual_projection_sheet_in_all_screen_styles() {
    let mut framed = 0;
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::ProjectionScreen, seed);
        let p = projection::parameters(&o);
        framed += usize::from(p.framed);
        assert!(p.image_width_m() > o.size.x * 0.65);
        assert!(p.surface_hi.y > p.surface_lo.y && p.surface_lo.y >= p.sheet_bottom_m);
        let a = build_object(&o);
        // Center plus near-corner first hits must be paper, not a black border,
        // weighted bar, or casing. This independently checks built triangles.
        if !p.framed {
            let mouth = Vec3::new(
                p.sheet_center_x_m,
                o.size.y - p.case_height_m - 0.0005,
                p.sheet_front_m + 1.,
            );
            assert_eq!(
                ray_hit(&a, mouth, Vec3::NEG_Z).unwrap().0,
                Surface::Ink,
                "detached hanging sheet {seed}"
            );
            assert!(p.bar_width_m > p.sheet_width_m + 0.007);
        }
        for u in [0.01, 0.5, 0.99] {
            for v in [0.01, 0.5, 0.99] {
                let target = Vec3::new(
                    p.surface_lo.x + (p.surface_hi.x - p.surface_lo.x) * u,
                    p.surface_lo.y + (p.surface_hi.y - p.surface_lo.y) * v,
                    p.surface_lo.z,
                );
                let (surface, t) = ray_hit(&a, target + Vec3::Z, Vec3::NEG_Z).unwrap();
                assert_eq!(surface, Surface::Paper, "{seed}/{u}/{v}");
                assert!((t - 1.).abs() < 1e-5);
            }
        }
    }
    assert!(framed > 20 && framed < 80);
}

#[test]
fn presentation_finishes_are_independent_of_meshes_and_optical_targets() {
    for kind in KINDS {
        let mut finishes = BTreeSet::new();
        for seed in 0..48 {
            let mut o = audit::fixture(kind, seed);
            let geometry = audit::descriptor(&o, &build_object(&o));
            let target = (kind == ObjectKind::ProjectionScreen)
                .then(|| projection::parameters(&o).target_local());
            o.finish_seed = Some(seed + 1000);
            let a = build_object(&o);
            finishes.insert(format!("{:?}", a.parts.keys().collect::<Vec<_>>()));
            assert_eq!(geometry, audit::descriptor(&o, &a), "{kind:?}/{seed}");
            if let Some(target) = target {
                assert_eq!(target, projection::parameters(&o).target_local());
            }
        }
        assert!(finishes.len() >= 5, "{kind:?}: {}", finishes.len());
    }
}

fn ray_hit(a: &Assembly, origin: Vec3, direction: Vec3) -> Option<(Surface, f32)> {
    a.parts
        .iter()
        .flat_map(|((surface, _), g)| {
            g.indices.as_chunks::<3>().0.iter().filter_map(move |ids| {
                let p: [Vec3; 3] =
                    std::array::from_fn(|i| Vec3::from_array(g.positions[ids[i] as usize]));
                let e = p[1] - p[0];
                let f = p[2] - p[0];
                let h = direction.cross(f);
                let det = e.dot(h);
                if det.abs() < 1e-9 {
                    return None;
                }
                let s = origin - p[0];
                let u = s.dot(h) / det;
                let q = s.cross(e);
                let v = direction.dot(q) / det;
                let t = f.dot(q) / det;
                (u >= -1e-6 && v >= -1e-6 && u + v <= 1.000001 && t >= 0.).then_some((*surface, t))
            })
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
}
