use super::{test_geometry::top_surface_at, *};
use crate::scene::procedural_indoor::{
    materials::{finish_slot, registered_finish},
    objects::{audit, build_object, part_label},
};
use std::collections::BTreeSet;

#[test]
fn top_down_rays_distinguish_scored_metal_from_actual_empty_and_filled_open_cans() {
    let mut cases = [0; 3];
    let mut score_areas_mm2 = Vec::new();
    for seed in 0..384 {
        let mut o = audit::fixture(ObjectKind::SodaCan, seed);
        o.size = match seed % 3 {
            0 => Vec3::new(0.051, 0.089, 0.051),
            1 => Vec3::new(0.061, 0.170, 0.061),
            _ => o.size,
        };
        let p = can::parameters(&o);
        let a = build_object(&o);
        audit::check(&o, &a).unwrap_or_else(|e| panic!("{e}"));
        let h = o.size.y * p.height_fraction;
        let neck = o.size.x.min(o.size.z) * p.radius_fraction * p.neck;
        let panel = neck * p.end.panel_fraction;
        let deck_y = h - neck * p.end.recess_fraction;
        let half = p.end.port_half_fraction;
        score_areas_mm2.push(PI * panel.powi(2) * half.x * half.y * 1e6);
        assert!(half.x > half.y);
        for i in 0..64 {
            let (sin, cos) = (TAU * i as f32 / 64.).sin_cos();
            let q = Vec2::new(half.x * cos, half.y * sin + p.end.port_offset_fraction);
            assert!(
                q.length() < 0.96 && q.y > 0.064,
                "score intersects seam or rivet web"
            );
        }
        let tap = |x: f32, z: f32| {
            let q = Quat::from_rotation_y(p.lid_yaw) * (Vec3::new(x, 0., z) * panel);
            top_surface_at(&a, q.xz()).expect("closed can base")
        };
        let (pour_y, surface) = tap(
            0.013,
            p.end.port_offset_fraction + 0.70 * p.end.port_half_fraction.y,
        );
        let (target, expected, case) = match (p.opened, p.end.fill_fraction) {
            (false, None) => (deck_y - 0.07 * p.end.gauge_m, Surface::Metal, 0),
            (true, None) => (h * 0.036, Surface::Metal, 1),
            (true, Some(fill)) => (h * fill, Surface::Drink, 2),
            (false, Some(_)) => panic!("sealed contents are invisible"),
        };
        cases[case] += 1;
        assert!(
            (pour_y - target).abs() < 2e-6 && surface == expected,
            "seed {seed}, pour {pour_y} {surface:?}, expected {target}"
        );
        let tab_len = 0.16 + 0.83 * p.tab_length;
        let (finger_y, _) = tap(0.007, -0.13 - 0.27 * tab_len);
        assert!(
            (finger_y - deck_y).abs() < 2e-6,
            "finger hole blocked on {seed}"
        );
        assert!(
            tap(0., 0.035).0 > deck_y + 0.0002,
            "missing rivet on {seed}"
        );
        assert!(a.bounds().0.y.abs() < 1e-6 && a.bounds().1.y <= h + 1e-6);
    }
    assert!(cases.iter().all(|&n| n > 24), "{cases:?}");
    assert!(score_areas_mm2.iter().any(|&a| a < 180.) && score_areas_mm2.iter().any(|&a| a > 315.));
}

#[test]
fn label_edges_are_welded_to_the_body_and_one_revolution_spans_one_print() {
    let mut printed = [0; 2];
    for seed in 0..128 {
        for (i, kind) in [ObjectKind::WaterBottle, ObjectKind::SodaCan]
            .into_iter()
            .enumerate()
        {
            let o = audit::fixture(kind, seed);
            let a = build_object(&o);
            assert!(a
                .parts
                .keys()
                .all(|(_, name)| part_label(name) == kind.class_name()));
            let expected = if i == 0 {
                parameters(&o).style < 2
            } else {
                !can::parameters(&o).bare
            };
            let wrap = a.parts.iter().find(|((_, l), _)| l.ends_with("#uv_local"));
            assert_eq!(wrap.is_some(), expected);
            let Some(((surface, label), g)) = wrap else {
                continue;
            };
            printed[i] += 1;
            assert!(registered_finish(
                *surface,
                finish_slot(*surface, label).unwrap()
            ));
            let body = a
                .parts
                .iter()
                .find(|((s, label), _)| {
                    if i == 0 {
                        *s == Surface::ContainerGlass
                    } else {
                        label == crate::scene::procedural_indoor::materials::package_print::ALUMINUM
                    }
                })
                .unwrap()
                .1;
            assert_eq!(g.uvs.len() % 33, 0);
            for row in g.uvs.chunks_exact(33) {
                assert!((row[0][0] - row[32][0] - 1.).abs() < 2e-7);
                assert!(row
                    .windows(2)
                    .all(|w| w[0][0] > w[1][0] && w[0][1] == w[1][1]));
            }
            for (uv, pos) in g
                .uvs
                .iter()
                .zip(&g.positions)
                .filter(|(uv, _)| uv[1] == 0. || uv[1] == 1.)
            {
                assert!(
                    body.positions.contains(pos),
                    "unwelded label edge {seed} {kind:?} {uv:?}"
                );
            }
            assert_eq!(g.uvs.first().unwrap()[1], 1.);
            assert_eq!(g.uvs.last().unwrap()[1], 0.);
        }
    }
    assert!(printed[0] > 24 && printed[1] > 72);
}

#[test]
fn bottle_cap_and_fill_fit_the_sampled_profile_including_ribs_and_shoulders() {
    let mut flutes = BTreeSet::new();
    let mut submerged_shoulders = 0;
    for seed in 0..256 {
        let o = audit::fixture(ObjectKind::WaterBottle, seed);
        let p = parameters(&o);
        let c = bottle::Closure::sample(&o, &p);
        let a = build_object(&o);
        audit::check(&o, &a).unwrap_or_else(|e| panic!("{e}"));
        assert!(a.bounds().0.y.abs() < 1e-6 && a.bounds().1.y <= o.size.y + 1e-6);
        flutes.insert(c.flutes);
        let closure = &a.parts[&(
            Surface::Plastic,
            crate::scene::procedural_indoor::materials::package_print::cap(&o).into(),
        )];
        let ymax = if p.style == 5 { 0.938 } else { 0.993 } * o.size.y;
        let grip_y = ymax - (1. - 0.09) * c.cap_height_fraction * o.size.y;
        let grip: Vec<_> = closure
            .positions
            .iter()
            .map(|&v| Vec3::from_array(v))
            .filter(|v| (v.y - grip_y).abs() < 1e-6)
            .collect();
        assert_eq!(grip.len(), (c.flutes * 4 + 1) as usize);
        let span = grip
            .iter()
            .map(|v| v.xz().length())
            .fold((f32::MAX, 0_f32), |(lo, hi), r| (lo.min(r), hi.max(r)));
        assert!(((1. - span.0 / span.1) - c.grip_depth).abs() < 2e-5);
        let liquid = a.parts.iter().find(|((s, _), _)| *s == Surface::Liquid);
        assert_eq!(liquid.is_some(), p.style <= 2 && !p.empty);
        let Some((_, water)) = liquid else { continue };
        let level = o.size.y * p.fill_fraction;
        let wet = water
            .positions
            .iter()
            .map(|&v| Vec3::from_array(v))
            .filter(|v| (v.y - level).abs() < 1e-6)
            .map(|v| v.xz().length())
            .fold(0., f32::max);
        let glass = &a.parts[&(Surface::ContainerGlass, o.kind.class_name().into())];
        // Intersect the +X meridian of the actual inner wall at the measured fill line.
        let wall = super::test_geometry::triangles(glass)
            .flat_map(|tri| [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])])
            .filter(|(u, v)| {
                u.z.abs() < 1e-7
                    && v.z.abs() < 1e-7
                    && u.x > 0.
                    && v.x > 0.
                    && (u.y <= level) != (v.y <= level)
            })
            .map(|(u, v)| u.lerp(v, (level - u.y) / (v.y - u.y)).x)
            .fold(f32::INFINITY, f32::min);
        assert!(
            (0.00001..0.00032).contains(&(wall - wet)),
            "seed {seed}: wall {wall}, water {wet}"
        );
        submerged_shoulders += usize::from(p.fill_fraction > p.shoulder_fraction);
    }
    assert!(flutes.len() > 20 && submerged_shoulders > 4);
}
