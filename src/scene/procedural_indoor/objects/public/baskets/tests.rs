use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object, part_label};

#[test]
fn basket_constructions_fit_their_stack_and_keep_the_upper_cavity_open() {
    let mut forms = [0; 2];
    let mut carriers = [0; 3];
    let mut counts = [0; 7];
    let mut raised = 0;
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::BasketStack, seed);
        let p = BasketStack::for_object(&o);
        let mesh = build_object(&o);
        let check = audit::check(&o, &mesh).unwrap();
        assert!(check.triangles < 24000, "{seed} {}", check.triangles);
        let (lo, hi) = mesh.bounds();
        assert!(
            lo.y.abs() < 0.001
                && lo.x >= -o.size.x * 0.5 - 1e-5
                && hi.x <= o.size.x * 0.5 + 1e-5
                && lo.z >= -o.size.z * 0.5 - 1e-5
                && hi.z <= o.size.z * 0.5 + 1e-5
                && hi.y <= o.size.y + 1e-5,
            "{seed} {lo} {hi} vs {}",
            o.size
        );
        let separation = (p.rim_half_m - p.bottom_half_m) * p.nesting_pitch_m / p.body_height_m;
        assert!(separation.min_element() >= p.wall_m + 0.001);
        let mut fixture = false;
        for (surface, label) in mesh.parts.keys() {
            match part_label(label) {
                CLASS => (),
                CADDY => fixture = true,
                other => panic!("unexpected {other}"),
            };
            if let Some(slot) =
                crate::scene::procedural_indoor::materials::finish_slot(*surface, label)
            {
                assert!(
                    crate::scene::procedural_indoor::materials::registered_finish(*surface, slot),
                    "{surface:?} {slot}"
                );
            }
        }
        assert_eq!(fixture, p.carrier != Carrier::Floor);
        forms[p.shell as usize] += 1;
        carriers[p.carrier as usize] += 1;
        counts[p.count - 1] += 1;
        raised += usize::from(p.top_handle_angles_rad[0] > 0.);
        // A vertical ray inside the upper opening should hit the bottom or
        // pass between the welded base rods, including with latched handles.
        let start = p.bottom_m(p.count - 1);
        // A raised bail legitimately crosses the opening between the center
        // and the rim. Its two external pivot offsets leave the middle clear.
        let xz = p.bottom_half_m * Vec2::new(0.26, 0.);
        let mut top = f32::NEG_INFINITY;
        for g in mesh.parts.values() {
            for face in g.indices.as_chunks::<3>().0 {
                let [a, b, c] = face.map(|i| Vec3::from_array(g.positions[i as usize]));
                let (u, v) = (b.xz() - a.xz(), c.xz() - a.xz());
                let det = u.perp_dot(v);
                if det.abs() < 1e-10 {
                    continue;
                }
                let d = xz - a.xz();
                let (s, t) = (d.perp_dot(v) / det, u.perp_dot(d) / det);
                if s >= 0. && t >= 0. && s + t <= 1. {
                    top = top.max(a.y + s * (b.y - a.y) + t * (c.y - a.y));
                }
            }
        }
        assert!(top <= start + 0.012, "{seed}: top opening capped at {top}");
        if p.shell == BasketShell::Molded {
            assert!((top - start - p.wall_m).abs() < 1e-5);
        }
    }
    assert!(forms.into_iter().all(|v| v > 10) && carriers.into_iter().all(|v| v > 10));
    assert!(counts.into_iter().all(|v| v > 2) && (10..80).contains(&raised));
}

#[test]
fn molded_apertures_transmit_rays_between_opaque_ribs() {
    let mut tested = 0;
    for seed in 0..36 {
        let p = BasketStack::sample(seed, seed);
        if p.shell != BasketShell::Molded {
            continue;
        }
        let mut g = Geometry::default();
        shell::molded(&mut g, &p, 0.);
        // Cast in through +Z. Slots and ribs are centered along the flat face.
        // A solid box with a printed grid would fill every sample in this row.
        let hits = |x: f32, y: f32| {
            let q = Vec2::new(x, y);
            g.indices.as_chunks::<3>().0.iter().any(|face| {
                let [a, b, c] = face.map(|i| Vec3::from_array(g.positions[i as usize]));
                if a.z.min(b.z).min(c.z) < 0. {
                    return false;
                }
                let (u, v) = (b.xy() - a.xy(), c.xy() - a.xy());
                let det = u.perp_dot(v);
                if det.abs() < 1e-10 {
                    return false;
                }
                let delta = q - a.xy();
                let (s, t) = (delta.perp_dot(v) / det, u.perp_dot(delta) / det);
                s >= 0. && t >= 0. && s + t <= 1.
            })
        };
        for t in [0.31, 0.67] {
            let half = p.bottom_half_m.lerp(p.rim_half_m, t).x * (1. - p.chamfer_fraction);
            let n = p.openings[0];
            for k in 0..n {
                let hole = half * (2. * (k as f32 + 0.5) / n as f32 - 1.);
                // A ray at the center of a hole must pass through both skins.
                assert!(!hits(hole, t * p.body_height_m), "{seed} slot {k}");
                if k > 0 {
                    let rib = half * (2. * k as f32 / n as f32 - 1.);
                    assert!(hits(rib, t * p.body_height_m), "{seed} rib {k}");
                }
            }
        }
        tested += 1;
    }
    assert!(tested > 16);
}
