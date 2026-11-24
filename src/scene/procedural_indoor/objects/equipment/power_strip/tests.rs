use super::*;
use crate::scene::procedural_indoor::objects::{
    audit, build_object, equipment::receptacles, wall::OutletFace,
};

/// Nearest upward-facing surface under a vertical ray in local strip coordinates.
fn top(mesh: &Assembly, q: Vec2) -> f32 {
    let mut top = f32::NEG_INFINITY;
    for g in mesh.parts.values() {
        for tri in g.indices.as_chunks::<3>().0 {
            let [a, b, c] = tri.map(|ix| Vec3::from_array(g.positions[ix as usize]));
            let (e, f, d) = (b.xz() - a.xz(), c.xz() - a.xz(), q - a.xz());
            let area = e.perp_dot(f);
            if area >= -1e-12 {
                continue;
            }
            let (u, v) = (d.perp_dot(f) / area, e.perp_dot(d) / area);
            if u >= -1e-5 && v >= -1e-5 && u + v <= 1.00001 {
                top = top.max(a.y + u * (b.y - a.y) + v * (c.y - a.y));
            }
        }
    }
    top
}

#[test]
fn contact_rays_hit_the_pin_backings_and_the_routed_plug_mating_plane() {
    for seed in 0..96 {
        let mut o = audit::fixture(ObjectKind::PowerStrip, seed);
        o.size = dimensions(&mut stream(seed, 10701));
        o.position = Vec3::new(1.2, 0.74, -0.59);
        o.yaw = 0.43;
        let p = parameters(&o);
        let d = p.deck.unwrap();
        let mesh = build_object(&o);
        let rotation = Mat2::from_angle(p.socket_yaw);
        let pins: &[[f32; 2]] = match d.face {
            OutletFace::RoundWell => &[[-0.0093, 0.], [0.0093, 0.]],
            OutletFace::ThreePin => &[[-0.0111, -0.004], [0.0111, -0.004], [0., 0.010]],
            OutletFace::Polarized => &[[-0.00635, 0.003], [0.00635, 0.003], [0., -0.009]],
        };
        let round = d.face == OutletFace::RoundWell;
        let pin_depth = if round { 0.0016 } else { 0.0017 } * d.contact_scale;
        let bank = receptacles::MainsBank::for_target(&o).unwrap();
        assert_eq!(bank.socket_count(), p.socket_count);
        assert!(bank.local(o.size, p.socket_count).is_none());
        assert_eq!(receptacles::recessed_connection(&o, Some(0)), round);
        assert!(!receptacles::recessed_connection(&o, None));
        for i in 0..p.socket_count {
            let port = bank.local(o.size, i).unwrap();
            let world = receptacles::target_port(&o, false, Some(i as u8)).unwrap();
            assert!(world.distance(o.transform().transform_point(port)) < 1e-6);
            let trace = |point: Vec2| {
                let q = rotation * point;
                top(&mesh, port.xz() + Vec2::new(q.x, -q.y))
            };
            assert!(
                (trace(Vec2::ZERO) - port.y).abs() < 2e-6,
                "{seed}/{i}: mating plane"
            );
            for &pin in pins {
                assert!(
                    (trace(Vec2::from_array(pin) * d.contact_scale) - (port.y - pin_depth)).abs()
                        < 2e-6,
                    "{seed}/{i}: pin {pin:?}"
                );
            }
            if round {
                assert!(
                    (trace(Vec2::X * (0.0205 * d.contact_scale)) - port.y - d.well_depth_m).abs()
                        < 2e-6
                );
            }
            // Sample all eight sectors where a rotated insert meets its deck.
            for j in 0..8 {
                let angle = (j as f32 + 0.35) * PI * 0.25;
                let dir = Vec2::from_angle(angle);
                let from = (d.insert_width_m * 0.5) / dir.abs().max_element();
                let to = (d.cut_width_m * 0.5) / (rotation * dir).abs().max_element();
                assert!(
                    (trace(dir * ((from + to) * 0.5)) - o.size.y * p.height_fraction).abs() < 2e-6
                );
            }
        }
    }
}

#[test]
fn twin_cartridges_use_the_full_clear_deck_inside_the_bevels() {
    let mut o = audit::fixture(ObjectKind::PowerStrip, 8747063381748563241);
    o.variant = crate::scene::procedural_indoor::objects::wall::installed_variant(1);
    o.size = Vec3::new(0.27319732, 0.030902337, 0.11710332);
    let p = parameters(&o);
    let d = p.deck.unwrap();
    assert_eq!(p.rows, 2);
    let usable = o.size.z - 2. * o.size.y * p.height_fraction * p.bevel_fraction;
    assert!(d.row_pitch_m >= 0.047 * d.contact_scale && d.row_pitch_m + d.cut_width_m <= usable);
    assert!(d.row_pitch_m < o.size.z * 0.5);
}

#[test]
fn sampled_strip_cells_are_disjoint_bounded_and_finish_independent() {
    use crate::scene::procedural_indoor::objects::{audit, build_object};
    let mut configurations = std::collections::BTreeSet::new();
    let mut shapes = std::collections::BTreeSet::new();
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::PowerStrip, seed);
        o.size = dimensions(&mut stream(seed, 10701));
        let p = parameters(&o);
        configurations.insert((p.rows, p.end_caps, (p.socket_yaw * 100.) as i32));
        assert!(p.socket_count >= 2, "catalog strip {seed}");
        shapes.insert((
            (o.size.x * 1000.) as i32,
            (o.size.y * 1000.) as i32,
            (o.size.z * 1000.) as i32,
        ));
        let a = build_object(&o);
        audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed}: {e}"));
        let (lo, hi) = a.bounds();
        assert!(lo.y.abs() < 1e-6);
        assert!(hi.y <= o.size.y + 1e-6);
        for i in 0..p.socket_count {
            let center = p.socket_center(o.size, i);
            assert!(
                (center.abs() + Vec2::splat(p.deck.unwrap().cut_width_m * 0.5))
                    .cmplt(o.size.xz() * 0.5)
                    .all()
            );
            for j in 0..i {
                assert!(
                    (center - p.socket_center(o.size, j)).length() > p.deck.unwrap().cut_width_m
                );
            }
        }
        let replay = build_object(&o);
        for (key, g) in &a.parts {
            assert_eq!(g.positions, replay.parts[key].positions);
            assert_eq!(g.indices, replay.parts[key].indices);
        }
        o.finish_seed = Some(888);
        let alt = parameters(&o);
        assert_eq!(p.socket_count, alt.socket_count);
        assert_eq!(p.output_port(o.size, false), alt.output_port(o.size, false));
        assert_eq!(p.inlet(o.size), alt.inlet(o.size));
        assert_eq!(p.socket_yaw, alt.socket_yaw);
    }
    // The broadest catalog stock can fit diagonally mounted modules on two
    // rows. Check the joint support explicitly alongside random stock.
    for seed in 0..128 {
        let mut o = audit::fixture(ObjectKind::PowerStrip, seed);
        o.size = Vec3::new(0.29, 0.030, 0.129);
        let p = parameters(&o);
        configurations.insert((p.rows, p.end_caps, (p.socket_yaw * 100.) as i32));
    }
    assert_eq!(configurations.len(), 16);
    assert!(shapes.len() > 240);
}

#[test]
fn archived_strip_endpoints_keep_the_raised_deck_offset() {
    for seed in 0..128 {
        let mut o =
            crate::scene::procedural_indoor::objects::audit::fixture(ObjectKind::PowerStrip, seed);
        o.variant = (seed % 4) as u32;
        let p = parameters(&o);
        assert!(p.deck.is_none());
        let mut a = Assembly::default();
        build(&mut a, &o);
        let port = p.output_port(o.size, false);
        let q = port.xz();
        let mut top = f32::NEG_INFINITY;
        for g in a.parts.values() {
            for tri in g.indices.as_chunks::<3>().0 {
                let v = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                let area = (v[1].xz() - v[0].xz()).perp_dot(v[2].xz() - v[0].xz());
                if area.abs() < 1e-10 {
                    continue;
                }
                let u = (v[1].xz() - q).perp_dot(v[2].xz() - q) / area;
                let w = (v[2].xz() - q).perp_dot(v[0].xz() - q) / area;
                let t = 1. - u - w;
                if u >= -1e-5 && w >= -1e-5 && t >= -1e-5 {
                    top = top.max(u * v[0].y + w * v[1].y + t * v[2].y);
                }
            }
        }
        assert!(
            (0.0003..0.0021).contains(&(port.y - top)),
            "{seed}: port={port:?}, mesh={top}"
        );
        assert_eq!(
            super::super::power_port(&o, false).unwrap(),
            o.transform().transform_point(port)
        );
    }
}
