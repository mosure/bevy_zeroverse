use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object, equipment::receptacles};

/// First surface struck by a ray from above, in object coordinates.
fn height_at(a: &Assembly, point: Vec2) -> (f32, Surface) {
    a.parts
        .iter()
        .flat_map(|((surface, _), g)| {
            g.indices.as_chunks::<3>().0.iter().filter_map(move |tri| {
                let [a, b, c] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                let (u, v, w) = ((b - a).xz(), (c - a).xz(), point - a.xz());
                let area = u.perp_dot(v);
                if area.abs() < 1e-10 {
                    return None;
                }
                let (s, t) = (w.perp_dot(v) / area, u.perp_dot(w) / area);
                (s >= -1e-5 && t >= -1e-5 && s + t <= 1.00001)
                    .then_some((a.y + s * (b.y - a.y) + t * (c.y - a.y), *surface))
            })
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap()
}

#[test]
fn floor_service_covers_and_socket_contacts_share_the_same_metric_cut() {
    let mut double = [0; 3];
    let mut single = [0; 3];
    let mut depths = std::collections::BTreeSet::new();
    for seed in 0..48 {
        for standard in 0..3 {
            let mut o = audit::fixture(ObjectKind::FloorOutlet, seed);
            o.variant = installed_variant((seed % 3) as u32, standard);
            // Include the minimum possible height for an open installation.
            o.size = Vec3::new(if seed % 2 == 0 { 0.16 } else { 0.238 }, 0.07, 0.208);
            o.position = Vec3::new(0.8, 0.1, -2.);
            o.yaw = 0.83;
            let p = parameters(&o).unwrap();
            depths.insert(p.hatch_depth_m.to_bits());
            if p.socket_count == 2 {
                double[standard as usize] += 1;
            } else {
                single[standard as usize] += 1;
            }
            let a = build_object(&o);
            audit::check(&o, &a).unwrap();
            for i in 0..p.socket_count {
                let port = p.port(i).unwrap();
                let world = o.transform().transform_point(port);
                assert_eq!(
                    receptacles::target_port(&o, false, Some(i as u8)),
                    Some(world)
                );
                // Ray through an actual slot, next to the solid mating surface.
                let pin = match p.face {
                    OutletFace::Polarized => Vec2::new(0.00635, -0.003),
                    OutletFace::RoundWell => Vec2::X * 0.0093,
                    OutletFace::ThreePin => Vec2::new(0.0111, 0.004),
                };
                let (depth, material) = height_at(&a, port.xz() + pin);
                assert_eq!(material, Surface::Ink);
                assert!((0.0015..0.0019).contains(&(port.y - depth)));
                assert!((height_at(&a, port.xz()).0 - port.y).abs() < 1e-5);
                // The widest molded plug fits below the angled hatch, including
                // the 39 mm axial approach used by routing.
                let map = |q: Vec3| {
                    let local = p.hinge();
                    local.rotation.inverse() * (Vec3::new(q.x, -q.z, q.y) - local.translation)
                };
                let (lo, hi) = p.lid_bounds();
                assert!(!crate::scene::procedural_indoor::layout::segment_hits_box(
                    map(port),
                    map(port + Vec3::Y * 0.039),
                    lo - Vec3::splat(0.0178),
                    hi + Vec3::splat(0.0178)
                ));
            }
            assert!(receptacles::target_port(&o, false, None).is_none());
            assert!(receptacles::target_port(&o, true, Some(0)).is_none());
            assert!(receptacles::target_port(&o, false, Some(p.socket_count as u8)).is_none());
            assert_eq!(
                receptacles::recessed_connection(&o, Some(0)),
                p.face == OutletFace::RoundWell
            );
            o.variant |= 8;
            o.size.y = 0.006;
            let closed = parameters(&o).unwrap();
            assert_eq!(closed.hatch_size(), p.hatch_size());
            assert_eq!(closed.hinge().translation, p.hinge().translation);
            assert!(receptacles::target_port(&o, false, Some(0)).is_none());
            let a = build_object(&o);
            audit::check(&o, &a).unwrap();
            assert!(a.bounds().1.y <= 0.006 && a.bounds().0.y >= -1e-6);
            for i in 0..p.socket_count {
                let (h, material) = height_at(&a, p.port(i).unwrap().xz());
                assert_eq!(material, p.hatch_finish);
                assert!((h - 0.0053).abs() < 1e-5);
            }
        }
    }
    assert_eq!(
        depths.len(),
        48,
        "sample inside the fit interval without clipping to a standard depth"
    );
    assert!(
        single.iter().chain(&double).all(|&n| n >= 6),
        "{single:?} {double:?}"
    );
}

#[test]
fn floor_flange_uses_one_brush_direction_across_the_aperture_and_bevel() {
    for case in 0..3 {
        let mut o = audit::fixture(ObjectKind::FloorOutlet, 13);
        o.variant = installed_variant(case, 0);
        let p = parameters(&o).unwrap();
        let mut a = Assembly::default();
        flange(&mut a, &p, super::super::service_size(&o).xz());
        let g = &a.parts[&(p.outlet_finish, LABEL.into())];
        assert!(g
            .positions
            .iter()
            .zip(&g.uvs)
            .all(|(pos, uv)| *uv == [pos[0], pos[1]]));
    }
}

#[test]
fn floor_service_routes_seat_full_plugs_and_can_use_both_sockets() {
    use crate::scene::procedural_indoor::{
        layout::{equipment::cable_route_clear, IndoorLayout, IndoorManifest},
        objects::equipment::wires::MainsPlug,
    };
    for (seed, required) in [(93410, 1), (93435, 2)] {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap();
        let mut most = 0;
        for target in scene.objects.iter().filter(|o| installed(o)) {
            let mut used = std::collections::BTreeSet::new();
            for cable in scene
                .objects
                .iter()
                .filter(|o| o.cable_target == Some(target.id))
            {
                let index = cable.cable_target_socket.unwrap();
                assert!(used.insert(index));
                assert!(MainsPlug::world(cable).unwrap().fits());
                assert!(cable_route_clear(&scene, cable));
            }
            most = most.max(used.len());
        }
        assert!(most >= required, "room {seed}: {most} seated plugs");
    }
}
