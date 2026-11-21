use super::*;
use crate::scene::procedural_indoor::{
    architecture, layout::IndoorManifest, objects::audit, IndoorQuality,
};
use bevy::ecs::world::CommandQueue;

#[test]
fn bulb_poses_stay_on_the_cable_and_clear_neighbors_and_backing() {
    let mut gravity_tilts = Vec::new();
    let mut clip_tilts = Vec::new();
    let mut bay_changes = 0;
    for seed in 0..512 {
        let mut o = audit::fixture(ObjectKind::StringLights, seed);
        for size in [
            Vec3::new(1.4, 0.20, 0.08),
            Vec3::new(3., 0.65, 0.14),
            Vec3::new(7.5, 0.90, 3.4),
        ] {
            o.size = size;
            let p = program(&o);
            assert!(
                p.bulbs <= (if p.practical { MAX_PRACTICAL } else { 36 })
                    && p.colors_srgb.len() <= 8
            );
            let line = p.line(size);
            let mut last = f32::NEG_INFINITY;
            let mut used = 0u16;
            for bulb in p.sockets(&o) {
                used |= 1 << bulb.channel;
                let distance = line
                    .windows(2)
                    .map(|s| {
                        let ab = s[1] - s[0];
                        bulb.tap.distance(
                            s[0] + ab
                                * ((bulb.tap - s[0]).dot(ab) / ab.length_squared()).clamp(0., 1.),
                        )
                    })
                    .fold(f32::INFINITY, f32::min);
                assert!(distance < 0.00001);
                assert!((bulb.axis.length() - 1.).abs() < 1e-5);
                let (lo, hi) = bulb.bounds();
                assert!(
                    lo.x > last && lo.z >= -size.z * 0.5 - 1e-4 && hi.z <= size.z * 0.5 + 1e-4,
                    "seed {seed} bounds {lo:?}..{hi:?} after {last}"
                );
                last = hi.x;
                let mut g = Geometry::default();
                bulb.aperture(&mut g);
                for v in g.positions.iter().copied().map(Vec3::from_array) {
                    let eps = Vec3::splat(1e-6);
                    assert!(v.cmpge(lo-eps).all() && v.cmple(hi+eps).all() && v.y >= 0. && v.y <= size.y,
                        "seed {seed} size {size:?}, shape {}, tap {:?} socket {:?} axis {:?} vertex {v:?} bounds {lo:?}..{hi:?}",bulb.shape,bulb.tap,bulb.socket,bulb.axis);
                }
                let tilt = bulb.axis.angle_between(Vec3::NEG_Y).to_degrees();
                if p.clipped {
                    clip_tilts.push(tilt);
                } else {
                    gravity_tilts.push(tilt);
                }
            }
            assert_eq!(used, (1 << p.colors_srgb.len()) - 1);
            if p.bays > 1 && (p.span_sags[0] - p.span_sags[1]).abs() > 0.07 {
                bay_changes += 1;
            }
        }
    }
    assert!(bay_changes > 70);
    assert!(gravity_tilts.iter().all(|&v| v < 16.));
    assert!(gravity_tilts.iter().any(|&v| v > 9.));
    assert!(clip_tilts.iter().filter(|&&v| v > 30.).count() > 500);
}

#[test]
fn practical_sources_match_emitting_area_color_and_world_axis() {
    let mut scene =
        IndoorManifest::generate_with_humans(4, Default::default(), 0.65, 0, 0.).unwrap();
    scene.objects.clear();
    let mut shapes = [false; 5];
    for seed in 0..160 {
        let mut o = audit::fixture(ObjectKind::StringLights, seed);
        o.variant = 3;
        o.size = Vec3::new(4., 0.7, 0.9);
        o.position = Vec3::new(2., 3., -1.);
        o.yaw = 1.27;
        o.id = 7;
        let p = program(&o);
        shapes[p.shape as usize] = true;
        scene.objects.push(o.clone());
        let scale = illumination_scale(&scene);
        let sources: Vec<_> = practical_lights(&scene).collect();
        assert_eq!(sources.len(), p.bulbs);
        for (b, (_, _, l)) in p.sockets(&o).zip(sources) {
            assert!(l
                .position
                .abs_diff_eq(o.transform().transform_point(b.center()), 1e-6));
            assert!(l.axis.abs_diff_eq(o.transform().rotation * b.axis, 1e-6));
            let mut g = Geometry::default();
            b.aperture(&mut g);
            let area: f32 = g
                .indices
                .as_chunks::<3>()
                .0
                .iter()
                .map(|idx| {
                    let v = |i: usize| Vec3::from_array(g.positions[idx[i] as usize]);
                    ((v(1) - v(0)).cross(v(2) - v(0))).length() * 0.5
                })
                .sum();
            let projected = if p.shape == 4 {
                assert!(g
                    .normals
                    .iter()
                    .all(|&n| Vec3::from_array(n).dot(b.axis) > 0.9999));
                assert!(l.cone.is_some());
                area
            } else {
                assert!(l.cone.is_none());
                area * 0.25
            };
            let c = p.emission(b.channel, scale).unwrap();
            let expected = Vec3::new(c.red, c.green, c.blue) * projected;
            let actual = l.color.to_linear();
            let flux = Vec3::new(actual.red, actual.green, actual.blue) * l.candela;
            // The smooth photometric profile has slightly more area than its tessellation.
            assert!(
                (flux - expected).length() < 0.08 * expected.length(),
                "shape {}: {flux:?} vs {expected:?}",
                p.shape
            );
        }
        scene.objects.clear();
    }
    assert!(shapes.into_iter().all(|s| s));
}

#[test]
fn room_spawns_shadowed_diffuse_and_directional_string_practicals() {
    let mut scene =
        IndoorManifest::generate_with_humans(4, Default::default(), 0.65, 0, 0.).unwrap();
    scene.objects.clear();
    for (id, directional) in [(0, false), (1, true)] {
        let o = (0..256)
            .map(|seed| {
                let mut o = audit::fixture(ObjectKind::StringLights, seed);
                o.variant = 3;
                o.id = id;
                o.yaw = 0.92;
                o
            })
            .find(|o| (program(o).shape == 4) == directional)
            .unwrap();
        scene.objects.push(o);
    }
    let mut micro = audit::fixture(ObjectKind::StringLights, 4);
    micro.id = 2;
    micro.variant = 1;
    scene.objects.push(micro);
    let mut world = World::new();
    let root = world.spawn_empty().id();
    let mut queue = CommandQueue::default();
    architecture::spawn_lights(
        &scene,
        IndoorQuality::Auto,
        root,
        &mut Commands::new(&mut queue, &world),
    );
    queue.apply(&mut world);
    for (id, slot, l) in practical_lights(&scene) {
        let name = format!("indoor_string_{id}_{slot}");
        let (t, point, spot) = world
            .query::<(&Name, &Transform, Option<&PointLight>, Option<&SpotLight>)>()
            .iter(&world)
            .find_map(|(n, t, p, s)| (n.as_str() == name).then_some((t, p, s)))
            .unwrap();
        assert!(t.translation.abs_diff_eq(l.position, 1e-6));
        if let Some(s) = spot {
            assert!(t.forward().dot(l.axis) > 0.99999 && s.shadow_maps_enabled);
            assert_eq!(Some([s.inner_angle, s.outer_angle]), l.cone);
            assert!(s.shadow_map_near_z < s.radius);
            assert_eq!(s.intensity, 4. * PI * l.candela);
            assert_eq!(s.color, l.color);
        } else {
            let p = point.unwrap();
            assert!(p.shadow_maps_enabled && l.cone.is_none());
            assert!(p.shadow_map_near_z < p.radius);
            assert_eq!(p.intensity, 4. * PI * l.candela);
            assert_eq!(p.color, l.color);
        }
    }
    assert!(!world
        .query::<&Name>()
        .iter(&world)
        .any(|n| n.starts_with("indoor_string_2_")));
}

#[test]
fn miniature_fallback_recovers_nearby_illumination_without_shadow_views() {
    let mut scene =
        IndoorManifest::generate_with_humans(4, Default::default(), 0.65, 0, 0.).unwrap();
    scene.objects.clear();
    for seed in 0..12 {
        let mut o = audit::fixture(ObjectKind::StringLights, seed);
        o.id = seed as usize;
        o.variant = (seed % 4) as u32;
        o.yaw = 1.81;
        o.size = Vec3::new(3.5, 0.62, if suspended(&o) { 1.1 } else { 0.11 });
        scene.objects.push(o);
    }
    let mut world = World::new();
    let root = world.spawn_empty().id();
    let mut queue = CommandQueue::default();
    spawn_miniature_lights(&scene, root, &mut Commands::new(&mut queue, &world));
    queue.apply(&mut world);
    let mut query = world.query::<(&Name, &PointLight, &Transform)>();
    let mut count = 0;
    for o in scene.objects.iter().filter(|o| !practical(o)) {
        let p = program(o);
        for (slot, b) in p.sockets(o).enumerate() {
            let name = format!("indoor_string_micro_{}_{slot}", o.id);
            let (_, source, tf) = query
                .iter(&world)
                .find(|(n, _, _)| n.as_str() == name)
                .unwrap();
            let local = b.center();
            assert!(tf
                .translation
                .abs_diff_eq(o.transform().transform_point(local), 1e-5));
            let emission = p.emission(b.channel, illumination_scale(&scene)).unwrap();
            let expected =
                Vec3::new(emission.red, emission.green, emission.blue) * b.projected_area();
            let c = source.color.to_linear();
            let actual = Vec3::new(c.red, c.green, c.blue) * (source.intensity / (4. * PI));
            assert!(actual.abs_diff_eq(expected, 1e-5));
            assert!(!source.shadow_maps_enabled && source.range <= 0.28);
            let backing = if p.suspended {
                o.size.y - local.y + 0.05
            } else {
                local.z + 0.5 * o.size.z + scene.envelope.as_ref().unwrap().wall_thickness()
            };
            assert!(source.range < backing && source.range > source.radius);
            count += 1;
        }
    }
    assert_eq!(count, query.iter(&world).count());
}
