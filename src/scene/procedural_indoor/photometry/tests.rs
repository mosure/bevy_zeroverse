use super::*;
use crate::scene::procedural_indoor::photometry::source_color;
use crate::scene::procedural_indoor::{
    appearance::AppearanceSettings,
    architecture,
    layout::IndoorLayout,
    materials::screens::{self, DisplayKind},
    objects::{self, audit},
    IndoorQuality,
};
use std::collections::BTreeSet;

fn scene() -> IndoorManifest {
    IndoorManifest::generate_with_humans(6, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap()
}

#[test]
fn lamp_sources_cover_brightness_temperature_and_tint_with_exact_replay() {
    let scene = scene();
    let mut report = Vec::new();
    for kind in [
        ObjectKind::FloorLamp,
        ObjectKind::DeskLamp,
        ObjectKind::CeilingPendant,
        ObjectKind::WallSconce,
    ] {
        let mut values = BTreeSet::new();
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        let mut off = 0;
        for seed in 0..1024 {
            let object = audit::fixture(kind, seed);
            let p = lamp(&scene, &object);
            assert_eq!(p, lamp(&scene, &object));
            assert!(
                p.color.is_finite() && p.color.min_element() > 0. && p.color.max_element() <= 1.
            );
            assert!(p.lumens.is_finite() && (0.0..=3000.).contains(&p.lumens));
            off += usize::from(p.lumens == 0.);
            values.insert((p.kelvin.to_bits(), p.tint.to_bits(), p.lumens.to_bits()));
            let v = Vec3::new(p.lumens, p.kelvin, p.tint);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert_eq!(values.len(), 1024);
        assert!((50..150).contains(&off));
        assert!(hi.y - lo.y > 3500. && hi.z - lo.z > 0.10);
        report.push(serde_json::json!({"kind":kind, "samples":1024, "off":off, "min_lumens_kelvin_tint":lo, "max_lumens_kelvin_tint":hi}));
    }
    println!("source distributions: {}", serde_json::json!(report));
}

#[test]
fn lamp_dimming_is_applied_once_and_relighting_never_moves_geometry() {
    let original = scene();
    let mut dimmed = original.clone();
    dimmed
        .apply_appearance(AppearanceSettings {
            illumination_scale: 0.2,
            ..default()
        })
        .unwrap();
    let mut relit = original.clone();
    relit
        .apply_appearance(AppearanceSettings {
            lighting_seed: Some(901),
            ..default()
        })
        .unwrap();
    for seed in 0..32 {
        let o = audit::fixture(ObjectKind::DeskLamp, seed);
        let a = lamp(&original, &o);
        let b = lamp(&dimmed, &o);
        assert_eq!(a.color, b.color);
        assert!((b.lumens - a.lumens * 0.2).abs() < 0.0001);
        assert_ne!(a, lamp(&relit, &o));
    }
    assert_eq!(
        serde_json::to_value(&original.objects).unwrap(),
        serde_json::to_value(&relit.objects).unwrap()
    );
    assert_eq!(
        architecture::fixture_positions(&original),
        architecture::fixture_positions(&relit)
    );
}

#[test]
fn lamps_fit_extreme_placement_envelopes_and_preserve_semantic_ownership() {
    for seed in 0..128 {
        for kind in [
            ObjectKind::FloorLamp,
            ObjectKind::DeskLamp,
            ObjectKind::CeilingPendant,
            ObjectKind::WallSconce,
        ] {
            for scale in [
                Vec3::ONE,
                Vec3::new(0.70, 1.15, 1.20),
                Vec3::new(1.20, 0.80, 0.70),
            ] {
                let mut o = audit::fixture(kind, seed);
                o.size *= scale;
                o.id = 74;
                let a = objects::build_object(&o);
                audit::check(&o, &a)
                    .unwrap_or_else(|e| panic!("{kind:?} seed {seed} scale {scale:?}: {e}"));
                for (surface, label) in a.parts.keys() {
                    if *surface == super::super::materials::Surface::Light {
                        assert!(label.contains("#source74"));
                        assert_eq!(objects::part_label(label), kind.class_name());
                    }
                }
            }
        }
    }
}

#[test]
fn runtime_lamp_lights_match_exportable_power_and_preserve_shadow_budget() {
    let mut scene = scene();
    scene.objects.clear();
    for kind in [
        ObjectKind::FloorLamp,
        ObjectKind::DeskLamp,
        ObjectKind::CeilingPendant,
        ObjectKind::WallSconce,
    ] {
        for powered in [true, false] {
            let mut o = (0..256)
                .map(|seed| audit::fixture(kind, seed))
                .find(|o| (lamp(&scene, o).lumens > 0.) == powered)
                .unwrap();
            o.id = scene.objects.len();
            scene.objects.push(o);
        }
    }
    let mut world = World::new();
    let root = world.spawn_empty().id();
    let mut queue = bevy::ecs::world::CommandQueue::default();
    architecture::spawn_lights(
        &scene,
        IndoorQuality::Auto,
        root,
        &mut Commands::new(&mut queue, &world),
    );
    queue.apply(&mut world);
    let points: Vec<_> = world.query::<&PointLight>().iter(&world).collect();
    assert_eq!(points.len(), 1);
    let power = lamp(&scene, &scene.objects[0]);
    assert_eq!(points[0].intensity, power.lumens);
    assert_eq!(points[0].color, source_color(&scene, power.color));
    assert!(points[0].shadow_maps_enabled);
    let spots: Vec<_> = world
        .query::<(&Name, &SpotLight, &Transform)>()
        .iter(&world)
        .collect();
    assert_eq!(
        spots.len(),
        architecture::powered_ceiling_fixtures(&scene, &architecture::fixture_positions(&scene))
            .count()
            + 3
    );
    for o in scene
        .objects
        .iter()
        .filter(|o| o.kind != ObjectKind::FloorLamp)
    {
        let source_entity = spots
            .iter()
            .find(|(name, _, _)| name.as_str() == format!("indoor_desk_lamp_{}", o.id));
        let power = lamp(&scene, o);
        if power.lumens == 0. {
            assert!(source_entity.is_none());
            continue;
        }
        let (_, light, pose) = source_entity.unwrap();
        let (beam, source) = architecture::object_fixture_photometry(&scene, o);
        assert_eq!((source.color, source.lumens), (power.color, power.lumens));
        assert!(Vec3::from(pose.forward()).abs_diff_eq(beam.direction, 1e-5));
        assert!(pose.translation.abs_diff_eq(beam.position, 1e-6));
        assert_eq!(
            light.intensity,
            architecture::spot_intensity_for_lumens(power.lumens, beam.inner, beam.outer)
        );
        assert_eq!(light.color, source_color(&scene, source.color));
        assert!(light.shadow_maps_enabled);
    }
}

#[test]
fn display_output_has_device_tails_and_calibrated_white_luminance() {
    for kind in [
        DisplayKind::Desktop,
        DisplayKind::Phone,
        DisplayKind::Television,
    ] {
        let mut lo = f32::INFINITY;
        let mut hi = 0_f32;
        let mut warm = 0;
        for seed in 0..1024 {
            let p = screens::parameters_for(seed, kind);
            lo = lo.min(p.luminance);
            hi = hi.max(p.luminance);
            warm += usize::from(p.white_kelvin < 5000.);
            let c = display_emission(p.luminance, p.white_kelvin, p.white_tint);
            let y = c.red * 0.2126 + c.green * 0.7152 + c.blue * 0.0722;
            assert!((y / p.luminance - 1.).abs() < 1e-6);
        }
        assert!(lo < 20. && hi > 500. && (100..300).contains(&warm));
        println!("display output {kind:?}: {lo:.2}..{hi:.2} cd/m2, warm={warm}/1024");
    }
}

#[test]
fn batched_ceiling_output_is_identical_to_individual_queries() {
    let scene = scene();
    let positions = architecture::fixture_positions(&scene);
    let baseline: Vec<_> = (0..positions.len())
        .map(|i| architecture::fixture_photometry(&scene, i))
        .collect();
    let batched: Vec<_> = (0..positions.len())
        .map(|i| architecture::fixture_photometry_at(&scene, i, &positions))
        .collect();
    assert_eq!(baseline, batched);
    let time = |batched: bool| {
        let start = std::time::Instant::now();
        for _ in 0..50 {
            let positions = architecture::fixture_positions(&scene);
            for i in 0..positions.len() {
                std::hint::black_box(if batched {
                    architecture::fixture_photometry_at(&scene, i, &positions)
                } else {
                    architecture::fixture_photometry(&scene, i)
                });
            }
        }
        start.elapsed().as_secs_f64() * 1e6 / 50.
    };
    println!("ceiling output sampling ({} fixtures): reused_positions_us={:.2}, repeated_queries_us={:.2}; not completed room throughput", positions.len(),time(true),time(false));
}

#[test]
fn low_light_room_prior_dims_local_sources_without_resampling_them() {
    let mut dark = scene();
    dark.target_lux = 30.;
    let mut bright = dark.clone();
    bright.target_lux = 300.;
    for seed in 0..64 {
        let object = audit::fixture(ObjectKind::FloorLamp, seed);
        let a = lamp(&dark, &object);
        let b = lamp(&bright, &object);
        assert_eq!(a.color, b.color);
        assert!((a.lumens - b.lumens * 0.1).abs() < 0.0001);
    }
}

#[test]
fn desk_lamp_base_keeps_metric_texture_pitch() {
    for seed in 0..64 {
        let o = audit::fixture(ObjectKind::DeskLamp, seed);
        let p = objects::accessories::lamp_parameters(&o);
        let a = objects::build_object(&o);
        let mut side_span = 0_f32;
        let mut cap_span = 0_f32;
        for g in a.parts.values() {
            for tri in g.indices.as_chunks::<3>().0 {
                if tri
                    .iter()
                    .any(|&i| g.positions[i as usize][1] >= o.size.y * 0.056)
                {
                    continue;
                }
                let cap = tri.iter().all(|&i| g.normals[i as usize][1].abs() > 0.999);
                let side = tri.iter().all(|&i| g.normals[i as usize][1].abs() < 0.001);
                for j in 0..3 {
                    let i = tri[j] as usize;
                    let k = tri[(j + 1) % 3] as usize;
                    let span = (g.uvs[i][0] - g.uvs[k][0]).abs();
                    if cap {
                        cap_span = cap_span.max(span);
                    }
                    if side {
                        side_span = side_span.max(span);
                    }
                }
            }
        }
        // The cap's planar chart spans a radius from centre to rim. On the
        // cylindrical chart each of the twenty rim segments spans an arc.
        let radius = o.size.x * p.base_width;
        let arc = std::f32::consts::TAU * radius / 20.;
        assert!(
            (cap_span - radius).abs() < 1e-5,
            "seed {seed}: cap {cap_span} vs {radius}"
        );
        assert!(
            (side_span - arc).abs() < 1e-5,
            "seed {seed}: side {side_span} vs {arc}"
        );
    }
}
