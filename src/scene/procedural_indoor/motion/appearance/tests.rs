use super::*;
use crate::camera::Playback;

#[test]
fn clock_drives_private_screen_pages_visible_emission_and_direct_power_in_both_directions() {
    let mut app = App::new();
    app.init_resource::<Playback>()
        .init_resource::<Assets<Image>>()
        .init_resource::<Assets<StandardMaterial>>();
    install(&mut app);
    let pages: [Handle<Image>; 2] = std::array::from_fn(|_| {
        app.world_mut()
            .resource_mut::<Assets<Image>>()
            .add(Image::default())
    });
    let raw = StandardMaterial {
        emissive_texture: Some(pages[0].clone()),
        ..default()
    };
    let mut assets = app.world_mut().resource_mut::<Assets<StandardMaterial>>();
    let screen = assets.add(raw.clone());
    let unchanged = assets.add(raw);
    let emission = LinearRgba::new(8., 4., 2., 1.);
    let aperture = assets.add(StandardMaterial {
        emissive: emission,
        ..default()
    });
    let curve = DimmerCurve {
        interval: [0.2, 0.8],
        end_multiplier: 0.25,
    };
    let mut root = AppearancePlayback::default();
    root.screen(0.6, screen.clone(), pages.clone());
    root.aperture(curve, aperture.clone(), emission);
    app.world_mut().spawn(root);
    let point = app
        .world_mut()
        .spawn((
            PointLight {
                intensity: 800.,
                ..default()
            },
            DimmedLight::new(curve, 800.),
        ))
        .id();
    let spot = app
        .world_mut()
        .spawn((
            SpotLight {
                intensity: 240.,
                ..default()
            },
            DimmedLight::new(curve, 240.),
        ))
        .id();
    for t in [0., 0.51, 1., 0.6, 0.59, 0.12, 1., 0.] {
        app.world_mut().resource_mut::<Playback>().progress = t;
        app.update();
        let f = curve.multiplier_at(t);
        let assets = app.world().resource::<Assets<StandardMaterial>>();
        assert_eq!(
            assets.get(&screen).unwrap().emissive_texture.as_ref(),
            Some(&pages[usize::from(t >= 0.6)])
        );
        assert_eq!(
            assets.get(&unchanged).unwrap().emissive_texture.as_ref(),
            Some(&pages[0])
        );
        assert_eq!(
            assets.get(&aperture).unwrap().emissive,
            LinearRgba::new(8. * f, 4. * f, 2. * f, 1.)
        );
        assert_eq!(
            app.world().get::<PointLight>(point).unwrap().intensity,
            800. * f
        );
        assert_eq!(
            app.world().get::<SpotLight>(spot).unwrap().intensity,
            240. * f
        );
        assert_eq!(assets.len(), 3);
        assert_eq!(app.world().resource::<Assets<Image>>().len(), 2);
    }
}

#[test]
fn all_display_types_have_different_replayable_second_pages_at_the_same_mip_resolution() {
    for kind in [
        screens::DisplayKind::Desktop,
        screens::DisplayKind::Phone,
        screens::DisplayKind::Television,
    ] {
        let seed = (0..100)
            .find(|&seed| !screens::parameters_for(seed, kind).dormant)
            .unwrap();
        let first = match kind {
            screens::DisplayKind::Desktop => screens::pixels(seed),
            screens::DisplayKind::Phone => screens::mobile_pixels(seed),
            screens::DisplayKind::Television => screens::television_pixels(seed),
        };
        let next = screens::next_page(seed, kind, 29);
        assert_eq!(next.texture_descriptor.size.width, 256);
        assert!(next.texture_descriptor.mip_level_count > 1);
        assert!(
            next.data == screens::next_page(seed, kind, 29).data,
            "deterministic alternate page"
        );
        assert!(
            first.as_slice() != &next.data.as_ref().unwrap()[..first.len()],
            "unchanged page for {kind:?}"
        );
    }
}

#[test]
fn sampled_room_programs_respect_installed_circuits_power_state_and_nested_probabilities() {
    use crate::scene::procedural_indoor::{layout::IndoorLayout, objects, validation};
    let mut rooms = vec![];
    let mut screen_count = 0;
    let mut ceiling_count = 0;
    let mut lamp_count = 0;
    let mut elapsed = std::time::Duration::ZERO;
    for (i, layout) in IndoorLayout::PROFILES.into_iter().enumerate() {
        let scene =
            IndoorManifest::generate_with_humans(88200 + i as u64, layout, 0.66, 2, 0.).unwrap();
        validation::validate_layout(&scene).unwrap();
        // Appearance admission only inspects geometry for optical faces.
        let geometry = SceneGeometry {
            architecture: default(),
            humans: vec![],
            objects: scene
                .objects
                .iter()
                .map(|o| {
                    if display(o.kind).is_some() {
                        objects::build_manifest_object(&scene, o)
                    } else {
                        Assembly::default()
                    }
                })
                .collect(),
        };
        let sample = |fraction| {
            AppearanceTimeline::sample(
                &scene,
                &geometry,
                &IndoorMotionConfig {
                    screen_change_fraction: fraction,
                    light_change_fraction: fraction,
                    ..default()
                },
            )
        };
        assert!(sample(0.).is_empty());
        let start = std::time::Instant::now();
        let all = sample(1.);
        elapsed += start.elapsed();
        assert_eq!(all, sample(1.));
        assert!(all
            .screens
            .windows(2)
            .all(|p| p[0].object_id < p[1].object_id));
        assert!(all.lights.windows(2).all(|p| p[0].source < p[1].source));
        let subset = sample(0.4);
        assert!(subset.screens.iter().all(|v| all.screens.contains(v)));
        assert!(subset.lights.iter().all(|v| all.lights.contains(v)));
        let positions = architecture::fixture_positions(&scene);
        let powered: Vec<_> = architecture::powered_ceiling_fixtures(&scene, &positions)
            .map(|(i, _, _, _)| i)
            .collect();
        let ceiling: Vec<_> = all
            .lights
            .iter()
            .filter_map(|l| match l.source {
                LightSource::Ceiling(i) => Some(i),
                _ => None,
            })
            .collect();
        assert_eq!(ceiling, powered);
        let wiring = scene
            .domain()
            .and_then(|d| d.photometry.ceiling_circuits.as_ref())
            .unwrap();
        let mut circuits = std::collections::BTreeMap::new();
        for light in &all.lights {
            if let LightSource::Ceiling(i) = light.source {
                if let Some(previous) = circuits.insert(wiring.mount_circuits[i], light.curve) {
                    assert_eq!(previous, light.curve);
                }
            } else if let LightSource::Object(id) = light.source {
                assert!(photometry::lamp(&scene, &scene.objects[id]).lumens > 0.);
            }
        }
        for sc in &all.screens {
            let (surface, device) = display(scene.objects[sc.object_id].kind).unwrap();
            assert!(!screens::parameters_for(sc.initial_content_seed, device).dormant);
            assert!(geometry.objects[sc.object_id]
                .parts
                .keys()
                .any(|(s, _)| *s == surface));
        }
        screen_count += all.screens.len();
        ceiling_count += ceiling.len();
        lamp_count += all.lights.len() - ceiling.len();
        rooms.push(serde_json::json!({"seed":scene.seed,"layout":layout,"screens":all.screens.len(),"ceiling_sources":ceiling.len(),"object_lamps":all.lights.len()-ceiling.len(),"active_ceiling_circuits":circuits.len()}));
    }
    assert!(screen_count > 10 && ceiling_count > 20 && lamp_count > 0);
    let path = std::path::Path::new("out/indoor_temporal_programs");
    std::fs::create_dir_all(path).unwrap();
    std::fs::write(path.join("program-sample.json"),serde_json::to_vec_pretty(&serde_json::json!({"rooms":rooms,"active_screen_count":screen_count,"powered_ceiling_sources":ceiling_count,"powered_object_lamps":lamp_count,"appearance_selection_ms_total":elapsed.as_secs_f64()*1000.})).unwrap()).unwrap();
}
