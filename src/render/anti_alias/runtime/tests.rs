use super::*;
use bevy_args::ValueEnum;

#[test]
fn presentation_policy_keeps_dataset_views_and_numeric_annotations_unjittered() {
    let mut config = BevyZeroverseConfig {
        camera_grid: false,
        headless: false,
        image_copiers: false,
        ..default()
    };
    assert_eq!(
        requested(&config, &RenderMode::Color, true, false),
        Some(PreviewAa::Taa)
    );
    config.camera_grid = true;
    assert_eq!(requested(&config, &RenderMode::Color, false, false), None);
    config.kiosk = true;
    assert_eq!(
        requested(&config, &RenderMode::Color, false, false),
        Some(PreviewAa::Taa)
    );
    assert_eq!(requested(&config, &RenderMode::Color, true, false), None);
    for &aa in PreviewAa::value_variants() {
        config.preview_aa = aa;
        for mode in RenderMode::value_variants() {
            assert_eq!(
                requested(&config, mode, false, false).is_some(),
                *mode == RenderMode::Color
            );
            assert!(requested(&config, mode, false, true).is_none());
            config.image_copiers = true;
            assert!(requested(&config, mode, false, false).is_none());
            config.image_copiers = false;
            config.headless = true;
            assert!(requested(&config, mode, false, false).is_none());
            config.headless = false;
            config.room_schematic = true;
            assert!(requested(&config, mode, false, false).is_none());
            config.room_schematic = false;
        }
    }
}

#[test]
fn angular_temporal_requests_and_unsupported_sample_counts_have_spatial_fallbacks() {
    let four = SampleCounts(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4);
    for angular in [true, false] {
        assert!(
            select(PreviewAa::Msaa8, angular, false, |n| four.resolve(n))
                == Filter::Msaa(Msaa::Sample4)
        );
        assert!(
            select(PreviewAa::Msaa2, angular, false, |n| four.resolve(n))
                == Filter::Smaa(SmaaPreset::High)
        );
        assert!(
            select(PreviewAa::SmaaUltra, angular, false, |_| unreachable!())
                == Filter::Smaa(SmaaPreset::Ultra)
        );
        assert!(
            select(PreviewAa::Taa, false, true, |_| unreachable!())
                == Filter::Smaa(SmaaPreset::Ultra)
        );
        let temporal = select(PreviewAa::Taa, angular, false, |_| unreachable!());
        assert!(
            temporal
                == if angular {
                    Filter::LensSmaa(SmaaPreset::High)
                } else {
                    Filter::Taa
                }
        );
        assert!(
            select(PreviewAa::TaaFxaa, angular, false, |_| unreachable!())
                == if angular {
                    Filter::LensSmaa(SmaaPreset::Ultra)
                } else {
                    Filter::TaaFxaa
                }
        );
    }
    assert!(
        select(PreviewAa::TaaFxaa, false, true, |_| unreachable!())
            == Filter::Smaa(SmaaPreset::Ultra)
    );
}

#[test]
fn changing_modes_resets_history_on_cuts_and_releases_only_owned_prepasses() {
    let mut app = App::new();
    app.insert_resource(BevyZeroverseConfig {
        camera_grid: false,
        headless: false,
        ..default()
    })
    .init_resource::<RenderMode>()
    .add_message::<SceneLoadedEvent>()
    .add_systems(Update, configure);
    let id = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            ProcessedEditorCameraMarker,
            Msaa::Off,
            MotionVectorPrepass,
        ))
        .id();
    app.update();
    let clear_history = |app: &mut App| {
        app.world_mut()
            .get_mut::<TemporalAntiAliasing>(id)
            .unwrap()
            .reset = false
    };
    let was_reset = |app: &App| app.world().get::<TemporalAntiAliasing>(id).unwrap().reset;
    assert!(was_reset(&app));
    assert_eq!(
        app.world().get::<MipBias>(id).unwrap().0,
        MipBias::default().0
    );
    clear_history(&mut app);
    // Changing the post-tonemap filter must keep the accumulated history and
    // the ownership of the prepasses from the original temporal selection.
    for setting in [PreviewAa::TaaFxaa, PreviewAa::Taa] {
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .preview_aa = setting;
        app.update();
        let camera = app.world().entity(id);
        assert_eq!(camera.contains::<Fxaa>(), setting == PreviewAa::TaaFxaa);
        assert!(camera.contains::<TemporalJitter>() && !was_reset(&app));
        assert_eq!(camera.get::<Msaa>(), Some(&Msaa::Off));
    }
    app.world_mut()
        .get_mut::<Transform>(id)
        .unwrap()
        .translation
        .x += 0.05;
    app.update();
    assert!(!was_reset(&app));
    app.world_mut()
        .get_mut::<Transform>(id)
        .unwrap()
        .translation
        .x += 5.;
    app.update();
    assert!(was_reset(&app));
    clear_history(&mut app);
    app.world_mut().write_message(SceneLoadedEvent);
    app.update();
    assert!(was_reset(&app));
    clear_history(&mut app);
    app.update();
    assert!(!was_reset(&app));
    for (setting, fxaa, smaa) in [
        (PreviewAa::SmaaUltra, false, true),
        (PreviewAa::Fxaa, true, false),
        (PreviewAa::Off, false, false),
    ] {
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .preview_aa = setting;
        app.update();
        let camera = app.world().entity(id);
        assert_eq!(camera.contains::<Fxaa>(), fxaa);
        assert_eq!(camera.contains::<Smaa>(), smaa);
        assert!(camera.contains::<MotionVectorPrepass>());
        assert!(
            !camera.contains::<DepthPrepass>()
                && !camera.contains::<TemporalJitter>()
                && !camera.contains::<MipBias>()
                && !camera.contains::<TemporalAntiAliasing>()
        );
        assert_eq!(camera.get::<Msaa>(), Some(&Msaa::Off));
    }
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .preview_aa = PreviewAa::Taa;
    app.update();
    *app.world_mut().resource_mut::<RenderMode>() = RenderMode::Position;
    app.update();
    let camera = app.world().entity(id);
    assert!(!camera.contains::<ManagedPreviewAa>() && !camera.contains::<TemporalJitter>());
    assert!(camera.contains::<MotionVectorPrepass>());
}

#[test]
fn mixed_kiosk_grid_restores_lens_sampling_and_external_mip_bias_before_capture() {
    let mut app = App::new();
    app.insert_resource(BevyZeroverseConfig {
        kiosk: true,
        camera_grid: true,
        preview_aa: PreviewAa::TaaFxaa,
        ..default()
    })
    .init_resource::<RenderMode>()
    .add_message::<SceneLoadedEvent>()
    .add_systems(Update, configure);
    let pinhole = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            ZeroverseCamera::default(),
            Msaa::Off,
            MipBias(0.25),
        ))
        .id();
    let angular = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            ZeroverseCamera::default(),
            Msaa::Off,
            LensCamera::test_camera(),
        ))
        .id();
    app.update();
    let pin = app.world().entity(pinhole);
    assert!(pin.contains::<TemporalAntiAliasing>() && pin.contains::<Fxaa>());
    assert_eq!(pin.get::<MipBias>().unwrap().0, 0.25);
    let lens = app.world().entity(angular);
    assert!(
        lens.contains::<LensPreviewAa>() && lens.get::<Smaa>().unwrap().preset == SmaaPreset::Ultra
    );
    assert!(!lens.contains::<TemporalJitter>() && !lens.contains::<Fxaa>());

    app.insert_resource(SamplerState::default());
    app.update();
    for id in [pinhole, angular] {
        let camera = app.world().entity(id);
        assert!(!camera.contains::<ManagedPreviewAa>() && !camera.contains::<LensPreviewAa>());
        assert!(
            !camera.contains::<Fxaa>()
                && !camera.contains::<Smaa>()
                && !camera.contains::<TemporalJitter>()
        );
    }
    assert_eq!(app.world().get::<MipBias>(pinhole).unwrap().0, 0.25);
}
