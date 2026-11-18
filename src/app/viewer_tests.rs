use super::*;
use bevy::ecs::system::RunSystemOnce;

#[cfg(target_os = "linux")]
#[test]
fn explicit_cuda_mask_preserves_vulkan_compute_and_raster_features() {
    let rays =
        WgpuFeatures::EXPERIMENTAL_RAY_QUERY | WgpuFeatures::EXPERIMENTAL_RAY_TRACING_PIPELINES;
    for mask in ["", "-1"] {
        assert_eq!(masked_cuda_features(Some(mask)), Some(rays));
    }
    for selection in [None, Some("0"), Some("1,0"), Some("GPU-visible-uuid")] {
        assert_eq!(masked_cuda_features(selection), None);
    }
}

#[test]
fn grid_disables_editor_scene_but_preserves_ui_and_restores_editor() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(BevyZeroverseConfig::default());
    app.init_resource::<ZeroverseRoomSettings>();
    app.world_mut().spawn(Window::default());
    let editor = app
        .world_mut()
        .spawn((
            EditorCameraMarker::default(),
            PanOrbitCamera::default(),
            Camera::default(),
        ))
        .id();
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .camera_grid = true;
    app.world_mut().run_system_once(setup_camera).unwrap();
    assert!(!app.world().get::<Camera>(editor).unwrap().is_active);
    let ui = app
        .world_mut()
        .query_filtered::<Entity, With<MaterialGridCameraMarker>>()
        .single(app.world())
        .unwrap();
    assert!(app.world().get::<Camera>(ui).unwrap().is_active);
    assert!(matches!(
        app.world().get::<Camera>(ui).unwrap().clear_color,
        ClearColorConfig::Custom(_)
    ));
    assert!(app.world().get::<bevy::ui::IsDefaultUiCamera>(ui).is_some());
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .camera_grid = false;
    app.world_mut().run_system_once(setup_camera).unwrap();
    assert!(app.world().get::<Camera>(editor).unwrap().is_active);
    let ClearColorConfig::Custom(clear) = app.world().get::<Camera>(ui).unwrap().clear_color else {
        panic!("UI intermediate must clear transparent every frame")
    };
    assert_eq!(clear, Color::NONE);
    let bevy::camera::CameraOutputMode::Write {
        blend_state,
        clear_color,
    } = app.world().get::<Camera>(ui).unwrap().output_mode
    else {
        panic!("UI must reach the window")
    };
    assert_eq!(
        blend_state,
        Some(bevy::render::render_resource::BlendState::ALPHA_BLENDING)
    );
    assert!(matches!(clear_color, ClearColorConfig::None));
}

#[test]
fn co_visibility_grid_uses_dedicated_preview_and_restores_rgb() {
    use crate::render::ground_truth::GroundTruthCamera;
    let mut app = App::new();
    app.init_resource::<Assets<Image>>();
    app.init_resource::<Assets<camera_grid::GridImage>>();
    app.insert_resource(RenderMode::CoVisibility);
    let (rgb, preview, gt) = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let rgb = images.add(Image::default());
        let preview = images.add(Image::default());
        let mut gt = GroundTruthCamera::new(&mut images, UVec2::new(32, 24));
        gt.enable_co_visibility(&mut images, UVec2::new(32, 24));
        gt.co_visibility.as_mut().unwrap().preview = Some(preview.clone());
        (rgb, preview, gt)
    };
    let camera = app
        .world_mut()
        .spawn((RenderTarget::Image(rgb.clone().into()), gt))
        .id();
    let material = app
        .world_mut()
        .resource_mut::<Assets<camera_grid::GridImage>>()
        .add(camera_grid::GridImage {
            image: rgb.clone(),
            filter_rgb: 1,
        });
    let node = app
        .world_mut()
        .spawn((
            CameraGridSource(camera),
            MaterialNode(material.clone()),
            ImageNode {
                image: rgb.clone(),
                ..default()
            },
        ))
        .id();
    app.world_mut()
        .run_system_once(camera_grid::update_images)
        .unwrap();
    assert_eq!(app.world().get::<ImageNode>(node).unwrap().image, preview);
    let grid = app
        .world()
        .resource::<Assets<camera_grid::GridImage>>()
        .get(&material)
        .unwrap();
    assert_eq!((grid.image.id(), grid.filter_rgb), (preview.id(), 0));
    *app.world_mut().resource_mut::<RenderMode>() = RenderMode::Color;
    app.world_mut()
        .run_system_once(camera_grid::update_images)
        .unwrap();
    assert_eq!(app.world().get::<ImageNode>(node).unwrap().image, rgb);
    let grid = app
        .world()
        .resource::<Assets<camera_grid::GridImage>>()
        .get(&material)
        .unwrap();
    assert_eq!((grid.image.id(), grid.filter_rgb), (rgb.id(), 1));
}

#[test]
fn camera_grid_footprint_shader_validates_without_optional_gpu_features() {
    let source = include_str!("camera_grid.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
}
