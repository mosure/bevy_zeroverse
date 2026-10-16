use super::*;
use bevy::{
    camera::primitives::CascadesFrusta,
    light::{
        cascade::build_directional_light_cascades, update_directional_light_frusta, Cascades,
        DirectionalLightShadowMap, SimulationLightSystems,
    },
};

#[derive(Resource, Default)]
struct LateChange(Option<Action>);

enum Action {
    Ready,
    Next,
    Submitted(Entity, u64),
}

fn late_change(
    mut action: ResMut<LateChange>,
    mut ready: ResMut<CaptureReadiness>,
    mut progress: ResMut<CaptureProgress>,
    copiers: Query<&ImageCopier>,
) {
    match action.0.take() {
        Some(Action::Ready) => ready.blocker = None,
        Some(Action::Next) => progress.pending = None,
        Some(Action::Submitted(camera, id)) => copiers.get(camera).unwrap().test_submit(id),
        None => {}
    }
}

fn fixture() -> (App, [Entity; 2], Entity) {
    let mut app = App::new();
    app.insert_resource(BevyZeroverseConfig {
        headless: true,
        editor: false,
        ..default()
    });
    app.init_resource::<SamplerState>();
    app.init_resource::<CaptureProgress>();
    app.init_resource::<CaptureReadiness>();
    app.init_resource::<CaptureFailure>();
    app.init_resource::<DirectionalLightShadowMap>();
    app.init_resource::<LateChange>();
    // Use Bevy's actual cascade and frustum construction, with the same ordering
    // as LightPlugin. No renderer, GPU, mock cascade data or sleep is needed.
    app.add_systems(
        PostUpdate,
        (
            build_directional_light_cascades
                .in_set(SimulationLightSystems::UpdateDirectionalLightCascades)
                .after(CameraUpdateSystems),
            update_directional_light_frusta.after(build_directional_light_cascades),
            late_change.after(update_directional_light_frusta),
        ),
    );
    let cameras = [0., 2.].map(|x| {
        app.world_mut()
            .spawn((
                Camera {
                    is_active: false,
                    ..default()
                },
                Projection::Perspective(PerspectiveProjection::default()),
                GlobalTransform::from_translation(Vec3::new(x, 1.6, 0.)),
                ZeroverseCamera::default(),
                ImageCopier::test_idle(),
            ))
            .id()
    });
    let light = app
        .world_mut()
        .spawn((
            DirectionalLight {
                shadow_maps_enabled: true,
                ..default()
            },
            GlobalTransform::IDENTITY,
            ViewVisibility::VISIBLE,
        ))
        .id();
    (app, cameras, light)
}

fn check(app: &App, cameras: [Entity; 2], light: Entity, active: [bool; 2]) {
    let world = app.world();
    let cascades = world.get::<Cascades>(light).unwrap();
    let frusta = world.get::<CascadesFrusta>(light).unwrap();
    for (camera, expected) in cameras.into_iter().zip(active) {
        assert_eq!(world.get::<Camera>(camera).unwrap().is_active, expected);
        if expected {
            assert!(!cascades.cascades.get(&camera).unwrap().is_empty());
            assert_eq!(
                cascades.cascades[&camera].len(),
                frusta.frusta[&camera].len()
            );
        }
    }
}

#[test]
fn late_activation_reproduces_the_missing_shadow_cascade() {
    let (mut app, cameras, light) = fixture();
    // The former Last-only gate enabled a view after Bevy had built no cascade
    // for it. prepare_lights subsequently unwraps this absent map entry.
    app.world_mut()
        .get_mut::<Camera>(cameras[0])
        .unwrap()
        .is_active = true;
    app.add_systems(Last, gate::<true>);
    app.world_mut().resource_mut::<LateChange>().0 = Some(Action::Ready);
    app.update();
    assert!(app.world().get::<Camera>(cameras[1]).unwrap().is_active);
    assert!(app
        .world()
        .get::<Cascades>(light)
        .unwrap()
        .cascades
        .contains_key(&cameras[0]));
    assert!(!app
        .world()
        .get::<Cascades>(light)
        .unwrap()
        .cascades
        .contains_key(&cameras[1]));
}

#[test]
fn camera_reactivation_always_has_current_cascades_and_late_submission_still_suspends() {
    let (mut app, cameras, light) = fixture();
    configure(&mut app);
    app.world_mut().resource_mut::<LateChange>().0 = Some(Action::Ready);
    app.update();
    check(&app, cameras, light, [false, false]);
    app.update();
    check(&app, cameras, light, [true, true]);
    for id in 1..=64 {
        app.world_mut().resource_mut::<CaptureProgress>().pending = Some(id);
        // Submission may complete between main-world visibility and extraction.
        app.world_mut().resource_mut::<LateChange>().0 = Some(Action::Submitted(cameras[0], id));
        app.update();
        check(&app, cameras, light, [false, true]);
        app.world_mut().resource_mut::<LateChange>().0 = Some(Action::Next);
        app.update();
        check(&app, cameras, light, [false, true]);
        app.update();
        check(&app, cameras, light, [true, true]);
    }
    app.world_mut().resource_mut::<CaptureFailure>().0 = Some("test failure".into());
    app.update();
    check(&app, cameras, light, [false, false]);
}
