use super::*;
use bevy::{core_pipeline::schedule::Core3d, ecs::schedule::ScheduleLabel};

fn camera() -> ExtractedCamera {
    ExtractedCamera {
        target: None,
        physical_viewport_size: None,
        physical_target_size: None,
        viewport: None,
        schedule: Core3d.intern(),
        order: 0,
        output_mode: default(),
        msaa_writeback: default(),
        clear_color: default(),
        sorted_camera_index_for_target: 0,
        exposure: 1.,
        hdr: true,
    }
}

#[test]
fn idle_shadow_roots_are_restored_before_the_next_capture() {
    let mut world = World::new();
    world.insert_resource(Enabled(true));
    world.init_resource::<Suspended>();
    world.init_resource::<Counts>();
    let light = world.spawn_empty().id();
    let shadows: Vec<_> = (0..7)
        .map(|face| {
            world
                .spawn((
                    RootNonCameraView(Core3d.intern()),
                    if face < 6 {
                        LightEntity::Point {
                            light_entity: light,
                            face_index: face,
                        }
                    } else {
                        LightEntity::Spot {
                            light_entity: light,
                        }
                    },
                ))
                .id()
        })
        .collect();
    let other = world.spawn(RootNonCameraView(Core3d.intern())).id();
    for _ in 0..32 {
        suspend(&mut world);
        for shadow in &shadows {
            assert!(world.get::<RootNonCameraView>(*shadow).is_none());
            assert!(world.get::<LightEntity>(*shadow).is_some());
        }
        assert!(world.get::<RootNonCameraView>(other).is_some());
        restore(&mut world);
        for shadow in &shadows {
            assert_eq!(
                world.get::<RootNonCameraView>(*shadow).unwrap().0,
                Core3d.intern()
            );
        }
        // An owner or internal fisheye face makes all shadow roots run normally.
        let view = world.spawn(camera()).id();
        suspend(&mut world);
        assert!(world.resource::<Suspended>().0.is_empty());
        assert!(shadows
            .iter()
            .all(|s| world.get::<RootNonCameraView>(*s).is_some()));
        restore(&mut world);
        world.despawn(view);
    }
    let counts = &world.resource::<Counts>().0;
    assert_eq!(counts.skipped.load(Ordering::Relaxed), 32 * 7);
    assert_eq!(counts.camera_frames.load(Ordering::Relaxed), 32);
    assert_eq!(counts.camera_views.load(Ordering::Relaxed), 32);
    world.resource_mut::<Enabled>().0 = false;
    suspend(&mut world);
    assert!(world.resource::<Suspended>().0.is_empty());
    assert!(shadows
        .iter()
        .all(|s| world.get::<RootNonCameraView>(*s).is_some()));
}

#[test]
fn deferred_shadow_despawn_is_not_resurrected() {
    let mut world = World::new();
    world.insert_resource(Enabled(true));
    world.init_resource::<Suspended>();
    world.init_resource::<Counts>();
    let entity = world.spawn_empty().id();
    world.entity_mut(entity).insert((
        RootNonCameraView(Core3d.intern()),
        LightEntity::Spot {
            light_entity: entity,
        },
    ));
    suspend(&mut world);
    world.despawn(entity);
    restore(&mut world);
    assert!(world.get_entity(entity).is_err());
    assert!(world.resource::<Suspended>().0.is_empty());
}

#[test]
fn unready_draw_preserves_camera_order_and_resumes_without_repreparing_views() {
    use bevy::render::camera::SortedCamera;
    let mut world = World::new();
    world.insert_resource(Enabled(true));
    world.init_resource::<Suspended>();
    world.init_resource::<Counts>();
    let views: Vec<_> = (0..5).map(|_| world.spawn(camera()).id()).collect();
    world.insert_resource(SortedCameras(
        views
            .iter()
            .enumerate()
            .map(|(i, entity)| SortedCamera {
                entity: *entity,
                order: i as isize,
                target: None,
                output_mode: default(),
            })
            .collect(),
    ));
    for _ in 0..8 {
        suspend_views(&mut world, true);
        assert!(world.resource::<SortedCameras>().0.is_empty());
        assert!(views
            .iter()
            .all(|v| world.get::<ExtractedCamera>(*v).is_some()));
        restore(&mut world);
        assert_eq!(
            world
                .resource::<SortedCameras>()
                .0
                .iter()
                .map(|c| (c.entity, c.order))
                .collect::<Vec<_>>(),
            views
                .iter()
                .enumerate()
                .map(|(i, e)| (*e, i as isize))
                .collect::<Vec<_>>()
        );
    }
    suspend_views(&mut world, false);
    assert_eq!(world.resource::<SortedCameras>().0.len(), 5);
    assert!(world.resource::<Suspended>().1.is_none());
    assert_eq!(
        world
            .resource::<Counts>()
            .0
            .camera_frames
            .load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        world
            .resource::<Counts>()
            .0
            .camera_views
            .load(Ordering::Relaxed),
        5
    );
    assert_eq!(
        world.resource::<Counts>().0.unready.load(Ordering::Relaxed),
        8
    );
}
