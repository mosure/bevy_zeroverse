use super::*;
use bevy::{ecs::system::RunSystemOnce, render::MainWorld};

#[derive(Component)]
struct Attachment;

#[test]
fn extraction_suppresses_only_submitted_owner_and_faces_and_restores_next_epoch() {
    for (headless, editor, suppressed) in [
        (true, false, true),
        (false, false, false),
        (true, true, false),
    ] {
        let mut main = World::new();
        main.insert_resource(BevyZeroverseConfig {
            headless,
            editor,
            ..default()
        });
        let mut render = World::new();
        let views: Vec<_> = (0..7)
            .map(|_| {
                render
                    .spawn((
                        Projection::default(),
                        NoIndirectDrawing,
                        RenderVisibleEntities::default(),
                        Attachment,
                    ))
                    .id()
            })
            .collect();
        let mut owners = Vec::new();
        for i in 0..3 {
            let copier = ImageCopier::test_idle();
            if i != 2 {
                copier.request(if i == 0 { 8 } else { 9 });
                copier.test_submit(8);
            }
            let owner = main
                .spawn((
                    ZeroverseCamera::default(),
                    copier,
                    RenderEntity::from(views[i]),
                ))
                .id();
            main.spawn((
                crate::camera::lens::LensFace { owner },
                RenderEntity::from(views[i + 3]),
            ));
            owners.push(owner);
        }
        let mut main_world = MainWorld::default();
        *main_world = main;
        render.insert_resource(main_world);
        render.run_system_once(suppress).unwrap();
        for (i, view) in views.iter().enumerate() {
            assert!(
                render.get::<Attachment>(*view).is_some(),
                "capture attachments are retained"
            );
            assert_eq!(
                render.get::<Projection>(*view).is_none(),
                suppressed && (i == 0 || i == 3)
            );
        }
        // Simulate normal extraction restoring an active view on a new request.
        let copier = ImageCopier::test_idle();
        copier.request(9);
        copier.test_submit(8);
        render
            .resource_mut::<MainWorld>()
            .entity_mut(owners[0])
            .insert(copier);
        for view in [views[0], views[3]] {
            render.entity_mut(view).insert(Projection::default());
        }
        render.run_system_once(suppress).unwrap();
        assert!(render.get::<Projection>(views[0]).is_some());
        assert!(render.get::<Projection>(views[3]).is_some());
    }
}
