use super::*;
use crate::{
    camera::{Playback, PlaybackMode},
    scene::procedural_indoor::motion::{AnimatedPart, IndoorMotionPlugin},
};
use bevy::camera::primitives::Aabb;

fn attributes(mesh: &Mesh) -> Vec<Vec<u8>> {
    [
        Mesh::ATTRIBUTE_POSITION,
        Mesh::ATTRIBUTE_NORMAL,
        Mesh::ATTRIBUTE_TANGENT,
        Mesh::ATTRIBUTE_UV_0,
    ]
    .map(|key| mesh.attribute(key).unwrap().get_bytes().to_vec())
    .into()
}

#[test]
fn real_window_coverings_restore_their_authored_shading_and_correspondence_when_scrubbed() {
    let mut count = 0;
    let mut authored = 0;
    for kind in KINDS {
        let f = facade(19, kind, Vec2::new(2.3, 1.8), 0.66);
        let mut a = Assembly {
            motion_policy: Some(IndoorMotionConfig {
                covering_fraction: 1.,
                ..default()
            }),
            ..default()
        };
        build(&mut a, Transform::IDENTITY, &f, &f.openings[0]);
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .insert_resource(Playback {
                mode: PlaybackMode::Once,
                ..default()
            })
            .add_plugins(IndoorMotionPlugin);
        let mut expected = vec![];
        for joint in &a.joints {
            for (key, end) in &joint.morph {
                let start = &a.parts[key];
                let animated = AnimatedPart::new(joint, key, start);
                let mesh = start.clone().into_mesh();
                let initial = attributes(&mesh);
                let mut averaged = mesh.clone();
                averaged.compute_smooth_normals();
                let averaged = averaged
                    .attribute(Mesh::ATTRIBUTE_NORMAL)
                    .unwrap()
                    .as_float3()
                    .unwrap();
                authored += start
                    .normals
                    .iter()
                    .enumerate()
                    .filter(|(v, n)| {
                        start.positions[*v] != end.positions[*v] && **n != averaged[*v]
                    })
                    .count();
                let mut end = end.clone();
                end.uvs.clone_from(&start.uvs);
                let final_pose = attributes(&end.into_mesh());
                let handle = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
                app.world_mut().spawn((
                    animated,
                    Mesh3d(handle.clone()),
                    Transform::IDENTITY,
                    Aabb::from_min_max(Vec3::ZERO, Vec3::ONE),
                ));
                expected.push((key, handle, initial, final_pose));
            }
        }
        if expected.is_empty() {
            continue;
        }
        count += 1;
        for progress in [0.43, 1., 0.63, 0., 0.43, 1., 0.] {
            app.world_mut().resource_mut::<Playback>().progress = progress;
            app.update();
            assert!(app
                .world()
                .resource::<crate::sample::CaptureFailure>()
                .0
                .is_none());
            for (key, handle, initial, final_pose) in &expected {
                let mesh = app.world().resource::<Assets<Mesh>>().get(handle).unwrap();
                let actual = attributes(mesh);
                if progress == 0. || progress == 1. {
                    let frame = if progress == 0. { initial } else { final_pose };
                    for (i, (a, b)) in actual.iter().zip(frame).enumerate() {
                        assert!(
                            a == b,
                            "{kind:?}, {key:?}: endpoint {progress}, attribute {i}"
                        );
                    }
                } else {
                    assert_eq!(actual[0].len(), initial[0].len());
                    assert!(actual[3] == initial[3]);
                    assert!(actual
                        .iter()
                        .flat_map(|bytes| bytes.as_chunks::<4>().0)
                        .all(|v| f32::from_ne_bytes(*v).is_finite()));
                }
            }
        }
    }
    eprintln!("{count} window covering families, {authored} moving vertices with authored normals distinct from triangle averages");
    assert!(count >= 5 && authored > 100);
}
