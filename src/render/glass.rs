//! Explicit geometric annotation semantics for transmissive surfaces.
use bevy::prelude::*;
use bevy_args::{Deserialize, Serialize, ValueEnum};

/// RGB transmission remains independent of geometric annotations. Through mode
/// follows unrefracted rays past every material with specular transmission > 0.
/// Depth, normal, position, semantic, flow and co-visibility share these hits.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Reflect, Serialize, Deserialize, ValueEnum,
)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "python", pyo3::pyclass(eq, eq_int))]
pub enum AnnotationGlass {
    #[default]
    Surface,
    Through,
}

impl AnnotationGlass {
    pub(crate) fn includes(self, material: Option<&StandardMaterial>) -> bool {
        self == Self::Surface || material.is_none_or(|m| m.specular_transmission <= 0.)
    }
}

#[derive(Component)]
pub(super) struct HiddenGlass(Visibility);

/// The viewer's material-based annotation previews use the same surface policy
/// as dataset MRT extraction. Save and restore visibility without touching RGB
/// materials, user-hidden geometry, semantic labels or voxel membership.
#[allow(clippy::type_complexity)]
pub(super) fn preview(
    mut commands: Commands,
    config: Res<crate::app::BevyZeroverseConfig>,
    mode: Res<super::RenderMode>,
    materials: Res<Assets<StandardMaterial>>,
    mut events: MessageReader<bevy::asset::AssetEvent<StandardMaterial>>,
    hidden_objects: Query<(Entity, &HiddenGlass)>,
    objects: Query<
        (
            Entity,
            Ref<Mesh3d>,
            Option<Ref<MeshMaterial3d<StandardMaterial>>>,
            Option<Ref<super::DisabledPbrMaterial>>,
            &Visibility,
            Option<&HiddenGlass>,
        ),
        Without<super::RenderOnlyOverlay>,
    >,
) {
    let modified: Vec<_> = events
        .read()
        .filter_map(|event| match event {
            bevy::asset::AssetEvent::Added { id }
            | bevy::asset::AssetEvent::Modified { id }
            | bevy::asset::AssetEvent::Removed { id } => Some(*id),
            _ => None,
        })
        .collect();
    let through = config.annotation_glass == AnnotationGlass::Through
        && !matches!(
            *mode,
            super::RenderMode::Color | super::RenderMode::CoVisibility
        );
    if !through {
        for (entity, hidden) in &hidden_objects {
            commands
                .entity(entity)
                .insert(hidden.0)
                .remove::<HiddenGlass>();
        }
        return;
    }
    for (entity, mesh, material, disabled, visibility, hidden) in &objects {
        let handle = material
            .as_ref()
            .map(|m| &m.0)
            .or_else(|| disabled.as_ref().map(|m| &m.material));
        if !mode.is_changed()
            && !config.is_changed()
            && !mesh.is_added()
            && !material.as_ref().is_some_and(|m| m.is_changed())
            && !disabled.as_ref().is_some_and(|m| m.is_changed())
            && !handle.is_some_and(|h| modified.contains(&h.id()))
        {
            continue;
        }
        let glass = !config
            .annotation_glass
            .includes(handle.and_then(|h| materials.get(h)));
        if glass && hidden.is_none() {
            commands
                .entity(entity)
                .insert((HiddenGlass(*visibility), Visibility::Hidden));
        } else if let Some(hidden) = hidden.filter(|_| !glass) {
            commands
                .entity(entity)
                .insert(hidden.0)
                .remove::<HiddenGlass>();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn opaque_and_unknown_surfaces_are_preserved() {
        let opaque = StandardMaterial::default();
        let glass = StandardMaterial {
            specular_transmission: 0.7,
            ..default()
        };
        assert!(AnnotationGlass::Surface.includes(Some(&glass)));
        assert!(!AnnotationGlass::Through.includes(Some(&glass)));
        assert!(AnnotationGlass::Through.includes(Some(&opaque)));
        assert!(AnnotationGlass::Through.includes(None));
    }
    #[test]
    fn preview_restores_rgb_and_preserves_user_hidden_geometry() {
        let mut app = App::new();
        app.add_message::<bevy::asset::AssetEvent<StandardMaterial>>();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .insert_resource(crate::app::BevyZeroverseConfig {
                annotation_glass: AnnotationGlass::Through,
                ..default()
            })
            .insert_resource(super::super::RenderMode::Depth)
            .add_systems(Update, preview);
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                specular_transmission: 0.8,
                ..default()
            });
        let entities: Vec<_> = [Visibility::Inherited, Visibility::Hidden]
            .into_iter()
            .map(|visibility| {
                app.world_mut()
                    .spawn((
                        Mesh3d::default(),
                        MeshMaterial3d(material.clone()),
                        visibility,
                    ))
                    .id()
            })
            .collect();
        app.update();
        for entity in &entities {
            assert_eq!(
                *app.world().get::<Visibility>(*entity).unwrap(),
                Visibility::Hidden
            );
        }
        // Material edits change hits even when the component's handle is unchanged.
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .get_mut(&material)
            .unwrap()
            .specular_transmission = 0.;
        app.world_mut()
            .write_message(bevy::asset::AssetEvent::Modified { id: material.id() });
        app.update();
        assert_eq!(
            *app.world().get::<Visibility>(entities[0]).unwrap(),
            Visibility::Inherited
        );
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .get_mut(&material)
            .unwrap()
            .specular_transmission = 0.8;
        app.world_mut()
            .write_message(bevy::asset::AssetEvent::Modified { id: material.id() });
        app.update();
        assert_eq!(
            *app.world().get::<Visibility>(entities[0]).unwrap(),
            Visibility::Hidden
        );
        *app.world_mut().resource_mut::<super::super::RenderMode>() =
            super::super::RenderMode::Color;
        app.update();
        assert_eq!(
            *app.world().get::<Visibility>(entities[0]).unwrap(),
            Visibility::Inherited
        );
        assert_eq!(
            *app.world().get::<Visibility>(entities[1]).unwrap(),
            Visibility::Hidden
        );
        for entity in entities {
            assert!(app.world().get::<HiddenGlass>(entity).is_none());
        }
    }
}
