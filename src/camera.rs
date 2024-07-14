use bevy::{
    prelude::*,
    core_pipeline::{
        bloom::BloomSettings,
        core_3d::ScreenSpaceTransmissionQuality,
        tonemapping::Tonemapping,
    },
    gizmos::config::{
        GizmoConfig,
        GizmoConfigGroup,
    },
    render::{
        camera::{
            Exposure,
            RenderTarget,
        },
        render_resource::{
            Extent3d,
            TextureDescriptor,
            TextureDimension,
            TextureFormat,
            TextureUsages,
        },
        view::RenderLayers,
    }
};

use crate::plucker::PluckerCamera;


pub struct ZeroverseCameraPlugin;
impl Plugin for ZeroverseCameraPlugin {
    fn build(&self, app: &mut App) {
        app.insert_gizmo_config(
            EditorCameraGizmoConfigGroup,
            GizmoConfig {
                render_layers: EDITOR_CAMERA_RENDER_LAYER,
                ..default()
            },
        );

        app.init_resource::<DefaultZeroverseCamera>();
        app.init_resource::<EnvironmentMapResource>();

        app.add_systems(PreStartup, load_environment_map);
        app.add_systems(
            Update,
            (
                setup_editor_camera,
                insert_cameras,
            )
        );
        app.add_systems(Update, draw_camera_gizmo);
    }
}


#[derive(Resource, Default, Debug, Reflect)]
pub struct EnvironmentMapResource {
    diffuse_map: Handle<Image>,
    specular_map: Handle<Image>,
}

// TODO: support multiple environment maps
fn load_environment_map(
    asset_server: Res<AssetServer>,
    mut environment_map: ResMut<EnvironmentMapResource>,
) {
    environment_map.diffuse_map = asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2");
    environment_map.specular_map = asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2");
}


#[derive(Debug, Reflect)]
pub enum CameraPositionSampler {
    Circle {
        radius: f32,
        rotation: Quat,
    },
    Cuboid {
        size: Vec3,
        rotation: Quat,
    },
    Ellipsoid {
        radius: Vec3,
        rotation: Quat,
    },
    Transform(Transform),
}

impl Default for CameraPositionSampler {
    fn default() -> Self {
        Self::Transform(Transform::default())
    }
}

#[derive(Component, Debug, Default, Reflect)]
pub struct ZeroverseCamera {
    pub sampler: CameraPositionSampler,
    pub resolution: Option<UVec2>,
}

#[derive(Resource, Debug, Default, Reflect)]
pub struct DefaultZeroverseCamera {
    pub resolution: Option<UVec2>,
}


fn insert_cameras(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    zeroverse_cameras: Query<
        (
            Entity,
            &ZeroverseCamera,
        ),
        Without<Camera>,
    >,
    environment_map: Res<EnvironmentMapResource>,
    default_zeroverse_camera: Res<DefaultZeroverseCamera>,
) {
    for (entity, zeroverse_camera) in zeroverse_cameras.iter() {
        let resolution = zeroverse_camera.resolution
            .unwrap_or(default_zeroverse_camera.resolution.expect("DefaultZeroverseCamera resolution must be set if ZeroverseCamera resolution is not set"));

        let size = Extent3d {
            width: resolution.x,
            height: resolution.y,
            depth_or_array_layers: 1,
        };

        let mut render_target = Image {
            texture_descriptor: TextureDescriptor {
                label: None,
                size,
                dimension: TextureDimension::D2,
                format: TextureFormat::Bgra8UnormSrgb,
                mip_level_count: 1,
                sample_count: 1,
                usage: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_DST
                    | TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            },
            ..default()
        };
        render_target.resize(size);
        let render_target = images.add(render_target);
        let target = RenderTarget::Image(render_target);

        // TODO: perform sampling
        let transform = match zeroverse_camera.sampler {
            CameraPositionSampler::Transform(transform) => transform,
            _ => Transform::default(),
        };

        commands.entity(entity).insert((
            Camera3dBundle {
                camera: Camera {
                    hdr: true,
                    target,
                    ..default()
                },
                camera_3d: Camera3d {
                    screen_space_specular_transmission_quality: ScreenSpaceTransmissionQuality::High,
                    ..default()
                },
                exposure: Exposure::BLENDER,
                transform,
                tonemapping: Tonemapping::None,
                ..default()
            },
            BloomSettings::default(),
            PluckerCamera,
            EnvironmentMapLight {
                diffuse_map: environment_map.diffuse_map.clone(),
                specular_map: environment_map.specular_map.clone(),
                intensity: 900.0,
            },
        ));
    }
}


#[derive(Component, Debug, Reflect)]
pub struct EditorCameraMarker;

#[derive(Component, Debug, Reflect)]
pub struct ProcessedEditorCameraMarker;

#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct EditorCameraGizmoConfigGroup;

pub const EDITOR_CAMERA_RENDER_LAYER: RenderLayers = RenderLayers::layer(1);

fn setup_editor_camera(
    mut commands: Commands,
    editor_cameras: Query<
        Entity,
        (With<EditorCameraMarker>, Without<ProcessedEditorCameraMarker>),
    >,
    environment_map: Res<EnvironmentMapResource>,
) {
    for entity in editor_cameras.iter() {
        let render_layer = RenderLayers::default().union(&EDITOR_CAMERA_RENDER_LAYER);
        commands.entity(entity)
            .insert((
                Camera3dBundle {
                    camera: Camera {
                        hdr: true,
                        ..default()
                    },
                    camera_3d: Camera3d {
                        screen_space_specular_transmission_quality: ScreenSpaceTransmissionQuality::High,
                        ..default()
                    },
                    exposure: Exposure::BLENDER,
                    transform: Transform::default(),
                    tonemapping: Tonemapping::None,
                    ..default()
                },
                BloomSettings::default(),
                PluckerCamera,
                EnvironmentMapLight {
                    diffuse_map: environment_map.diffuse_map.clone(),
                    specular_map: environment_map.specular_map.clone(),
                    intensity: 900.0,
                },
            ))
            .insert(render_layer)
            .insert(ProcessedEditorCameraMarker);
    }
}


#[allow(clippy::type_complexity)]
pub fn draw_camera_gizmo(
    mut gizmos: Gizmos<EditorCameraGizmoConfigGroup>,
    cameras: Query<
        (&GlobalTransform, &Projection),
        (
            With<Camera>,
            Without<EditorCameraMarker>,
        ),
    >,
) {
    let color = Color::srgb(1.0, 0.0, 1.0);

    for (global_transform, projection) in cameras.iter() {
        let transform = global_transform.compute_transform();

        // let cuboid_transform = transform.with_scale(Vec3::new(0.5, 0.5, 1.0));
        // gizmos.cuboid(cuboid_transform, color);

        let (aspect_ratio, fov_y) = match projection {
            Projection::Perspective(persp) => (persp.aspect_ratio, persp.fov),
            Projection::Orthographic(_) => {
                (1.0, 0.0)
            }
        };

        let tan_half_fov_y = (fov_y * 0.5).tan();
        let tan_half_fov_x = tan_half_fov_y * aspect_ratio;

        let scale = 1.5;

        let forward = transform.forward() * scale;
        let up = transform.up() * tan_half_fov_y * scale;
        let right = transform.right() * tan_half_fov_x * scale;

        gizmos.line(
            transform.translation,
            transform.translation + forward + up + right,
            color,
        );
        gizmos.line(
            transform.translation,
            transform.translation + forward - up + right,
            color,
        );
        gizmos.line(
            transform.translation,
            transform.translation + forward + up - right,
            color,
        );
        gizmos.line(
            transform.translation,
            transform.translation + forward - up - right,
            color,
        );

        let rect_transform = Transform::from_translation(transform.translation + forward);
        let rect_transform = rect_transform.mul_transform(Transform::from_rotation(transform.rotation));
        gizmos.rect(
            rect_transform.translation,
            rect_transform.rotation,
            Vec2::new(tan_half_fov_x * 2.0 * scale, tan_half_fov_y * 2.0 * scale),
            color,
        );
    }
}
