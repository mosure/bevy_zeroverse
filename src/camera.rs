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
    math::{
        primitives::{
            Circle,
            Sphere,
        },
        sampling::ShapeSample,
    },
    render::{
        camera::{
            Exposure,
            RenderTarget,
        },
        renderer::RenderDevice,
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
use rand::Rng;

use crate::{
    app::BevyZeroverseConfig,
    io,
    plucker::PluckerCamera,
};


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


#[derive(Clone, Debug, Reflect)]
pub enum LookingAtSampler {
    Exact(Vec3),
    Sphere{
        geometry: Sphere,
        transform: Transform,
    },
}

impl Default for LookingAtSampler {
    fn default() -> Self {
        Self::Exact(Vec3::ZERO)
    }
}

impl LookingAtSampler {
    pub fn sample(&self) -> Vec3 {
        match *self {
            LookingAtSampler::Exact(pos) => pos,
            LookingAtSampler::Sphere { geometry, transform } => {
                let rng = &mut rand::thread_rng();
                transform * geometry.sample_interior(rng)
            },
        }
    }

    pub fn look_at(&self, mut transform: Transform) -> Transform {
        transform.look_at(self.sample(), Vec3::Y);
        transform
    }
}

// TODO: add gizmos for sampler types
#[derive(Debug, Reflect)]
pub enum CameraPositionSamplerType {
    Band {
        size: Vec3,
        rotation: Quat,
        translate: Vec3,
    },
    Circle {
        radius: f32,
        rotation: Quat,
    },
    Sphere {
        radius: f32,
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

#[derive(Debug, Reflect, Default)]
pub struct CameraPositionSampler {
    pub looking_at: LookingAtSampler,
    pub sampler_type: CameraPositionSamplerType,
}

impl Default for CameraPositionSamplerType {
    fn default() -> Self {
        Self::Transform(Transform::default())
    }
}

impl CameraPositionSampler {
    pub fn sample(&self) -> Transform {
        let transform = match self.sampler_type {
            CameraPositionSamplerType::Band { size, rotation, translate } => {
                let rng = &mut rand::thread_rng();

                let face = rng.gen_range(0..4);

                let (x, z) = match face {
                    0 => (-size.x / 2.0, rng.gen_range(-size.z / 2.0..size.z / 2.0)),
                    1 => (size.x / 2.0, rng.gen_range(-size.z / 2.0..size.z / 2.0)),
                    2 => (rng.gen_range(-size.x / 2.0..size.x / 2.0), -size.z / 2.0),
                    3 => (rng.gen_range(-size.x / 2.0..size.x / 2.0), size.z / 2.0),
                    _ => unreachable!(),
                };

                let y = rng.gen_range(-size.y / 2.0..size.y / 2.0);
                let pos = Vec3::new(x, y, z) + translate;
                let pos = rotation.mul_vec3(pos);

                Transform::from_translation(pos)
            },
            CameraPositionSamplerType::Circle { radius, rotation } => {
                let rng = &mut rand::thread_rng();

                let xz = Circle::new(radius).sample_boundary(rng);
                let pos = rotation.mul_vec3(Vec3::new(xz.x, 0.0, xz.y));

                Transform::from_translation(pos)
            },
            CameraPositionSamplerType::Sphere { radius } => {
                let rng = &mut rand::thread_rng();

                let pos = Sphere::new(radius).sample_boundary(rng);

                Transform::from_translation(pos)
            },
            CameraPositionSamplerType::Transform(transform) => transform,
            _ => Transform::default(),
        };

        self.looking_at.look_at(transform)
    }
}


#[derive(Component, Debug, Default, Reflect)]
pub struct ZeroverseCamera {
    pub looking_at: LookingAtSampler,
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
    default_zeroverse_camera: Res<DefaultZeroverseCamera>,
    args: Res<BevyZeroverseConfig>,
    render_device: Res<RenderDevice>,
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
                label: "bevy_zeroverse_camera_target".into(),
                size,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                mip_level_count: 1,
                sample_count: 1,
                usage: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST
                    | TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            },
            ..default()
        };
        render_target.resize(size);
        let render_target = images.add(render_target);
        let target = RenderTarget::Image(render_target.clone());

        let mut camera = commands.entity(entity);
        camera
            .insert((
                Camera3dBundle {
                    camera: Camera {
                        hdr: false,
                        target,
                        ..default()
                    },
                    camera_3d: Camera3d {
                        screen_space_specular_transmission_quality: ScreenSpaceTransmissionQuality::High,
                        ..default()
                    },
                    exposure: Exposure::INDOOR,
                    projection: Projection::Perspective(
                        PerspectiveProjection {
                            fov: 60.0 * std::f32::consts::PI / 180.0,
                            ..default()
                        },
                    ),
                    transform: zeroverse_camera.sampler.sample(),
                    tonemapping: Tonemapping::None,
                    ..default()
                },
                PluckerCamera,
                BloomSettings::default(),
                Name::new("zeroverse_camera"),
            ));

        if args.image_copiers {
            let mut cpu_image = Image {
                texture_descriptor: TextureDescriptor {
                    label: "bevy_zeroverse_camera_cpu_image".into(),
                    size,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba32Float,
                    mip_level_count: 1,
                    sample_count: 1,
                    usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                ..Default::default()
            };
            cpu_image.resize(size);
            let cpu_image_handle = images.add(cpu_image);

            camera.insert(io::image_copy::ImageCopier::new(
                render_target,
                cpu_image_handle,
                size,
                TextureFormat::Rgba32Float,
                &render_device,
            ));
        }
    }
}


#[derive(Component, Debug, Default, Reflect)]
pub struct EditorCameraMarker {
    pub transform: Option<Transform>,
}


#[derive(Component, Debug, Reflect)]
pub struct ProcessedEditorCameraMarker;

#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct EditorCameraGizmoConfigGroup;

pub const EDITOR_CAMERA_RENDER_LAYER: RenderLayers = RenderLayers::layer(1);

fn setup_editor_camera(
    mut commands: Commands,
    editor_cameras: Query<
        (Entity, &EditorCameraMarker),
        Without<ProcessedEditorCameraMarker>,
    >,
) {
    for (entity, marker) in editor_cameras.iter() {
        let render_layer = RenderLayers::default().union(&EDITOR_CAMERA_RENDER_LAYER);
        commands.entity(entity)
            .insert((
                Camera3dBundle {
                    camera: Camera {
                        hdr: false,
                        ..default()
                    },
                    camera_3d: Camera3d {
                        screen_space_specular_transmission_quality: ScreenSpaceTransmissionQuality::High,
                        ..default()
                    },
                    exposure: Exposure::INDOOR,
                    transform: marker.transform.unwrap_or_default(),
                    tonemapping: Tonemapping::None,
                    ..default()
                },
                BloomSettings::default(),
                PluckerCamera,
            ))
            .insert(render_layer)
            .insert(ProcessedEditorCameraMarker)
            .insert(Name::new("editor_camera"));
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
