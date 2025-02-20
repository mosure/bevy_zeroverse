use bevy::{
    prelude::*,
    core_pipeline::{
        bloom::Bloom,
        core_3d::ScreenSpaceTransmissionQuality,
        prepass::MotionVectorPrepass,
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
    render::RenderMode,
};



// TODO: support camera trajectories, requires custom motion vector prepass during capture
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

        app.init_resource::<Playback>();
        app.register_type::<Playback>();

        app.add_systems(
            Update,
            (
                draw_camera_gizmo,
                insert_cameras,
                setup_editor_camera,
                update_camera_trajectory,
                update_tonemapping,
            )
        );
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
#[derive(Clone, Debug, Reflect)]
pub enum ExtrinsicsSamplerType {
    Band {
        size: Vec3,
        rotation: Quat,
        translate: Vec3,
    },
    Circle {
        radius: f32,
        rotation: Quat,
        translate: Vec3,
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

#[derive(Clone, Debug, Reflect, Default)]
pub struct ExtrinsicsSampler {
    pub cache: Option<Transform>,
    pub looking_at: LookingAtSampler,
    pub position: ExtrinsicsSamplerType,  // TODO: use position sampler here
}

impl Default for ExtrinsicsSamplerType {
    fn default() -> Self {
        Self::Transform(Transform::default())
    }
}

impl ExtrinsicsSampler {
    pub fn sample_cache(&mut self) -> Transform {
        if let Some(transform) = self.cache {
            return transform;
        }

        let transform = self.sample();
        self.cache = Some(transform);
        transform
    }

    pub fn sample(&self) -> Transform {
        let transform: Transform = match self.position {
            ExtrinsicsSamplerType::Band { size, rotation, translate } => {
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
            ExtrinsicsSamplerType::Circle { radius, rotation, translate } => {
                let rng = &mut rand::thread_rng();

                let xz = Circle::new(radius).sample_boundary(rng);
                let pos = rotation.mul_vec3(Vec3::new(xz.x, 0.0, xz.y)) + translate;

                Transform::from_translation(pos)
            },
            ExtrinsicsSamplerType::Sphere { radius } => {
                let rng = &mut rand::thread_rng();

                let pos = Sphere::new(radius).sample_boundary(rng);

                Transform::from_translation(pos)
            },
            ExtrinsicsSamplerType::Transform(transform) => transform,
            _ => Transform::default(),
        };

        self.looking_at.look_at(transform)
    }
}


#[derive(Component, Debug, Reflect)]
pub struct PerspectiveSampler {
    pub min_fov_deg: f32,
    pub max_fov_deg: f32,
}

impl Default for PerspectiveSampler {
    fn default() -> Self {
        Self {
            min_fov_deg: 20.0,
            max_fov_deg: 90.0,
        }
    }
}

impl PerspectiveSampler {
    pub fn sample(&self) -> PerspectiveProjection {
        let rng = &mut rand::thread_rng();
        let fov_deg = rng.gen_range(self.min_fov_deg..self.max_fov_deg);
        let fov = fov_deg * std::f32::consts::PI / 180.0;
        PerspectiveProjection {
            fov,
            ..default()
        }
    }
}


#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    Hash,
    PartialEq,
    Reflect,
)]
pub enum PlaybackMode {
    Loop,
    Once,
    PingPong,
    Sin,
    #[default]
    Still,
}

#[derive(
    Resource,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Reflect,
)]
#[reflect(Resource)]
pub struct Playback {
    pub mode: PlaybackMode,
    pub progress: f32,
    pub direction: f32,
    pub speed: f32,
}

impl Default for Playback {
    fn default() -> Self {
        Self {
            mode: PlaybackMode::default(),
            progress: 0.0,
            direction: 1.0,
            speed: 1.0,
        }
    }
}

impl Playback {
    pub fn step(&mut self, time: &Res<Time>) {
        if self.speed == 0.0 {
            return;
        }

        // bail condition
        match self.mode {
            PlaybackMode::Loop => {}
            PlaybackMode::Once => {
                if self.progress >= 1.0 {
                    return;
                }
            }
            PlaybackMode::PingPong => {}
            PlaybackMode::Sin => {}
            PlaybackMode::Still => {
                return;
            }
        }

        // forward condition
        match self.mode {
            PlaybackMode::Loop | PlaybackMode::Once | PlaybackMode::PingPong => {
                self.progress += time.delta_secs() * self.direction * self.speed;
            }
            PlaybackMode::Sin => {
                let theta = self.direction * self.speed * time.elapsed_secs();
                let y = (theta * 2.0 * std::f32::consts::PI).sin();
                self.progress = (y + 1.0) / 2.0;
            }
            PlaybackMode::Still => {}
        }

        // reset condition
        match self.mode {
            PlaybackMode::Loop => {
                if self.progress > 1.0 {
                    self.progress = 0.0;
                }
            }
            PlaybackMode::Once => {}
            PlaybackMode::PingPong => {
                if self.progress > 1.0 {
                    self.progress = 1.0;
                    self.direction = -self.direction;
                } else if self.progress < 0.0 {
                    self.progress = 0.0;
                    self.direction = -self.direction;
                }
            }
            PlaybackMode::Sin => {}
            PlaybackMode::Still => {}
        }
    }
}



#[derive(Debug, Reflect)]
pub enum TrajectorySampler {
    Static {
        start: ExtrinsicsSampler,
    },
    Linear {
        start: ExtrinsicsSampler,
        end: ExtrinsicsSampler,
    },
    // TODO: add arc, spline, etc.
}

impl Default for TrajectorySampler {
    fn default() -> Self {
        Self::Static {
            start: ExtrinsicsSampler::default(),
        }
    }
}

impl TrajectorySampler {
    pub fn sample(&mut self, progress: f32) -> Transform {
        match self {
            TrajectorySampler::Static { start } => {
                start.sample_cache()
            },
            TrajectorySampler::Linear { start, end } => {
                let progress = progress.clamp(0.0, 1.0);
                let start = start.sample_cache();
                let end = end.sample_cache();
                let pos = start.translation.lerp(end.translation, progress);
                let rot = start.rotation.slerp(end.rotation, progress);
                Transform::from_translation(pos).mul_transform(Transform::from_rotation(rot))
            },
        }
    }
}


#[derive(Component, Debug, Default, Reflect)]
pub struct ZeroverseCamera {
    pub perspective_sampler: PerspectiveSampler,
    pub resolution: Option<UVec2>,
    pub override_transform: Option<Transform>,
    pub trajectory: TrajectorySampler,
    pub playback: Playback,
}

#[derive(Resource, Debug, Default, Reflect)]
pub struct DefaultZeroverseCamera {
    pub resolution: Option<UVec2>,
}


fn update_camera_trajectory(
    mut cameras: Query<(
        &mut Transform,
        &mut ZeroverseCamera,
    )>,
    mut global_playback: ResMut<Playback>,
    time: Res<Time>,
) {
    let update_camera_playbacks = global_playback.is_changed() || global_playback.mode != PlaybackMode::Still;
    global_playback.step(&time);

    for (
        mut transform,
        mut camera,
    ) in cameras.iter_mut() {
        if update_camera_playbacks {
            camera.playback = *global_playback;
        } else {
            camera.playback.step(&time);
        }

        let playback = camera.playback;
        *transform = camera.trajectory.sample(playback.progress);
    }
}


fn insert_cameras(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut zeroverse_cameras: Query<
        (
            Entity,
            &mut ZeroverseCamera,
        ),
        Without<Camera>,
    >,
    default_zeroverse_camera: Res<DefaultZeroverseCamera>,
    args: Res<BevyZeroverseConfig>,
    render_mode: Res<RenderMode>,
    render_device: Res<RenderDevice>,
) {
    for (
        entity,
        mut zeroverse_camera,
    ) in zeroverse_cameras.iter_mut() {
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

        // TODO: modulate fov
        let mut camera = commands.entity(entity);
        camera
            .insert((
                Camera3d {
                    screen_space_specular_transmission_quality: ScreenSpaceTransmissionQuality::High,
                    ..default()
                },
                Camera {
                    hdr: true,
                    target,
                    ..default()
                },
                Exposure::INDOOR,
                Projection::Perspective(zeroverse_camera.perspective_sampler.sample()),
                zeroverse_camera.override_transform.unwrap_or(zeroverse_camera.trajectory.sample(0.0)),
                Bloom::default(),
                MotionVectorPrepass,
                render_mode.tonemapping(),
                PluckerCamera,
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
    render_mode: Res<RenderMode>,
) {
    for (entity, marker) in editor_cameras.iter() {
        let render_layer = RenderLayers::default().union(&EDITOR_CAMERA_RENDER_LAYER);
        commands.entity(entity)
            .insert((
                Camera3d {
                    screen_space_specular_transmission_quality: ScreenSpaceTransmissionQuality::High,
                    ..default()
                },
                Camera {
                    hdr: true,
                    ..default()
                },
                Exposure::INDOOR,
                marker.transform.unwrap_or_default(),
                render_mode.tonemapping(),
                Bloom::default(),
                MotionVectorPrepass,
                PluckerCamera,
            ))
            .insert(render_layer)
            .insert(ProcessedEditorCameraMarker)
            .insert(Name::new("editor_camera"));
    }
}


pub fn update_tonemapping(
    mut commands: Commands,
    editor_cameras: Query<
        Entity,
        With<ProcessedEditorCameraMarker>,
    >,
    zeroverse_cameras: Query<
        Entity,
        With<ZeroverseCamera>,
    >,
    render_mode: Res<RenderMode>,
) {
    if !render_mode.is_changed() {
        return;
    }

    for camera_entity in editor_cameras
        .iter()
        .chain(zeroverse_cameras.iter())
    {
        commands
            .entity(camera_entity)
            .insert(render_mode.tonemapping());
    }
}


#[allow(clippy::type_complexity)]
pub fn draw_camera_gizmo(
    args: Res<BevyZeroverseConfig>,
    mut gizmos: Gizmos<EditorCameraGizmoConfigGroup>,
    cameras: Query<
        (&GlobalTransform, &Projection),
        (
            With<Camera>,
            Without<EditorCameraMarker>,
        ),
    >,
) {
    if !args.gizmos {
        return;
    }

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
            rect_transform.to_isometry(),
            Vec2::new(tan_half_fov_x * 2.0 * scale, tan_half_fov_y * 2.0 * scale),
            color,
        );
    }
}
