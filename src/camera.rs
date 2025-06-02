use bevy::{
    prelude::*,
    core_pipeline::{
        bloom::Bloom,
        core_3d::ScreenSpaceTransmissionQuality,
        prepass::MotionVectorPrepass, // MOTION_VECTOR_PREPASS_FORMAT,
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
            ImageRenderTarget,
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
use bevy_args::{
    Deserialize,
    Parser,
    Serialize,
    ValueEnum,
};
use rand::Rng;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(not(feature = "web"))]
use bevy::core_pipeline::{
    // core_3d::CORE_3D_DEPTH_FORMAT,
    prepass::{
        // NORMAL_PREPASS_FORMAT,
        DepthPrepass,
        NormalPrepass,
    },
};

use crate::{
    app::BevyZeroverseConfig,
    io,
    // plucker::PluckerCamera,
    render::RenderMode,
    scene::{
        RegenerateSceneEvent,
        ZeroverseSceneRoot,
    },
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
                update_render_pipeline,
            )
        );
    }
}


// TODO: convert to position sampler
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
    BandShell {
        inner_size: Vec3,
        outer_size: Vec3,
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
        translate: Vec3,
    },
    SphereShell {
        inner_radius: f32,
        outer_radius: f32,
        translate: Vec3,
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
            ExtrinsicsSamplerType::BandShell {
                inner_size,
                outer_size,
                rotation,
                translate
            } => {
                let rng = &mut rand::thread_rng();

                let face = rng.gen_range(0..4);

                let (x, z) = match face {
                    0 => (
                        if rng.gen_bool(0.5) { outer_size.x / 2.0 } else { -outer_size.x / 2.0 },
                        rng.gen_range(-outer_size.z / 2.0..outer_size.z / 2.0),
                    ),
                    1 => (
                        if rng.gen_bool(0.5) { inner_size.x / 2.0 } else { -inner_size.x / 2.0 },
                        rng.gen_range(-inner_size.z / 2.0..inner_size.z / 2.0),
                    ),
                    2 => (
                        rng.gen_range(-outer_size.x / 2.0..outer_size.x / 2.0),
                        if rng.gen_bool(0.5) { outer_size.z / 2.0 } else { -outer_size.z / 2.0 },
                    ),
                    3 => (
                        rng.gen_range(-inner_size.x / 2.0..inner_size.x / 2.0),
                        if rng.gen_bool(0.5) { inner_size.z / 2.0 } else { -inner_size.z / 2.0 },
                    ),
                    _ => unreachable!(),
                };

                let y = rng.gen_range(-outer_size.y / 2.0..outer_size.y / 2.0);
                let pos = Vec3::new(x, y, z) + translate;
                let pos = rotation.mul_vec3(pos);
                Transform::from_translation(pos)
            }
            ExtrinsicsSamplerType::Circle { radius, rotation, translate } => {
                let rng = &mut rand::thread_rng();

                let xz = Circle::new(radius).sample_boundary(rng);
                let pos = rotation.mul_vec3(Vec3::new(xz.x, 0.0, xz.y)) + translate;
                Transform::from_translation(pos)
            },
            ExtrinsicsSamplerType::Sphere { radius, translate } => {
                let rng = &mut rand::thread_rng();

                let pos = Sphere::new(radius).sample_boundary(rng);
                Transform::from_translation(pos + translate)
            },
            ExtrinsicsSamplerType::SphereShell { inner_radius, outer_radius, translate } => {
                let rng = &mut rand::thread_rng();

                let radius = rng.gen_range(
                        inner_radius.powi(3)..outer_radius.powi(3)
                    ).powf(1.0 / 3.0);
                let direction = Vec3::new(
                        rng.gen_range(-1.0..1.0),
                        rng.gen_range(-1.0..1.0),
                        rng.gen_range(-1.0..1.0),
                    ).normalize();

                let pos = direction * radius;
                Transform::from_translation(pos + translate)
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
        let fov_deg = if self.min_fov_deg == self.max_fov_deg {
            self.min_fov_deg
        } else {
            let rng = &mut rand::thread_rng();
            rng.gen_range(self.min_fov_deg..self.max_fov_deg)
        };

        let fov = fov_deg * std::f32::consts::PI / 180.0;
        PerspectiveProjection {
            fov,
            near: 0.1,
            far: 25.0,
            ..default()
        }
    }

    pub fn exact(fov_deg: f32) -> Self {
        Self {
            min_fov_deg: fov_deg,
            max_fov_deg: fov_deg,
        }
    }
}


#[cfg(feature = "python")]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    Hash,
    PartialEq,
    Reflect,
    Deserialize,
    Parser,
    Serialize,
    ValueEnum,
)]
#[pyclass(eq, eq_int)]
pub enum PlaybackMode {
    Loop,
    Once,
    PingPong,
    Sin,
    #[default]
    Still,
}

#[cfg(not(feature = "python"))]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    Hash,
    PartialEq,
    Reflect,
    Deserialize,
    Parser,
    Serialize,
    ValueEnum,
)]
pub enum PlaybackMode {
    Loop,
    Once,
    PingPong,
    Sin,
    #[default]
    Still,
}

#[cfg(feature = "python")]
#[derive(
    Resource,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Reflect,
    Deserialize,
    Parser,
    Serialize,
)]
#[pyclass]
#[reflect(Resource)]
pub struct Playback {
    pub mode: PlaybackMode,
    pub progress: f32,
    pub direction: f32,
    pub speed: f32,
}

#[cfg(not(feature = "python"))]
#[derive(
    Resource,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Reflect,
    Deserialize,
    Parser,
    Serialize,
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



#[derive(Clone, Debug, Reflect)]
pub enum TrajectorySampler {
    Static {
        start: ExtrinsicsSampler,
    },
    Linear {
        start: ExtrinsicsSampler,
        end: ExtrinsicsSampler,
    },
    Avoidant {
        start: ExtrinsicsSampler,
        end: ExtrinsicsSampler,
        bend_away_from: Vec3,
        radius: f32,
    },
    AvoidantXZ {
        start: ExtrinsicsSampler,
        end: ExtrinsicsSampler,
        bend_away_from: Vec3,
        radius: f32,
    },
    // TODO: formal curve support /w composition
    // TODO: support varying intrinsics across trajectory
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
                let start    = start.sample_cache();
                let end      = end.sample_cache();

                let pos = start.translation.lerp(end.translation, progress);

                let mut rot = start.rotation.slerp(end.rotation, progress);
                let (yaw, pitch, _roll) = rot.to_euler(EulerRot::YXZ);      // extract yaw-pitch-roll
                rot = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0)      // roll = 0 → y-up
                    .normalize();

                Transform {
                    translation: pos,
                    rotation: rot,
                    ..Default::default()
                }
            },
            TrajectorySampler::Avoidant { start, end, bend_away_from, radius } => {
                let progress = progress.clamp(0.0, 1.0);
                let start_transform = start.sample_cache();
                let end_transform = end.sample_cache();

                let mid_point = start_transform.translation.lerp(end_transform.translation, 0.5);

                let path_dir = end_transform.translation - start_transform.translation;
                let path_length = path_dir.length();

                let avoidance_dir = (mid_point - *bend_away_from).normalize();
                let perpendicular_dir = avoidance_dir - path_dir.normalize() * avoidance_dir.dot(path_dir.normalize());
                let perpendicular_dir = perpendicular_dir.normalize();

                let effective_radius = radius.min(path_length * 0.5);
                let control_point = mid_point + perpendicular_dir * effective_radius;

                let t = progress;
                let pos = (1.0 - t).powi(2) * start_transform.translation
                    + 2.0 * (1.0 - t) * t * control_point
                    + t.powi(2) * end_transform.translation;

                let rot = start_transform.rotation.slerp(end_transform.rotation, progress);

                Transform::from_translation(pos).mul_transform(Transform::from_rotation(rot))
            },
            TrajectorySampler::AvoidantXZ { start, end, bend_away_from, radius } => {
                let progress = progress.clamp(0.0, 1.0);
                let start_transform = start.sample_cache();
                let end_transform = end.sample_cache();

                let mid_point = start_transform.translation.lerp(end_transform.translation, 0.5);

                let path_dir = (end_transform.translation - start_transform.translation).with_y(0.0);
                let path_length = path_dir.length();

                let avoidance_vec = (mid_point - *bend_away_from).with_y(0.0);

                let perpendicular_dir = if avoidance_vec.length_squared() < 1e-6 {
                    Vec3::new(-path_dir.z, 0.0, path_dir.x).normalize()
                } else {
                    let avoidance_dir = avoidance_vec.normalize();
                    let projected = avoidance_dir - path_dir.normalize() * avoidance_dir.dot(path_dir.normalize());
                    projected.normalize()
                };

                let effective_radius = radius.min(path_length * 0.5);
                let control_point = mid_point + perpendicular_dir * effective_radius;

                let t = progress;
                let pos = (1.0 - t).powi(2) * start_transform.translation
                    + 2.0 * (1.0 - t) * t * control_point
                    + t.powi(2) * end_transform.translation;

                let rot = start_transform.rotation.slerp(end_transform.rotation, progress);

                Transform::from_translation(pos).mul_transform(Transform::from_rotation(rot))
            },
        }
    }

    pub fn draw_gizmos(
        &mut self,
        gizmos: &mut Gizmos<EditorCameraGizmoConfigGroup>,
        transform: Transform,
    ) {
        match self {
            TrajectorySampler::Linear { start, end } => {
                let start = transform.transform_point(start.sample_cache().translation);
                let end = transform.transform_point(end.sample_cache().translation);
                gizmos.line(start, end, Color::WHITE);
            },
            TrajectorySampler::Avoidant { start, end, bend_away_from, radius } => {
                let start_transform = start.sample_cache();
                let end_transform = end.sample_cache();
                let mid_point = start_transform.translation.lerp(end_transform.translation, 0.5);

                let path_dir = end_transform.translation - start_transform.translation;
                let path_length = path_dir.length();

                let avoidance_vec = mid_point - *bend_away_from;

                let perpendicular_dir = if avoidance_vec.length_squared() < 1e-6 {
                    path_dir.any_orthonormal_vector().normalize()
                } else {
                    let avoidance_dir = avoidance_vec.normalize();
                    let projected = avoidance_dir - path_dir.normalize() * avoidance_dir.dot(path_dir.normalize());
                    projected.normalize()
                };

                let effective_radius = radius.min(path_length * 0.5);
                let control_point = mid_point + perpendicular_dir * effective_radius;

                let curve_length = start_transform.translation.distance(control_point)
                    + control_point.distance(end_transform.translation);
                let segments = (curve_length * 10.0).ceil().max(10.0) as usize;

                let step = 1.0 / segments as f32;
                let mut prev_point = transform.transform_point(start_transform.translation);
                for i in 1..=segments {
                    let t = step * i as f32;
                    let point = (1.0 - t).powi(2) * start_transform.translation
                        + 2.0 * (1.0 - t) * t * control_point
                        + t.powi(2) * end_transform.translation;
                    let transformed_point = transform.transform_point(point);
                    gizmos.line(prev_point, transformed_point, Color::WHITE);
                    prev_point = transformed_point;
                }
            },
            TrajectorySampler::AvoidantXZ { start, end, bend_away_from, radius } => {
                let start_transform = start.sample_cache();
                let end_transform = end.sample_cache();
                let mid_point = start_transform.translation.lerp(end_transform.translation, 0.5);

                let path_dir = (end_transform.translation - start_transform.translation).with_y(0.0);
                let path_length = path_dir.length();

                let avoidance_vec = (mid_point - *bend_away_from).with_y(0.0);

                let perpendicular_dir = if avoidance_vec.length_squared() < 1e-6 {
                    Vec3::new(-path_dir.z, 0.0, path_dir.x).normalize()
                } else {
                    let avoidance_dir = avoidance_vec.normalize();
                    let projected = avoidance_dir - path_dir.normalize() * avoidance_dir.dot(path_dir.normalize());
                    projected.normalize()
                };

                let effective_radius = radius.min(path_length * 0.5);
                let control_point = (mid_point + perpendicular_dir * effective_radius).with_y(mid_point.y);

                let curve_length = start_transform.translation.distance(control_point)
                    + control_point.distance(end_transform.translation);
                let segments = (curve_length * 10.0).ceil().max(10.0) as usize;

                let step = 1.0 / segments as f32;
                let mut prev_point = transform.transform_point(start_transform.translation);
                for i in 1..=segments {
                    let t = step * i as f32;
                    let point = (1.0 - t).powi(2) * start_transform.translation
                        + 2.0 * (1.0 - t) * t * control_point
                        + t.powi(2) * end_transform.translation;
                    let transformed_point = transform.transform_point(point);
                    gizmos.line(prev_point, transformed_point, Color::WHITE);
                    prev_point = transformed_point;
                }
            },
            _ => {},
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
    mut regenerate_events: EventReader<RegenerateSceneEvent>,
) {
    let mut update_camera_playbacks = global_playback.is_changed() || global_playback.mode != PlaybackMode::Still;
    global_playback.step(&time);

    if !regenerate_events.is_empty() {
        regenerate_events.clear();
        global_playback.progress = 0.0;
        update_camera_playbacks = true;
    }

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
        let target = RenderTarget::Image(
            ImageRenderTarget::from(render_target.clone())
        );

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
                MotionVectorPrepass,
                render_mode.dither(),
                render_mode.msaa(),
                render_mode.tonemapping(),
                // PluckerCamera,
                Name::new("zeroverse_camera"),
            ));

        if let Some(bloom) = render_mode.bloom() {
            camera.insert(bloom);
        }

        #[cfg(not(feature = "web"))]
        camera
            .insert((
                DepthPrepass,
                NormalPrepass,
            ));

        if args.image_copiers {
            // TODO: use pipeline color format
            {  // color
                let mut color_cpu_image = Image {
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
                color_cpu_image.resize(size);
                let color_cpu_image_handle = images.add(color_cpu_image);

                camera.insert(io::image_copy::ImageCopier::new(
                    render_target,
                    color_cpu_image_handle,
                    size,
                    TextureFormat::Rgba32Float,
                    &render_device,
                ));
            }

            // let mut copiers = Vec::new();

            // #[cfg(not(feature = "web"))]
            // { // depth
            //     let mut depth_cpu_image = Image {
            //         texture_descriptor: TextureDescriptor {
            //             label: "bevy_zeroverse_camera_depth_cpu_image".into(),
            //             size,
            //             dimension: TextureDimension::D2,
            //             format: CORE_3D_DEPTH_FORMAT,
            //             mip_level_count: 1,
            //             sample_count: 1,
            //             usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            //             view_formats: &[],
            //         },
            //         ..Default::default()
            //     };
            //     depth_cpu_image.resize(size);
            //     let depth_cpu_image_handle = images.add(depth_cpu_image);

            //     copiers.push(io::prepass_copy::PrepassCopier::new(
            //         RenderMode::Depth,
            //         depth_cpu_image_handle,
            //         size,
            //         CORE_3D_DEPTH_FORMAT,
            //         &render_device,
            //     ));
            // }

            // { // motion vector
            //     let mut motion_vectors_cpu_image = Image {
            //         texture_descriptor: TextureDescriptor {
            //             label: "bevy_zeroverse_camera_motion_vectors_cpu_image".into(),
            //             size,
            //             dimension: TextureDimension::D2,
            //             format: MOTION_VECTOR_PREPASS_FORMAT,
            //             mip_level_count: 1,
            //             sample_count: 1,
            //             usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            //             view_formats: &[],
            //         },
            //         ..Default::default()
            //     };
            //     motion_vectors_cpu_image.resize(size);
            //     let motion_vectors_cpu_image_handle = images.add(motion_vectors_cpu_image);

            //     copiers.push(io::prepass_copy::PrepassCopier::new(
            //         RenderMode::MotionVectors,
            //         motion_vectors_cpu_image_handle,
            //         size,
            //         MOTION_VECTOR_PREPASS_FORMAT,
            //         &render_device,
            //     ));
            // }

            // #[cfg(not(feature = "web"))]
            // { // normal
            //     let mut normal_cpu_image = Image {
            //         texture_descriptor: TextureDescriptor {
            //             label: "bevy_zeroverse_camera_normal_cpu_image".into(),
            //             size,
            //             dimension: TextureDimension::D2,
            //             format: NORMAL_PREPASS_FORMAT,
            //             mip_level_count: 1,
            //             sample_count: 1,
            //             usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            //             view_formats: &[],
            //         },
            //         ..Default::default()
            //     };
            //     normal_cpu_image.resize(size);
            //     let normal_cpu_image_handle = images.add(normal_cpu_image);

            //     copiers.push(io::prepass_copy::PrepassCopier::new(
            //         RenderMode::Normal,
            //         normal_cpu_image_handle,
            //         size,
            //         NORMAL_PREPASS_FORMAT,
            //         &render_device,
            //     ));
            // }

            // camera.insert(io::prepass_copy::PrepassCopiers(copiers));
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
        let mut entity = commands.entity(entity);
        entity
            .insert((
                Camera3d {
                    screen_space_specular_transmission_quality: ScreenSpaceTransmissionQuality::High,
                    ..default()
                },
                Camera {
                    hdr: true,
                    ..default()
                },
                Projection::Perspective(PerspectiveProjection {
                    far: 25.0,
                    ..default()
                }),
                Exposure::INDOOR,
                marker.transform.unwrap_or_default(),
                render_mode.dither(),
                render_mode.msaa(),
                render_mode.tonemapping(),
                MotionVectorPrepass,
                // PluckerCamera,
            ))
            .insert(render_layer)
            .insert(ProcessedEditorCameraMarker)
            .insert(Name::new("editor_camera"));

        if let Some(bloom) = render_mode.bloom() {
            entity.insert(bloom);
        }

        #[cfg(not(feature = "web"))]
        entity.insert((
                DepthPrepass,
                NormalPrepass,
            ));
    }
}


pub fn update_render_pipeline(
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
        let mut entity = commands.entity(camera_entity);
        entity
            .insert(render_mode.dither())
            .insert(render_mode.msaa())
            .insert(render_mode.tonemapping());

        if let Some(bloom) = render_mode.bloom() {
            entity.insert(bloom);
        } else {
            entity.remove::<Bloom>();
        }
    }

    for camera_entity in zeroverse_cameras.iter() {
        let mut entity = commands.entity(camera_entity);
        entity
            .insert(render_mode.dither())
            .insert(render_mode.msaa())
            .insert(render_mode.tonemapping());

        if let Some(bloom) = render_mode.bloom() {
            entity.insert(bloom);
        } else {
            entity.remove::<Bloom>();
        }
    }
}


#[allow(clippy::type_complexity)]
pub fn draw_camera_gizmo(
    args: Res<BevyZeroverseConfig>,
    mut gizmos: Gizmos<EditorCameraGizmoConfigGroup>,
    mut cameras: Query<
        (
            &GlobalTransform,
            &Projection,
            &mut ZeroverseCamera,
        ),
        (
            With<Camera>,
            Without<EditorCameraMarker>,
        ),
    >,
    scene: Query<&GlobalTransform, With<ZeroverseSceneRoot>>,
) {
    if !args.gizmos {
        return;
    }

    let color = Color::srgb(1.0, 0.0, 1.0);

    let scene_transform = match scene.single() {
        Ok(scene_transform) => scene_transform.compute_transform(),
        Err(_) => Transform::default(),
    };

    for (
        global_transform,
        projection,
        mut zeroverse_camera,
    ) in cameras.iter_mut() {
        let transform = global_transform.compute_transform();

        let (aspect_ratio, fov_y) = match projection {
            Projection::Perspective(persp) => (persp.aspect_ratio, persp.fov),
            Projection::Orthographic(_) => {
                (1.0, 0.0)
            }
            Projection::Custom(_) => todo!(),
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
        gizmos.sphere(transform.translation, 0.1, color);

        let rect_transform = Transform::from_translation(transform.translation + forward);
        let rect_transform = rect_transform.mul_transform(Transform::from_rotation(transform.rotation));
        gizmos.rect(
            rect_transform.to_isometry(),
            Vec2::new(tan_half_fov_x * 2.0 * scale, tan_half_fov_y * 2.0 * scale),
            color,
        );

        zeroverse_camera.trajectory.draw_gizmos(&mut gizmos, scene_transform);
    }
}
