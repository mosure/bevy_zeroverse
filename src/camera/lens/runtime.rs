use super::*;
use bevy::{
    camera::{Exposure, Hdr, ImageRenderTarget, RenderTarget},
    pbr::{ScreenSpaceTransmission, ScreenSpaceTransmissionQuality},
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_resource::*,
        RenderApp,
    },
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

/// Render implementation detail. Only its parent has a dataset camera index.
#[derive(Component)]
pub struct LensFace {
    pub owner: Entity,
}
#[derive(Component, Clone, ExtractComponent)]
#[extract_app(RenderApp)]
pub struct LensCamera {
    pub calibration: CameraCalibration,
    pub output: Handle<Image>,
    pub faces: [Handle<Image>; 6],
    pub near: f32,
    pub far: f32,
    pub(super) face_scale: f32,
    pub(super) completed: Arc<AtomicU64>,
}
impl LensCamera {
    pub fn ready(&self, request: u64) -> bool {
        self.completed.load(Ordering::Acquire) == request
    }
    #[cfg(test)]
    pub(super) fn test_camera() -> Self {
        Self {
            calibration: CameraCalibration::centered_pinhole(32, 32, 1., 1.).unwrap(),
            output: Handle::default(),
            faces: std::array::from_fn(|_| Handle::default()),
            near: 0.1,
            far: 50.,
            face_scale: 1.,
            completed: Arc::default(),
        }
    }
}

/// Co-located cube orientations with overlapping perspective guard bands.
pub(super) fn orientations() -> [Quat; 6] {
    [
        (Vec3::NEG_Z, Vec3::Y),
        (Vec3::X, Vec3::Y),
        (Vec3::NEG_X, Vec3::Y),
        (Vec3::Y, Vec3::Z),
        (Vec3::NEG_Y, Vec3::NEG_Z),
        (Vec3::Z, Vec3::Y),
    ]
    .map(|(forward, up)| Transform::IDENTITY.looking_to(forward, up).rotation)
}
pub struct FisheyeCameraPlugin;
impl Plugin for FisheyeCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<LensCamera>::default());
        app.add_systems(
            Update,
            (
                setup
                    .after(super::super::CameraBuildSystems::Logical)
                    .in_set(super::super::CameraBuildSystems::Lens),
                preview,
            ),
        );
        app.add_systems(
            PostUpdate,
            sync.after(crate::sample::camera_activity::CaptureCameraGate)
                .after(super::super::gate_interactive_views)
                .before(bevy::camera::CameraUpdateSystems),
        );
        app.add_systems(
            Last,
            sync.after(crate::sample::camera_activity::CaptureCameraGate),
        );
        super::resolve::install(app);
        super::settings::install(app);
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<
        (
            Entity,
            &super::super::ZeroverseCamera,
            &Projection,
            &RenderTarget,
            Option<&crate::render::ssaa::SsaaCamera>,
            Has<bevy::render::view::NoIndirectDrawing>,
        ),
        (
            With<Camera>,
            Without<LensCamera>,
            Without<super::super::InvalidCaptureTarget>,
        ),
    >,
    device: Option<Res<bevy::render::renderer::RenderDevice>>,
) {
    for (entity, camera, projection, target, ssaa, direct) in &cameras {
        let Some(sensor) = camera.fisheye else {
            continue;
        };
        let RenderTarget::Image(target) = target else {
            continue;
        };
        let Some(output) = images.get(&target.handle) else {
            continue;
        };
        let size = ssaa.map_or(output.size(), |s| s.output_size);
        let factor = ssaa.map_or(1, |s| s.factor);
        let calibration =
            match sensor.calibration(size.to_array(), camera.principal_point_offset.to_array()) {
                Ok(c) => c,
                Err(error) => {
                    commands.insert_resource(crate::sample::CaptureFailure(Some(error)));
                    commands
                        .entity(entity)
                        .insert(super::super::InvalidCaptureTarget);
                    continue;
                }
            };
        // Bound the output angular Jacobian over the certified lens domain.
        // Sampling only affects allocation; rendering always uses the polynomial.
        let mut density = 1.0_f32;
        for i in 0..=128 {
            let t = sensor.lens.max_angle * (i as f32 / 128.);
            let x = t * t;
            let k = sensor.lens.coefficients;
            let derivative =
                1. + x * (3. * k[0] + x * (5. * k[1] + x * (7. * k[2] + x * 9. * k[3])));
            let tangential = if t < 1e-6 {
                1.
            } else {
                sensor.lens.radius(t as f64) as f32 / t.sin()
            };
            density = density.max(derivative).max(tangential);
        }
        let requested =
            (2. * calibration.k[0][0].max(calibration.k[1][1]) * density * factor as f32).ceil();
        let limit = device
            .as_ref()
            .map_or(2048, |d| d.limits().max_texture_dimension_2d);
        // Validate before the float-to-integer cast/addition: custom calibrated
        // sensors must fail cleanly rather than saturate and overflow here.
        if !requested.is_finite() || requested < 1. || requested > limit.saturating_sub(4) as f32 {
            let message = format!(
                "fisheye face requires {requested} core pixels plus guard bands; device limit {limit}; reduce resolution, SSAA or lens magnification"
            );
            error!("{message}");
            commands.insert_resource(crate::sample::CaptureFailure(Some(message)));
            commands
                .entity(entity)
                .insert(super::super::InvalidCaptureTarget);
            continue;
        }
        let core = requested as u32;
        let face_count = if sensor.lens.max_angle <= std::f32::consts::FRAC_PI_4 {
            1
        } else {
            5
        };
        // Screen-space effects need context beyond a face edge, not just the
        // two texels required by bilinear lookup. A bounded 3% guard keeps the
        // overlap useful at all supported resolutions without a fixed large
        // overscan cost for tiny captures. Single-face lenses need no seam band.
        let guard = if face_count == 1 {
            2
        } else {
            core.div_ceil(32).clamp(4, 32)
        };
        let side = core + 2 * guard;
        if side > limit {
            commands.insert_resource(crate::sample::CaptureFailure(Some(format!(
                "fisheye face with overlap needs {side} pixels; device limit {limit}"
            ))));
            commands
                .entity(entity)
                .insert(super::super::InvalidCaptureTarget);
            continue;
        }
        let source = output.clone();
        let mut make_face = || {
            let mut image = source.clone();
            image.data = None;
            image.copy_on_resize = false;
            image.texture_descriptor.size.width = side;
            image.texture_descriptor.size.height = side;
            image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
            images.add(image)
        };
        let front = make_face();
        let faces = std::array::from_fn(|i| {
            if i > 0 && i < face_count {
                make_face()
            } else {
                front.clone()
            }
        });
        let (perspective, _) =
            super::super::pinhole::parameters(projection).expect("capture perspective");
        let face_scale = core as f32 / side as f32;
        let projection = Projection::Perspective(PerspectiveProjection {
            fov: 2. * (1. / face_scale).atan(),
            aspect_ratio: 1.,
            near: perspective.near * face_scale / 3_f32.sqrt(),
            far: perspective.far,
            ..default()
        });
        for (i, rotation) in orientations()
            .into_iter()
            .enumerate()
            .take(face_count)
            .skip(1)
        {
            let mut face = commands.spawn((
                (
                    LensFace { owner: entity },
                    ChildOf(entity),
                    Camera3d::default(),
                ),
                Camera::default(),
                RenderTarget::Image(ImageRenderTarget::from(faces[i].clone())),
                Hdr,
                Exposure::INDOOR,
                projection.clone(),
                Transform::from_rotation(rotation),
                bevy::core_pipeline::prepass::MotionVectorPrepass,
                crate::render::RenderMode::Color.msaa(),
                crate::render::RenderMode::Color.tonemapping(),
                crate::render::RenderMode::Color.dither(),
                ScreenSpaceTransmission {
                    quality: ScreenSpaceTransmissionQuality::High,
                    ..default()
                },
                bevy::post_process::bloom::Bloom::default(),
                Name::new("fisheye_internal_face"),
            ));
            // This marker must be selected at spawn. Do not silently introduce
            // GPU culling when the owner uses the validated direct-draw path.
            if direct {
                face.insert(bevy::render::view::NoIndirectDrawing);
            }
        }
        commands.entity(entity).insert((
            LensCamera {
                calibration,
                output: target.handle.clone(),
                faces: faces.clone(),
                near: perspective.near,
                far: perspective.far,
                face_scale,
                completed: Arc::default(),
            },
            projection,
            RenderTarget::Image(ImageRenderTarget::from(faces[0].clone())),
        ));
    }
}

#[allow(clippy::type_complexity)]
fn sync(
    parents: Query<&Camera, (With<LensCamera>, Without<LensFace>)>,
    mut faces: Query<(&LensFace, &mut Camera), Without<LensCamera>>,
    mut geometry: Query<(
        &LensCamera,
        &mut crate::render::ground_truth::GroundTruthCamera,
    )>,
) {
    for (face, mut camera) in &mut faces {
        if let Ok(parent) = parents.get(face.owner) {
            if camera.is_active != parent.is_active {
                camera.is_active = parent.is_active;
            }
            use bevy::camera::ClearColorConfig;
            let same_clear = match (camera.clear_color, parent.clear_color) {
                (ClearColorConfig::Default, ClearColorConfig::Default)
                | (ClearColorConfig::None, ClearColorConfig::None) => true,
                (ClearColorConfig::Custom(a), ClearColorConfig::Custom(b)) => a == b,
                _ => false,
            };
            if !same_clear {
                camera.clear_color = parent.clear_color;
            }
        } else {
            camera.is_active = false;
        }
    }
    for (lens, mut gt) in &mut geometry {
        gt.calibration = Some(lens.calibration.clone());
        gt.near = lens.near;
        gt.far = lens.far;
    }
}

#[derive(Component)]
struct LensPreview;
#[allow(clippy::type_complexity)]
fn preview(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mode: Res<crate::render::RenderMode>,
    cameras: Query<(
        Entity,
        &LensCamera,
        Has<crate::render::ground_truth::GroundTruthCamera>,
        Option<&crate::io::image_copy::ImageCopier>,
    )>,
    mut geometry: Query<&mut crate::render::ground_truth::GroundTruthCamera, With<LensPreview>>,
    mut frame: Local<u64>,
) {
    *frame = frame.wrapping_add(1).max(1);
    for (entity, lens, gt, capture) in &cameras {
        if capture.is_some() && gt && !geometry.contains(entity) {
            continue;
        }
        if *mode == crate::render::RenderMode::Color {
            if geometry.contains(entity) {
                commands
                    .entity(entity)
                    .remove::<(crate::render::ground_truth::GroundTruthCamera, LensPreview)>();
            }
            continue;
        }
        if !gt && *mode != crate::render::RenderMode::CoVisibility {
            let mut gt = crate::render::ground_truth::GroundTruthCamera::new(
                &mut images,
                UVec2::from_array(lens.calibration.image_size),
            );
            gt.calibration = Some(lens.calibration.clone());
            gt.near = lens.near;
            gt.far = lens.far;
            gt.frame_id = capture.map_or(*frame, |c| c.requested_id());
            if mode.is_flow() {
                gt.enable_flow(&mut images);
            }
            commands.entity(entity).insert((gt, LensPreview));
        }
    }
    for (entity, _, _, copier) in &cameras {
        let Ok(mut gt) = geometry.get_mut(entity) else {
            continue;
        };
        gt.frame_id = copier.map_or(*frame, |c| c.requested_id());
        if mode.is_flow() && gt.flow.is_none() {
            gt.enable_flow(&mut images);
        }
        if !mode.is_flow() {
            gt.flow = None;
        }
    }
}
