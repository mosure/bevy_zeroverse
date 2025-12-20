use bevy::prelude::*;

use crate::{
    annotation::obb::ObjectObb,
    app::BevyZeroverseConfig,
    camera::Playback,
    io::{channels, image_copy::ImageCopier},
    ovoxel::{OvoxelExport, OvoxelVolume},
    render::RenderMode,
    scene::{RegenerateSceneEvent, SceneAabb, SceneAabbNode},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct View {
    pub color: Vec<u8>,
    pub depth: Vec<u8>,
    pub normal: Vec<u8>,
    pub optical_flow: Vec<u8>,
    pub position: Vec<u8>,
    pub world_from_view: [[f32; 4]; 4],
    pub fovy: f32,
    pub near: f32,
    pub far: f32,
    pub time: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ObjectObbSample {
    pub center: [f32; 3],
    pub scale: [f32; 3],
    pub rotation: [f32; 4],
    pub class_name: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct OvoxelSample {
    pub coords: Vec<[u32; 3]>,
    pub dual_vertices: Vec<[u8; 3]>,
    pub intersected: Vec<u8>,
    pub base_color: Vec<[u8; 4]>,
    pub semantics: Vec<u16>,
    pub semantic_labels: Vec<String>,
    pub resolution: u32,
    pub aabb: [[f32; 3]; 2],
}

fn coords_sorted(coords: &[[u32; 3]]) -> bool {
    coords.windows(2).all(|w| w[0] <= w[1])
}

#[derive(Clone, Debug, Default, Resource, Serialize, Deserialize, PartialEq)]
pub struct Sample {
    pub views: Vec<View>,

    pub view_dim: u32,

    /// min and max corners of the axis-aligned bounding box
    pub aabb: [[f32; 3]; 2],

    pub object_obbs: Vec<ObjectObbSample>,

    /// Optional O-Voxel representation of the scene.
    pub ovoxel: Option<OvoxelSample>,
}

#[derive(Debug, Resource, Reflect)]
#[reflect(Resource)]
pub struct SamplerState {
    pub enabled: bool,
    pub regenerate_scene: bool,
    pub frames: u32,
    pub render_modes: Vec<RenderMode>,
    pub step: u32,
    pub timesteps: Vec<f32>,
    pub warmup_frames: u32,
    pub ovoxel_wait_frames: u32,
}

#[derive(Resource, Debug)]
pub struct StartupDelay {
    pub frames: u32,
    pub done: bool,
}

impl Default for StartupDelay {
    fn default() -> Self {
        Self {
            frames: 64,
            done: false,
        }
    }
}

impl Default for SamplerState {
    fn default() -> Self {
        SamplerState {
            enabled: true,
            regenerate_scene: true,
            frames: SamplerState::FRAME_DELAY,
            render_modes: vec![
                RenderMode::Color,
                // RenderMode::Depth,
                // RenderMode::Normal,
                // RenderMode::OpticalFlow,
                // RenderMode::Position,
            ],
            step: 0,
            timesteps: vec![0.125],
            warmup_frames: SamplerState::WARMUP_FRAME_DELAY,
            ovoxel_wait_frames: 0,
        }
    }
}

impl SamplerState {
    const FRAME_DELAY: u32 = 1;
    const WARMUP_FRAME_DELAY: u32 = 3;
    const MAX_OVOXEL_WAIT_FRAMES: u32 = 240;

    pub fn inference_only() -> Self {
        SamplerState {
            enabled: true,
            regenerate_scene: false,
            frames: SamplerState::FRAME_DELAY,
            render_modes: vec![RenderMode::Color],
            step: 0,
            timesteps: vec![0.125],
            warmup_frames: SamplerState::WARMUP_FRAME_DELAY,
            ovoxel_wait_frames: 0,
        }
    }

    pub fn reset(&mut self) {
        self.frames = SamplerState::FRAME_DELAY;
        self.warmup_frames = SamplerState::WARMUP_FRAME_DELAY;
    }
}

pub fn configure_sampler(app: &mut App, initial_state: SamplerState) {
    app.init_resource::<Sample>();
    app.insert_resource(initial_state);
    app.register_type::<SamplerState>();
    app.init_resource::<StartupDelay>();

    app.add_systems(
        PostUpdate,
        sample_stream.after(crate::io::image_copy::receive_images),
    );
}

#[allow(clippy::too_many_arguments)]
pub fn sample_stream(
    args: Res<BevyZeroverseConfig>,
    mut buffered_sample: ResMut<Sample>,
    mut state: ResMut<SamplerState>,
    mut startup_delay: ResMut<StartupDelay>,
    cameras: Query<(&GlobalTransform, &Projection, &ImageCopier)>,
    scene: Query<(&SceneAabbNode, &SceneAabb)>,
    object_obbs: Query<&ObjectObb>,
    images: Res<Assets<Image>>,
    mut render_mode: ResMut<RenderMode>,
    mut playback: ResMut<Playback>,
    mut regenerate_event: MessageWriter<RegenerateSceneEvent>,
    ovoxels: Query<&OvoxelVolume, With<OvoxelExport>>,
) {
    if !state.enabled {
        return;
    }

    if cameras.iter().count() == 0 {
        return;
    }

    if !startup_delay.done {
        if startup_delay.frames > 0 {
            startup_delay.frames -= 1;
            return;
        }
        startup_delay.done = true;
    }

    if state.warmup_frames > 0 {
        state.warmup_frames -= 1;
        return;
    }

    if state.frames > 0 {
        state.frames -= 1;
        return;
    }

    if state.render_modes.is_empty() {
        state.render_modes = args.render_modes.clone();
        if state.render_modes.is_empty() {
            state.render_modes.push(args.render_mode.clone());
        }
    }

    let camera_count = cameras.iter().count();
    let view_count = camera_count * args.playback_steps as usize;
    if buffered_sample.views.len() != view_count {
        buffered_sample.views.clear();

        for _ in 0..view_count {
            buffered_sample.views.push(View::default());
        }
    }

    let scene_aabb = scene.single().unwrap().1;
    buffered_sample.aabb = [scene_aabb.min.into(), scene_aabb.max.into()];
    buffered_sample.object_obbs.clear();
    for obb in object_obbs.iter() {
        buffered_sample.object_obbs.push(ObjectObbSample {
            center: obb.center.into(),
            scale: obb.scale.into(),
            rotation: [
                obb.rotation.x,
                obb.rotation.y,
                obb.rotation.z,
                obb.rotation.w,
            ],
            class_name: obb.class_name.clone(),
        });
    }

    let write_to = state.render_modes.remove(0);

    let mut missing_signal = false;

    for (i, (camera_transform, projection, image_copier)) in cameras.iter().enumerate() {
        let view_idx = i + camera_count * state.step as usize;
        let view = &mut buffered_sample.views[view_idx];

        view.time = playback.progress;

        match projection {
            Projection::Perspective(perspective) => {
                view.fovy = perspective.fov;
                view.near = perspective.near;
                view.far = perspective.far;
            }
            Projection::Orthographic(_) => panic!("orthographic projection not supported"),
            Projection::Custom(_) => panic!("custom projection not supported"),
        };

        let world_from_view = camera_transform.to_matrix().to_cols_array_2d();
        view.world_from_view = world_from_view;

        let image_data = image_copier
            .cpu_data
            .lock()
            .ok()
            .and_then(|buf| {
                if buf.is_empty() {
                    None
                } else {
                    Some(buf.clone())
                }
            })
            .or_else(|| {
                images
                    .get(&image_copier.dst_image)
                    .and_then(|img| img.clone().data)
            })
            .unwrap_or_default();

        if image_data.is_empty() || image_data.iter().all(|b| *b == 0) {
            missing_signal = true;
            break;
        }

        match write_to {
            RenderMode::Color => view.color = image_data,
            RenderMode::Depth => view.depth = image_data,
            RenderMode::MotionVectors => panic!("motion vector rendering not supported"),
            RenderMode::Normal => view.normal = image_data,
            RenderMode::OpticalFlow => view.optical_flow = image_data,
            RenderMode::Position => view.position = image_data,
            RenderMode::Semantic => panic!("semantic rendering not supported"),
        }
    }

    if missing_signal {
        state.render_modes.insert(0, write_to);
        state.reset();
        return;
    }

    if !state.render_modes.is_empty() {
        *render_mode = state.render_modes[0].clone();
        state.reset();
        return;
    }

    if !state.timesteps.is_empty() {
        playback.progress = state.timesteps.remove(0);
        state.step += 1;
        state.render_modes = args.render_modes.clone();
        if let Some(first) = state.render_modes.first() {
            *render_mode = first.clone();
        }
        state.reset();
        // TODO: set fixed previous cameras for optical flow across timesteps
        return;
    }

    playback.progress = 0.0;

    let views = std::mem::take(&mut buffered_sample.views);

    let include_ovoxel = matches!(
        args.ovoxel_mode,
        crate::app::OvoxelMode::CpuAsync | crate::app::OvoxelMode::GpuCompute
    );
    if !include_ovoxel {
        state.ovoxel_wait_frames = 0;
    }
    let ovoxel = if !include_ovoxel {
        None
    } else {
        match ovoxels.iter().next() {
            Some(v) => {
                let mut coords = v.coords.clone();
                let mut dual_vertices = v.dual_vertices.clone();
                let mut intersected = v.intersected.clone();
                let mut base_color = v.base_color.clone();
                let mut semantics = v.semantics.clone();

                debug_assert_eq!(coords.len(), dual_vertices.len());
                debug_assert_eq!(coords.len(), intersected.len());
                debug_assert_eq!(coords.len(), base_color.len());
                debug_assert_eq!(coords.len(), semantics.len());

                if !coords_sorted(&coords) {
                    #[allow(clippy::type_complexity)]
                    let mut zipped: Vec<([u32; 3], [u8; 3], u8, [u8; 4], u16)> = coords
                        .into_iter()
                        .zip(dual_vertices.into_iter())
                        .zip(intersected.into_iter())
                        .zip(base_color.into_iter())
                        .zip(semantics.into_iter())
                        .map(|((((c, d), i), bc), s)| (c, d, i, bc, s))
                        .collect();
                    zipped.sort_unstable_by(|a, b| a.0.cmp(&b.0));

                    let len = zipped.len();
                    coords = Vec::with_capacity(len);
                    dual_vertices = Vec::with_capacity(len);
                    intersected = Vec::with_capacity(len);
                    base_color = Vec::with_capacity(len);
                    semantics = Vec::with_capacity(len);

                    for (c, d, i, bc, s) in zipped {
                        coords.push(c);
                        dual_vertices.push(d);
                        intersected.push(i);
                        base_color.push(bc);
                        semantics.push(s);
                    }
                }

                let volume = OvoxelSample {
                    coords,
                    dual_vertices,
                    intersected,
                    base_color,
                    semantics,
                    semantic_labels: v.semantic_labels.clone(),
                    resolution: v.resolution,
                    aabb: v.aabb,
                };
                state.ovoxel_wait_frames = 0;
                Some(volume)
            }
            None => {
                state.ovoxel_wait_frames = state.ovoxel_wait_frames.saturating_add(1);
                if state.ovoxel_wait_frames < SamplerState::MAX_OVOXEL_WAIT_FRAMES {
                    // O-Voxel generation still in flight; wait until available before emitting a sample.
                    state.render_modes.insert(0, write_to);
                    state.reset();
                    return;
                }
                warn!(
                    "ovoxel volume unavailable after {} frames; emitting sample without ovoxel",
                    state.ovoxel_wait_frames
                );
                state.ovoxel_wait_frames = 0;
                None
            }
        }
    };

    let sample: Sample = Sample {
        views,
        view_dim: camera_count as u32,
        aabb: buffered_sample.aabb,
        object_obbs: buffered_sample.object_obbs.clone(),
        ovoxel,
    };

    let sender = channels::sample_sender();
    sender.send(sample).unwrap();

    // restore primary render mode for subsequent captures
    *render_mode = args
        .render_modes
        .first()
        .cloned()
        .unwrap_or_else(|| args.render_mode.clone());

    if state.regenerate_scene {
        regenerate_event.write(RegenerateSceneEvent);
    }

    state.ovoxel_wait_frames = 0;
    state.enabled = false;
}
