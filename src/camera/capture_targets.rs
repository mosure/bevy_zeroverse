//! Bounded headless capture storage. Cameras still get new entities, poses and
//! frame epochs per room; only completed GPU attachments and staging buffers
//! survive regeneration. Interactive cameras never enter this pool.
use super::*;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Key {
    resolution: UVec2,
    copy: bool,
    #[cfg(not(target_arch = "wasm32"))]
    geometry: bool,
    #[cfg(not(target_arch = "wasm32"))]
    flow: bool,
    #[cfg(not(target_arch = "wasm32"))]
    visibility: bool,
}
impl Key {
    fn new(args: &BevyZeroverseConfig, resolution: UVec2) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let modes = || {
            args.render_modes
                .iter()
                .chain(std::iter::once(&args.render_mode))
        };
        #[cfg(not(target_arch = "wasm32"))]
        let flow = modes().any(RenderMode::is_flow);
        #[cfg(not(target_arch = "wasm32"))]
        let visibility = modes().any(|mode| *mode == RenderMode::CoVisibility);
        Self {
            resolution,
            copy: args.image_copiers,
            #[cfg(not(target_arch = "wasm32"))]
            geometry: cfg!(not(target_arch = "wasm32"))
                && args.image_copiers
                && (flow
                    || visibility
                    || (args.scene_type == crate::scene::ZeroverseSceneType::ProceduralIndoor
                        && modes().any(|mode| *mode != RenderMode::Color))),
            #[cfg(not(target_arch = "wasm32"))]
            flow,
            #[cfg(not(target_arch = "wasm32"))]
            visibility,
        }
    }
}

#[derive(Clone)]
pub(super) struct Targets {
    pub color: Handle<Image>,
    pub copier: Option<io::image_copy::ImageCopier>,
    #[cfg(not(target_arch = "wasm32"))]
    pub geometry: Option<crate::render::ground_truth::GroundTruthCamera>,
}
impl Targets {
    fn new(key: Key, images: &mut Assets<Image>, device: &RenderDevice) -> Self {
        let size = Extent3d {
            width: key.resolution.x,
            height: key.resolution.y,
            depth_or_array_layers: 1,
        };
        let color = images.add(Image {
            // Render attachments are cleared by their camera. No CPU zero-fill
            // or redundant initial texture upload is needed.
            data: None,
            texture_descriptor: TextureDescriptor {
                label: Some("bevy_zeroverse_camera_target"),
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
        });
        #[allow(unused_mut)]
        let mut planes = vec![color.clone()];
        #[cfg(not(target_arch = "wasm32"))]
        let geometry = key.geometry.then(|| {
            let mut gt =
                crate::render::ground_truth::GroundTruthCamera::new(images, key.resolution);
            planes.extend([gt.world_depth.clone(), gt.normal_semantic.clone()]);
            if key.flow {
                planes.push(gt.enable_flow(images));
            }
            if key.visibility {
                planes.push(gt.enable_co_visibility(images, key.resolution));
            }
            gt
        });
        Self {
            color,
            copier: key.copy.then(|| {
                io::image_copy::ImageCopier::for_targets(
                    planes,
                    size,
                    TextureFormat::Rgba32Float,
                    device,
                )
            }),
            #[cfg(not(target_arch = "wasm32"))]
            geometry,
        }
    }

    fn recycled(&self) -> Option<Self> {
        Some(Self {
            color: self.color.clone(),
            copier: Some(self.copier.as_ref()?.recycled()?),
            #[cfg(not(target_arch = "wasm32"))]
            geometry: self.geometry.as_ref().map(|gt| gt.recycled()),
        })
    }
}

struct Entry {
    key: Key,
    owner: Entity,
    targets: Targets,
}

#[derive(Resource, Default)]
pub(super) struct CaptureTargets(HashMap<usize, Entry>);
impl CaptureTargets {
    pub fn retain(&mut self, args: &BevyZeroverseConfig) {
        self.0.retain(|index, _| {
            cfg!(not(target_arch = "wasm32"))
                && args.headless
                && args.image_copiers
                && *index < args.num_cameras
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn acquire(
        &mut self,
        owner: Entity,
        index: Option<&CaptureCameraIndex>,
        resolution: UVec2,
        args: &BevyZeroverseConfig,
        cameras: &Query<(), With<Camera>>,
        images: &mut Assets<Image>,
        device: &RenderDevice,
    ) -> Targets {
        let key = Key::new(args, resolution);
        let slot = index
            .filter(|index| {
                cfg!(not(target_arch = "wasm32"))
                    && args.headless
                    && args.image_copiers
                    && index.0 < args.num_cameras
            })
            .map(|index| index.0);
        if let Some(index) = slot {
            // Never share attachments with another live camera, or a canceled
            // capture whose asynchronous map callback has not completed.
            let targets = self
                .0
                .get(&index)
                .filter(|entry| entry.key == key && !cameras.contains(entry.owner))
                .and_then(|entry| entry.targets.recycled());
            let targets = targets.unwrap_or_else(|| Targets::new(key, images, device));
            self.0.insert(
                index,
                Entry {
                    key,
                    owner,
                    targets: targets.clone(),
                },
            );
            targets
        } else {
            Targets::new(key, images, device)
        }
    }
}
