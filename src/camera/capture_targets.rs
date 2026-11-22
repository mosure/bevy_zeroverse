//! Reuse compatible attachments by capture index across sequential rooms.
//! Readbacks require a retired copier. An automated viewer can retain its last
//! displayed image while a replacement camera acquires the same texture.
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
    pub ssaa: Option<crate::render::ssaa::SsaaCamera>,
    pub copier: Option<io::image_copy::ImageCopier>,
    #[cfg(not(target_arch = "wasm32"))]
    pub geometry: Option<crate::render::ground_truth::GroundTruthCamera>,
}
impl Targets {
    fn color_image(size: Extent3d, label: &'static str) -> Image {
        Image {
            // Render attachments are cleared by their camera. No CPU zero-fill
            // or redundant initial texture upload is needed.
            data: None,
            texture_descriptor: TextureDescriptor {
                label: Some(label),
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
        }
    }

    fn new(key: Key, images: &mut Assets<Image>, device: &RenderDevice) -> Self {
        let size = Extent3d {
            width: key.resolution.x,
            height: key.resolution.y,
            depth_or_array_layers: 1,
        };
        let color = images.add(Self::color_image(size, "bevy_zeroverse_camera_target"));
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
            ssaa: None,
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

    /// The copier and numeric planes are sized in output pixels. A change in
    /// RGB sampling density only needs a new high-resolution color attachment.
    fn set_raster(&mut self, factor: Option<u32>, resolution: UVec2, images: &mut Assets<Image>) {
        if self.ssaa.as_ref().map(|ssaa| ssaa.factor) == factor {
            return;
        }
        let output = self
            .ssaa
            .take()
            .map(|ssaa| ssaa.output)
            .unwrap_or_else(|| self.color.clone());
        if let Some(factor) = factor {
            let size = Extent3d {
                width: resolution.x * factor,
                height: resolution.y * factor,
                depth_or_array_layers: 1,
            };
            self.color = images.add(Self::color_image(
                size,
                "bevy_zeroverse_supersampled_target",
            ));
            self.ssaa = Some(crate::render::ssaa::SsaaCamera::new(
                self.color.clone(),
                output,
                resolution,
                factor,
            ));
        } else {
            self.color = output;
        }
    }

    fn recycled(&self) -> Option<Self> {
        Some(Self {
            color: self.color.clone(),
            ssaa: self.ssaa.as_ref().map(|ssaa| ssaa.recycled()),
            copier: match &self.copier {
                Some(copier) => Some(copier.recycled()?),
                None => None,
            },
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
    fn reusable(args: &BevyZeroverseConfig, index: usize) -> bool {
        index < args.num_cameras
            && ((cfg!(not(target_arch = "wasm32")) && args.headless && args.image_copiers)
                || (!args.headless
                    && !args.image_copiers
                    && args.regenerate_ms > 0
                    && args.scene_type == crate::scene::ZeroverseSceneType::ProceduralIndoor))
    }
    pub fn retain(&mut self, args: &BevyZeroverseConfig) {
        self.0.retain(|index, _| Self::reusable(args, *index));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn acquire(
        &mut self,
        owner: Entity,
        index: Option<&CaptureCameraIndex>,
        sensor: bool,
        resolution: UVec2,
        ssaa_factor: u32,
        args: &BevyZeroverseConfig,
        cameras: &Query<(), With<Camera>>,
        images: &mut Assets<Image>,
        device: &RenderDevice,
    ) -> Targets {
        let key = Key::new(args, resolution);
        let slot = index
            .filter(|index| Self::reusable(args, index.0))
            .map(|index| index.0);
        // Never share attachments with another live camera, or a canceled
        // capture whose asynchronous map callback has not completed.
        let mut targets = slot
            .and_then(|index| self.0.get(&index))
            .filter(|entry| entry.key == key && !cameras.contains(entry.owner))
            .and_then(|entry| entry.targets.recycled())
            .unwrap_or_else(|| Targets::new(key, images, device));
        let raster = (ssaa_factor > 1 || sensor).then_some(ssaa_factor);
        targets.set_raster(raster, resolution, images);
        if let Some(index) = slot {
            self.0.insert(
                index,
                Entry {
                    key,
                    owner,
                    targets: targets.clone(),
                },
            );
        }
        targets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automated_presentation_reuses_indexed_outputs_without_cpu_copiers() {
        let mut args = BevyZeroverseConfig {
            headless: false,
            image_copiers: false,
            regenerate_ms: 1000,
            num_cameras: 2,
            scene_type: crate::scene::ZeroverseSceneType::ProceduralIndoor,
            ..default()
        };
        assert!(CaptureTargets::reusable(&args, 1));
        assert!(!CaptureTargets::reusable(&args, 2));
        let mut images = Assets::<Image>::default();
        let output = images.add(Image::default());
        let mut targets = Targets {
            color: output.clone(),
            ssaa: None,
            copier: None,
            #[cfg(not(target_arch = "wasm32"))]
            geometry: None,
        };
        targets.set_raster(Some(2), UVec2::splat(64), &mut images);
        let next = targets.recycled().unwrap();
        assert_eq!(next.color, targets.color);
        assert_eq!(next.ssaa.unwrap().output, output);
        assert!(next.copier.is_none());
        args.regenerate_ms = 0;
        assert!(!CaptureTargets::reusable(&args, 0));
    }

    #[test]
    fn changing_rgb_density_preserves_the_resolved_color_and_geometric_planes() {
        let mut images = Assets::<Image>::default();
        let resolution = UVec2::new(160, 96);
        let output = images.add(Image::default());
        let mut targets = Targets {
            color: output.clone(),
            ssaa: None,
            copier: Some(io::image_copy::ImageCopier::test_idle()),
            #[cfg(not(target_arch = "wasm32"))]
            geometry: Some(crate::render::ground_truth::GroundTruthCamera::new(
                &mut images,
                resolution,
            )),
        };
        #[cfg(not(target_arch = "wasm32"))]
        let planes = {
            let gt = targets.geometry.as_ref().unwrap();
            (gt.world_depth.id(), gt.normal_semantic.id())
        };
        for factor in [Some(2), Some(4), Some(3), Some(1), None, Some(2)] {
            targets = targets.recycled().expect("completed camera is reusable");
            targets.set_raster(factor, resolution, &mut images);
            if let Some(scale) = factor {
                let ssaa = targets.ssaa.as_ref().unwrap();
                assert_eq!(ssaa.output, output);
                assert_eq!(ssaa.output_size, resolution);
                assert_eq!(ssaa.source, targets.color);
                assert_eq!(ssaa.factor, scale);
                assert_eq!(
                    images.get(&targets.color).unwrap().size(),
                    resolution * scale
                );
            } else {
                assert!(targets.ssaa.is_none());
                assert_eq!(targets.color, output);
            }
            let existing = targets.color.id();
            targets.set_raster(factor, resolution, &mut images);
            assert_eq!(existing, targets.color.id());
            #[cfg(not(target_arch = "wasm32"))]
            {
                let gt = targets.geometry.as_ref().unwrap();
                assert_eq!(planes, (gt.world_depth.id(), gt.normal_semantic.id()));
            }
        }
    }
}
