//! Same-time, first-surface visibility among capture cameras. Editor cameras are
//! never references. Numeric masks and the additive RGB preview share bit order.
use bevy::{asset::RenderAssetUsages, prelude::*, render::render_resource::*};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};

use super::{ground_truth::GroundTruthCamera, RenderMode};
use crate::camera::{CaptureCameraIndex, ZeroverseCamera};
use bevy::camera::RenderTarget;

mod gpu;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use gpu::CoVisibilityLabel;
#[cfg(test)]
mod tests;

pub use bevy_zeroverse_capture::{camera_color, mask_color};
pub const MAX_CAMERAS: usize = bevy_zeroverse_capture::MAX_VISIBILITY_CAMERAS;
pub const ABSOLUTE_TOLERANCE_M: f32 = 0.001;
pub const RELATIVE_TOLERANCE: f32 = 0.0001;

pub fn validate_config(modes: &[RenderMode], camera_count: usize) -> Result<(), String> {
    if modes.contains(&RenderMode::CoVisibility) && !(1..=MAX_CAMERAS).contains(&camera_count) {
        return Err(format!(
            "co-visibility requires 1..={MAX_CAMERAS} capture cameras"
        ));
    }
    Ok(())
}

/// Validate the raw GPU contract before an exporter narrows it to integer masks.
pub fn validate_plane(
    bytes: &[u8],
    pixels: usize,
    count: usize,
    source: usize,
) -> Result<(), String> {
    if !(1..=MAX_CAMERAS).contains(&count) || source >= count || bytes.len() != pixels * 16 {
        return Err("invalid co-visibility plane dimensions or camera count".into());
    }
    for pixel in bytes.as_chunks::<16>().0 {
        let p: [f32; 4] = bytemuck::pod_read_unaligned(pixel);
        if !p.iter().all(|x| x.is_finite())
            || p[0] < 0.0
            || p[0] >= (1u32 << count) as f32
            || p[0].fract() != 0.0
        {
            return Err("invalid co-visibility membership bits".into());
        }
        let mask = p[0] as u16;
        if mask & (1 << source) != 0
            || p[1] != mask.count_ones() as f32
            || ![0.0, 1.0].contains(&p[2])
            || p[3] != 0.0
            || (p[2] == 0.0 && mask != 0)
        {
            return Err("invalid co-visibility source bit, count or validity".into());
        }
    }
    Ok(())
}

/// Decode a lossless RGB code. Lossy codecs, resizing and color grading are not
/// compatible with this representation; training exports use integer masks.
pub fn mask_from_color(rgb: [u8; 3], count: usize) -> Option<u16> {
    if !(1..=MAX_CAMERAS).contains(&count) {
        return None;
    }
    let mut mask = 0;
    for (channel, byte) in rgb.into_iter().enumerate() {
        if channel >= count {
            if byte != 0 {
                return None;
            }
            continue;
        }
        let bits = (count - channel).div_ceil(3);
        let scale = 255 / ((1 << bits) - 1);
        let value = byte as usize;
        if !value.is_multiple_of(scale) || value / scale >= 1 << bits {
            return None;
        }
        for rank in 0..bits {
            if (value / scale) & (1 << (bits - 1 - rank)) != 0 {
                mask |= 1 << (channel + rank * 3);
            }
        }
    }
    Some(mask)
}

pub fn annotation_metadata(camera_indices: &[usize]) -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "camera_count": camera_indices.len(), "max_cameras": MAX_CAMERAS,
        "direction": "source pixel to other capture cameras at the same timestep",
        "surface": "first geometric surface, including opaque glass; no reflected or refracted visibility",
        "membership": "bit i identifies the i-th ordered capture camera; source bit is always zero; editor excluded",
        "rgba_layout": ["membership_mask", "popcount", "source_valid", "reserved_zero"],
        "mask_dtype": "uint16", "background": "mask=0, valid=0; valid unshared surfaces have mask=0, valid=1",
        "visibility_test": "in-frustum and symmetric tangent-plane agreement at the nearest target pixel center",
        "absolute_tolerance_m": ABSOLUTE_TOLERANCE_M,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "rgb_encoding": "sum of camera rgb8 codes in sRGB byte space; lossless PNG only; no tonemapping",
        "legend": camera_indices.iter().enumerate().map(|(bit, index)| serde_json::json!({
            "bit":bit, "camera_index":index, "mask":1u32 << bit,
            "rgb8": camera_color(bit, camera_indices.len())
        })).collect::<Vec<_>>()
    })
}

pub fn validate_metadata(metadata: &serde_json::Value, count: usize) -> Result<(), String> {
    if !(1..=MAX_CAMERAS).contains(&count)
        || metadata["schema_version"] != 1
        || metadata["camera_count"] != count
    {
        return Err("unknown co-visibility convention or camera count".into());
    }
    let legend = metadata["legend"]
        .as_array()
        .ok_or("missing co-visibility legend")?;
    let mut indices = std::collections::HashSet::new();
    if legend.len() != count {
        return Err("invalid co-visibility legend length".into());
    }
    for (bit, item) in legend.iter().enumerate() {
        if item["bit"] != bit
            || item["mask"] != (1u32 << bit)
            || item["rgb8"] != serde_json::json!(camera_color(bit, count))
            || item["camera_index"]
                .as_u64()
                .is_none_or(|i| !indices.insert(i))
        {
            return Err("invalid co-visibility camera ordering or RGB code".into());
        }
    }
    Ok(())
}

#[derive(Default)]
struct Status {
    valid: AtomicBool,
    frame: AtomicU64,
}

#[derive(Clone)]
pub struct CoVisibilityOutput {
    pub image: Handle<Image>,
    pub(crate) preview: Option<Handle<Image>>,
    pub(crate) slot: usize,
    status: Arc<Status>,
}
impl CoVisibilityOutput {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn recycled(&self) -> Self {
        Self {
            status: Arc::default(),
            slot: 0,
            ..self.clone()
        }
    }

    pub(crate) fn new(images: &mut Assets<Image>, size: UVec2) -> Self {
        let mut image = Image::new_target_texture(size.x, size.y, TextureFormat::Rgba32Float, None);
        image.data = None;
        image.copy_on_resize = false;
        image.asset_usage = RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD;
        image.texture_descriptor.usage =
            TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
        Self {
            image: images.add(image),
            preview: None,
            slot: 0,
            status: Arc::default(),
        }
    }
    pub fn rendered_frame(&self) -> Option<u64> {
        self.status
            .valid
            .load(Ordering::Acquire)
            .then(|| self.status.frame.load(Ordering::Acquire))
    }
}

#[derive(Resource, Default)]
pub struct CoVisibilityLegend {
    pub camera_indices: Vec<usize>,
    pub error: Option<String>,
}
#[derive(Default, Clone, Debug, serde::Serialize)]
pub struct CoVisibilityStats {
    pub pipeline_initializations: u64,
    pub atlas_allocations: u64,
    pub atlas_bytes: usize,
    pub dispatches: u64,
    pub views: u64,
    pub pixels: u64,
}
#[derive(Resource, Default, Clone)]
pub struct CoVisibilityDiagnostics(Arc<Mutex<CoVisibilityStats>>);
impl CoVisibilityDiagnostics {
    pub fn snapshot(&self) -> CoVisibilityStats {
        self.0.lock().unwrap().clone()
    }
}
#[derive(Component)]
struct PreviewGeometry;

pub struct CoVisibilityPlugin;
impl Plugin for CoVisibilityPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CoVisibilityLegend>();
        app.init_resource::<CoVisibilityDiagnostics>();
        app.add_systems(Last, update_preview);
        gpu::install(app);
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn update_preview(
    mut commands: Commands,
    mode: Res<RenderMode>,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<
        (
            Entity,
            Option<&CaptureCameraIndex>,
            &RenderTarget,
            &Projection,
            Option<&crate::io::image_copy::ImageCopier>,
            Option<&PreviewGeometry>,
        ),
        With<ZeroverseCamera>,
    >,
    mut geometry: Query<&mut GroundTruthCamera>,
    mut legend: ResMut<CoVisibilityLegend>,
    mut frame: Local<u64>,
) {
    if *mode != RenderMode::CoVisibility && geometry.iter().all(|g| g.co_visibility.is_none()) {
        legend.camera_indices.clear();
        legend.error = None;
        return;
    }
    let mut ordered: Vec<_> = cameras.iter().collect();
    ordered.sort_by_key(|(e, i, ..)| (i.map_or(usize::MAX, |i| i.0), e.to_bits()));
    legend.camera_indices = ordered
        .iter()
        .enumerate()
        .map(|(slot, (_, i, ..))| i.map_or(slot, |i| i.0))
        .collect();
    legend.error = (ordered.len() > MAX_CAMERAS)
        .then(|| format!("Co-visibility supports at most {MAX_CAMERAS} capture cameras"));
    let unique: std::collections::HashSet<_> = legend.camera_indices.iter().collect();
    if unique.len() != legend.camera_indices.len() {
        legend.error = Some("Co-visibility requires unique CaptureCameraIndex values".into());
    }
    *frame = frame.wrapping_add(1).max(1);
    for (slot, (entity, _, target, projection, copier, owned)) in ordered.into_iter().enumerate() {
        // Captures own their request stamps and attachment lifetimes. Never let
        // the interactive preview advance a pending asynchronous readback.
        if copier.is_some() {
            if let Ok(mut gt) = geometry.get_mut(entity) {
                if let Some(error) = &legend.error {
                    gt.fail(error.clone());
                }
                if let Some(output) = gt.co_visibility.as_mut() {
                    output.slot = slot;
                }
            }
            continue;
        }
        if *mode != RenderMode::CoVisibility || legend.error.is_some() {
            if owned.is_some() {
                commands
                    .entity(entity)
                    .remove::<(GroundTruthCamera, PreviewGeometry)>();
            }
            continue;
        }
        let RenderTarget::Image(target) = target else {
            continue;
        };
        let Some(image) = images.get(&target.handle) else {
            continue;
        };
        let size = image.size();
        let configure = |gt: &mut GroundTruthCamera, images: &mut Assets<Image>| {
            if images
                .get(&gt.world_depth)
                .is_none_or(|image| image.size() != size)
            {
                *gt = GroundTruthCamera::new(images, size);
            }
            if gt.co_visibility.is_none() {
                gt.enable_co_visibility(images, size);
            }
            let output = gt.co_visibility.as_mut().unwrap();
            output.slot = slot;
            // CameraDriver renders the grid before this cross-camera pass.
            // A separate image lets the grid display the previous completed
            // annotation instead of the RGB image overwritten on every frame.
            if output.preview.is_none() {
                let mut preview =
                    Image::new_target_texture(size.x, size.y, TextureFormat::Rgba32Float, None);
                preview.data = None;
                preview.copy_on_resize = false;
                output.preview = Some(images.add(preview));
            }
            gt.frame_id = *frame;
            if let Projection::Perspective(p) = projection {
                gt.near = p.near;
                gt.far = p.far;
            }
        };
        if let Ok(mut gt) = geometry.get_mut(entity) {
            configure(&mut gt, &mut images);
        } else {
            let mut gt = GroundTruthCamera::new(&mut images, size);
            configure(&mut gt, &mut images);
            commands.entity(entity).insert((gt, PreviewGeometry));
        }
    }
}
