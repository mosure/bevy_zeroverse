//! Metric top-down diagrams for calibration and prediction inspection.
//!
//! Coordinates are world-space metres, +Y up. The image looks down onto X/Z:
//! +X points right and +Z down. A schematic is an orthographic *diagram* of
//! footprints, not a visibility mask or an architectural section. Ceilings and
//! neighboring rooms are omitted; furniture bounds may overestimate its mesh.
mod export;
pub use export::Document;
mod scene;
mod svg;
#[cfg(test)]
mod tests;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Footprint {
    pub label: String,
    pub instance_id: Option<usize>,
    pub points: Vec<[f32; 3]>,
    pub kind: FootprintKind,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum FootprintKind {
    Floor,
    Level,
    Mezzanine,
    Wall,
    Window,
    Door,
    Furniture,
    Prop,
    Plant,
}

/// Camera convention matches `View::world_from_view`: column-major, -Z forward,
/// +Y up. Lens footprints project the optical center and a finite image plane.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Camera {
    pub label: String,
    pub world_from_view: [[f32; 4]; 4],
    pub fov_y: f32,
    pub aspect: f32,
    /// Exact captured intrinsics when available; takes precedence over fov/aspect.
    #[serde(default)]
    pub calibration: Option<bevy_zeroverse_capture::calibration::CameraCalibration>,
    pub path: Vec<[f32; 3]>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Pose {
    pub label: String,
    pub joints: Vec<[f32; 3]>,
    /// Parent index per joint; -1 marks a root.
    pub parents: Vec<i64>,
}
/// Predictions never change the metric extent or the ground-truth coordinates.
/// Dashed magenta distinguishes every prediction from solid ground truth.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Overlay {
    pub cameras: Vec<Camera>,
    pub poses: Vec<Pose>,
    pub footprints: Vec<Footprint>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Schematic {
    pub schema_version: u32,
    pub seed: u64,
    pub trajectory_progress: f32,
    pub time_seconds: Option<f32>,
    pub footprints: Vec<Footprint>,
    pub cameras: Vec<Camera>,
    pub poses: Vec<Pose>,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct RenderOptions {
    pub width: u32,
    pub height: u32,
    pub labels: bool,
    pub camera_paths: bool,
    /// Finite display length, not the camera's clipping plane or measured overlap.
    pub frustum_length: f32,
}
impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            labels: true,
            camera_paths: true,
            frustum_length: 1.5,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Projection {
    pub schema_version: u32,
    pub image_size: [u32; 2],
    /// Pixel-boundary coordinates: centers of raster pixels are (i+0.5,j+0.5).
    /// [u,v] = [scale*x + offset_x, scale*z + offset_z]. Y is discarded.
    pub pixels_per_metre: f32,
    pub offset: [f32; 2],
}
impl Projection {
    pub fn project(&self, world: [f32; 3]) -> [f32; 2] {
        [
            world[0] * self.pixels_per_metre + self.offset[0],
            world[2] * self.pixels_per_metre + self.offset[1],
        ]
    }
    pub fn unproject(&self, pixel: [f32; 2], height: f32) -> [f32; 3] {
        [
            (pixel[0] - self.offset[0]) / self.pixels_per_metre,
            height,
            (pixel[1] - self.offset[1]) / self.pixels_per_metre,
        ]
    }
}
impl Schematic {
    /// Apply a recorded root transform delta to diagram geometry and planned trajectories.
    pub fn transform_geometry(&mut self, delta: Mat4) {
        let point = |p: &mut [f32; 3]| {
            *p = delta.transform_point3(Vec3::from_array(*p)).to_array();
        };
        for f in &mut self.footprints {
            f.points.iter_mut().for_each(point);
        }
        for c in &mut self.cameras {
            c.world_from_view =
                (delta * Mat4::from_cols_array_2d(&c.world_from_view)).to_cols_array_2d();
            c.path.iter_mut().for_each(point);
        }
        for p in &mut self.poses {
            p.joints.iter_mut().for_each(point);
        }
    }
    pub fn projection(&self, options: &RenderOptions) -> Result<Projection, String> {
        if self.schema_version != 1
            || !self.trajectory_progress.is_finite()
            || !(0.0..=1.0).contains(&self.trajectory_progress)
            || self.time_seconds.is_some_and(|t| !t.is_finite())
        {
            return Err("invalid schematic version or timeline".into());
        }
        if !(128..=4096).contains(&options.width)
            || !(128..=4096).contains(&options.height)
            || !options.frustum_length.is_finite()
            || !(0.1..=100.).contains(&options.frustum_length)
        {
            return Err("schematic size must be 128..=4096 and frustum length 0.1..=100m".into());
        }
        let floor = self
            .footprints
            .iter()
            .find(|p| p.kind == FootprintKind::Floor)
            .ok_or("schematic needs a primary room floor")?;
        if floor.points.len() < 3 {
            return Err("schematic floor needs three vertices".into());
        }
        let mut lo = Vec2::splat(f32::INFINITY);
        let mut hi = Vec2::splat(f32::NEG_INFINITY);
        for p in &floor.points {
            let p = Vec3::from_array(*p);
            if !p.is_finite() {
                return Err("non-finite footprint".into());
            }
            lo = lo.min(p.xz());
            hi = hi.max(p.xz());
        }
        let extent = hi - lo;
        if extent.min_element() < 0.001 {
            return Err("degenerate schematic floor".into());
        }
        let scale = ((options.width as f32 - 64.) / extent.x)
            .min((options.height as f32 - 112.) / extent.y);
        let center = (hi + lo) * 0.5;
        Ok(Projection {
            schema_version: 1,
            image_size: [options.width, options.height],
            pixels_per_metre: scale,
            offset: [
                options.width as f32 * 0.5 - center.x * scale,
                options.height as f32 * 0.5 - center.y * scale,
            ],
        })
    }
    pub fn svg(&self, options: &RenderOptions, predictions: &Overlay) -> Result<String, String> {
        svg::render(self, options, predictions)
    }
    /// Same deterministic CPU rasterizer on native and Wasm; no GPU initialization.
    pub fn rgba(&self, options: &RenderOptions, predictions: &Overlay) -> Result<Vec<u8>, String> {
        svg::raster(&self.svg(options, predictions)?)
    }
}

impl crate::sample::Sample {
    /// Uses the recorded cameras and skeletons at this capture step, never the
    /// editor camera or resampled poses. Disabled export incurs no work.
    pub fn schematic(&self, step: usize) -> Result<Schematic, String> {
        let manifest = self
            .indoor
            .as_ref()
            .ok_or("schematic requires procedural_indoor")?;
        let count = self.view_dim as usize;
        if count == 0 || !self.views.len().is_multiple_of(count) || step >= self.views.len() / count
        {
            return Err("invalid schematic capture step or camera count".into());
        }
        let views = &self.views[step * count..(step + 1) * count];
        let progress = views[0].trajectory_progress.unwrap_or(views[0].time);
        let mut plan = Schematic::from_manifest(manifest, progress)?;
        if let Some(root) = self
            .indoor_render_metadata
            .as_ref()
            .and_then(|m| m.get("world_from_scene"))
            .filter(|m| !m.is_null())
        {
            let matrix: [[f32; 4]; 4] =
                serde_json::from_value(root.clone()).map_err(|e| e.to_string())?;
            plan.transform_geometry(
                Mat4::from_cols_array_2d(&matrix) * Mat4::from_rotation_y(-manifest.world_yaw),
            );
        }
        plan.time_seconds = views[0].time_seconds;
        let planned = std::mem::take(&mut plan.cameras);
        for (i, v) in views.iter().enumerate() {
            let aspect = v
                .calibration
                .as_ref()
                .map_or(manifest.camera_aspect_ratio, |c| {
                    c.image_size[0] as f32 / c.image_size[1] as f32
                });
            plan.cameras.push(Camera {
                label: format!("C{i}"),
                world_from_view: v.world_from_view,
                fov_y: v.fovy,
                aspect,
                calibration: v.calibration.clone(),
                path: planned.get(i).map_or_else(Vec::new, |c| c.path.clone()),
            });
        }
        plan.poses.clear();
        let poses = self
            .human_pose_steps
            .get(step)
            .or_else(|| (step == 0).then_some(&self.human_poses));
        if let Some(poses) = poses {
            for (i, p) in poses.iter().enumerate() {
                plan.poses.push(Pose {
                    label: format!(
                        "P{}",
                        self.human_instance_ids.get(i).copied().unwrap_or(i as i64)
                    ),
                    joints: p.bone_positions.clone(),
                    parents: self.human_bone_parents.clone(),
                });
            }
        }
        Ok(plan)
    }
}
