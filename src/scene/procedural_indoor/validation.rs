//! Acceptance checks shared by the runtime, CPU distribution audit and regression tests.
use super::layout::{IndoorLayout, IndoorManifest, CAMERA_CLEARANCE, GENERATOR_VERSION};
use bevy::prelude::*;
use serde::Serialize;
use std::{borrow::Cow, collections::BTreeMap};
mod geometry;
#[cfg(test)]
mod tests;
pub use geometry::{validate_geometry, GeometryStats};

pub fn validate_layout(scene: &IndoorManifest) -> Result<(), String> {
    // Validate base scalars before dependent programs divide by room dimensions
    // or inspect objects. Malformed archived manifests must fail without panic.
    if scene.generator_version != GENERATOR_VERSION
        || !scene.room_size.is_finite()
        || scene.room_size.min_element() <= 0.0
        || !scene.world_yaw.is_finite()
        || !(0.0..=1.0).contains(&scene.density)
        || !(0.0..=1.0).contains(&scene.human_density)
        || !scene.sun_elevation.is_finite()
        || !scene.sun_azimuth.is_finite()
        || !(0.0..=150_000.0).contains(&scene.daylight_lux)
        || !(0.0..=10_000.0).contains(&scene.target_lux)
        || !(1000.0..=12_000.0).contains(&scene.light_kelvin)
    {
        return Err(format!(
            "seed {}: invalid scene dimensions, density, transform or lighting",
            scene.seed
        ));
    }
    scene.camera_settings.validate()?;
    if !scene.camera_aspect_ratio.is_finite() || scene.camera_aspect_ratio <= 0.0 {
        return Err("camera aspect ratio must be positive and finite".into());
    }
    if let Some(program) = &scene.program {
        program.validate(scene)?;
    }
    if let Some(exterior) = &scene.exterior {
        exterior.validate(scene.room_size)?;
    }
    if let Some(envelope) = &scene.envelope {
        envelope.validate(scene)?;
    }
    super::humans::validate(scene)?;
    let fail = |message: &str| Err(format!("seed {}: {message}", scene.seed));
    if scene.objects.is_empty() {
        return fail("empty scene");
    }
    let mut main_furniture = 0;
    for (i, object) in scene.objects.iter().enumerate() {
        if object.id != i
            || !object.position.is_finite()
            || !object.size.is_finite()
            || object.size.min_element() <= 0.0
            || !object.yaw.is_finite()
        {
            return fail("invalid instance dimensions, transform or identity");
        }
        if let Some(target) = object.interaction_target {
            let other = scene
                .objects
                .get(target)
                .ok_or_else(|| format!("seed {}: missing interaction target", scene.seed))?;
            let valid = if object.kind == super::layout::ObjectKind::Chair {
                matches!(
                    other.kind,
                    super::layout::ObjectKind::Desk
                        | super::layout::ObjectKind::Table
                        | super::layout::ObjectKind::CoffeeTable
                )
            } else {
                matches!(
                    object.kind,
                    super::layout::ObjectKind::Laptop | super::layout::ObjectKind::Monitor
                ) && other.kind == super::layout::ObjectKind::Chair
                    && other.interaction_target == object.support
            };
            if !valid || other.neighbor != object.neighbor {
                return fail("inconsistent interaction relationship");
            }
            if matches!(
                object.kind,
                super::layout::ObjectKind::Laptop | super::layout::ObjectKind::Monitor
            ) {
                let direction = (other.position - object.position)
                    .with_y(0.0)
                    .normalize_or_zero();
                if (Quat::from_rotation_y(object.yaw) * Vec3::Z).dot(direction) < 0.98 {
                    return fail("screen faces away from intended user");
                }
            }
        }
        if !object.neighbor
            && matches!(
                object.kind,
                super::layout::ObjectKind::Display
                    | super::layout::ObjectKind::Whiteboard
                    | super::layout::ObjectKind::WallArt
                    | super::layout::ObjectKind::Clock
                    | super::layout::ObjectKind::WallOutlet
                    | super::layout::ObjectKind::LightSwitch
            )
            && !scene.wall_attachment_clear(object)
        {
            return fail(
                "wall fixture is unmounted or lacks solid backing beside exterior aperture",
            );
        }
        if object.solid && !object.neighbor {
            main_furniture += 1;
            let (lo, hi) = object.bounds();
            let half = scene.room_size * 0.5;
            if lo.x < -half.x + 0.299
                || hi.x > half.x - 0.299
                || lo.z < -half.z + 0.299
                || hi.z > half.z - 0.299
            {
                return fail("furniture intersects room boundary");
            }
            if !scene.floor_support_clear(lo, hi) {
                return fail("unsupported floor furniture");
            }
            if hi.x > scene.door_x - 0.70 && lo.x < scene.door_x + 0.70 && hi.z > half.z - 1.45 {
                return fail("door approach blocked");
            }
            if !scene.placement_clear(object, 0.0) {
                return fail("furniture intersects another object, a pillar, or the room boundary");
            }
            for other in scene
                .objects
                .iter()
                .skip(i + 1)
                .filter(|o| o.solid && !o.neighbor)
            {
                let (a, b) = other.bounds();
                if lo.y < b.y
                    && hi.y > a.y
                    && super::footprint::Footprint::object(object)
                        .overlaps(super::footprint::Footprint::object(other), 0.0)
                {
                    return fail(&format!(
                        "furniture overlap: {} and {}",
                        object.id, other.id
                    ));
                }
            }
        }
        if object.solid {
            let (lo, hi) = object.bounds();
            if (object.neighbor && lo.y.abs() > 1e-5)
                || (!object.neighbor && !scene.floor_support_clear(lo, hi))
                || hi.y > scene.ceiling_height(object.position.xz()) - 0.30
            {
                return fail("floor furniture floats, sinks or intersects ceiling fixtures");
            }
            if object.neighbor {
                let half = scene.room_size * 0.5;
                if lo.x < -half.x + 0.299
                    || hi.x > half.x - 0.299
                    || lo.z < half.z + 0.299
                    || hi.z > half.z + super::layout::NEIGHBOR_DEPTH - 0.299
                {
                    return fail("neighbor furniture intersects room boundary or partition");
                }
                for other in scene
                    .objects
                    .iter()
                    .skip(i + 1)
                    .filter(|o| o.solid && o.neighbor)
                {
                    let (a, b) = other.bounds();
                    if lo.x < b.x && hi.x > a.x && lo.z < b.z && hi.z > a.z {
                        return fail("neighbor furniture overlap");
                    }
                }
                // Keep the open leaf, handle and approach in the neighboring room clear.
                if hi.x > scene.door_x - 0.70 && lo.x < scene.door_x + 0.70 && lo.z < half.z + 1.45
                {
                    return fail("neighbor door approach blocked");
                }
            }
        }
        if let Some(support) = object.support {
            let Some(parent) = scene.objects.get(support) else {
                return fail("missing prop support");
            };
            if !parent.kind.is_surface() {
                return fail("prop support is not a surface");
            }
            if parent.neighbor != object.neighbor || parent.id >= object.id {
                return fail("prop has an inconsistent support relationship");
            }
            if !scene.prop_clear(object, parent, 0.0) {
                return fail("supported prop overlaps a peer or overhangs its support");
            }
            if (object.position.y - parent.position.y - parent.size.y).abs() > 0.002 {
                return fail(&format!(
                    "prop floats or sinks into support: {:?} instance {} at y={:.6}, {:?} support {} top={:.6}",
                    object.kind, object.id, object.position.y, parent.kind, parent.id,
                    parent.position.y + parent.size.y
                ));
            }
            let local = parent
                .transform()
                .compute_affine()
                .inverse()
                .transform_point3(object.position);
            let yaw = object.yaw - parent.yaw;
            let half_x = (yaw.cos().abs() * object.size.x + yaw.sin().abs() * object.size.z) * 0.5;
            let half_z = (yaw.sin().abs() * object.size.x + yaw.cos().abs() * object.size.z) * 0.5;
            if local.x.abs() + half_x > parent.size.x * 0.5 + 0.002
                || local.z.abs() + half_z > parent.size.z * 0.5 + 0.002
            {
                return fail("prop overhangs support");
            }
        }
    }
    if !scene.objects.iter().any(|o| {
        !o.neighbor
            && matches!(
                o.kind,
                super::layout::ObjectKind::Desk
                    | super::layout::ObjectKind::Table
                    | super::layout::ObjectKind::CoffeeTable
            )
    }) {
        return fail("missing primary activity furniture");
    }
    if main_furniture < scene.minimum_main_objects() {
        return fail("insufficient main room furniture");
    }
    for camera in &scene.cameras {
        if camera
            .motion
            .as_ref()
            .and_then(|m| m.orientations)
            .is_some_and(|q| {
                q.iter()
                    .any(|r| !r.is_finite() || (r.length_squared() - 1.).abs() > 1e-4)
            })
        {
            return fail("invalid free camera orientation");
        }
        if !camera.target.is_finite()
            || !(27.0..=108.0).contains(&camera.fov_degrees)
            || camera.start.distance(camera.target) < 1.5
        {
            return fail("invalid camera intrinsics or target");
        }
        if !scene.camera_curve_clear(camera) {
            return fail("camera path intersects geometry");
        }
        // Explicit samples guard the continuous sweep implementation independently.
        for step in 0..=32 {
            if !scene.camera_clear(camera.transform_at(step as f32 / 32.0).translation) {
                return fail("unsafe sampled camera path");
            }
        }
    }
    if let Some(policy) = &scene.camera_settings.multiview {
        if scene
            .camera_group_geometry()
            .is_some_and(|g| !g.accepts(policy))
        {
            return fail(
                "multi-view camera spacing, group spread or trajectory variation constraint violated",
            );
        }
        if scene
            .camera_overlap()
            .iter()
            .any(|pair| !scene.accepts_camera_pair(pair))
        {
            return Err(format!(
                "seed {}: multi-view camera overlap or baseline constraint violated",
                scene.seed
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct AnnotationAlignment {
    pub checked_pixels: usize,
    pub depth_position_p99_metres: f32,
    pub depth_position_max_metres: f32,
    pub reprojection_p99_pixels: f32,
    pub reprojection_max_pixels: f32,
    pub normal_length_max_error: f32,
    pub position_quantization_budget_p99_ratio: f32,
}

/// Independent check that geometric modalities and camera calibration describe
/// the same rendered frame. Position is decoded with the captured scene AABB.
pub fn validate_annotations(
    view: &crate::sample::View,
    aabb: [[f32; 3]; 2],
    width: u32,
    height: u32,
) -> Result<AnnotationAlignment, String> {
    validate_annotations_with_precision(
        view,
        aabb,
        width,
        height,
        crate::sample::AnnotationPrecision::Float16Hdr,
    )
}

// Native tightly packed render planes already contain f32 values. Borrow when
// aligned; archives and user-provided buffers may require an unaligned fallback.
// Both paths retain every channel's bits and scan all values, including misses.
fn annotation_values(bytes: &[u8], expected: usize) -> Result<Cow<'_, [f32]>, String> {
    if bytes.len() != expected || !bytes.len().is_multiple_of(4) {
        return Err("wrong annotation buffer length".into());
    }
    let values = match bytemuck::try_cast_slice::<u8, f32>(bytes) {
        Ok(values) => Cow::Borrowed(values),
        Err(_) => Cow::Owned(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_ne_bytes(*b))
                .collect(),
        ),
    };
    if values.iter().any(|v| !v.is_finite()) {
        return Err("non-finite annotation".into());
    }
    Ok(values)
}

// Only the exact historical order statistic is exported. Selection preserves
// total ordering (including signed zero) without sorting the other samples.
fn percentile99(values: &mut [f32]) -> f32 {
    let index = (values.len() as f32 * 0.99) as usize;
    *values.select_nth_unstable_by(index, f32::total_cmp).1
}

pub fn validate_annotations_with_precision(
    view: &crate::sample::View,
    aabb: [[f32; 3]; 2],
    width: u32,
    height: u32,
    precision: crate::sample::AnnotationPrecision,
) -> Result<AnnotationAlignment, String> {
    let full_precision = precision == crate::sample::AnnotationPrecision::Float32Geometry;
    let expected_bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(16))
        .ok_or("wrong annotation buffer length")?;
    let depth = annotation_values(&view.depth, expected_bytes)?;
    let position = annotation_values(&view.position, expected_bytes)?;
    let normal = annotation_values(&view.normal, expected_bytes)?;
    let world_from_view = Mat4::from_cols_array_2d(&view.world_from_view);
    let view_from_world = world_from_view.inverse();
    let lo = Vec3::from_array(aabb[0]);
    let range = Vec3::from_array(aabb[1]) - lo;
    let aspect = width as f32 / height as f32;
    let calibration = match &view.calibration {
        Some(calibration) => Cow::Borrowed(calibration),
        None => Cow::Owned(crate::calibration::CameraCalibration::centered_pinhole(
            width, height, view.fovy, aspect,
        )?),
    };
    calibration.validate()?;
    if calibration.image_size != [width, height] {
        return Err("calibration/image dimensions disagree".into());
    }
    // The historical percentile grid has a two-pixel border. Small valid
    // captures cannot qualify that grid; reject them without unsigned underflow.
    // Buffer/finite/calibration checks above retain their original error order.
    if width < 5 || height < 5 {
        return Err("insufficient visible annotation pixels".into());
    }
    let k = calibration.k;
    let pixel_footprint = world_from_view.transform_vector3(Vec3::X / k[0][0]).abs()
        + world_from_view
            .transform_vector3(Vec3::new(-k[0][1] / (k[0][0] * k[1][1]), -1. / k[1][1], 0.))
            .abs();
    let mut errors = Vec::new();
    let mut reproject = Vec::new();
    let mut checked_pixels = 0;
    let mut depth_error_max = 0.0_f32;
    let mut reprojection_error_max = 0.0_f32;
    let mut normal_error = 0.0_f32;
    let mut quantization_ratios = Vec::new();
    // Float32 qualification checks every foreground pixel: a single thin
    // triangle can fail while both a regular grid and p99 remain excellent.
    // Keep percentile samples bounded to the historical grid; maxima are exact.
    let (start, end_margin, stride) = if full_precision { (0, 0, 1) } else { (2, 2, 3) };
    for y in (start..height - end_margin).step_by(stride) {
        for x in (start..width - end_margin).step_by(stride) {
            let index = ((y * width + x) * 4) as usize;
            if depth[index + 3] < 0.5 || position[index + 3] < 0.5 {
                continue;
            }
            let p =
                lo + Vec3::new(position[index], position[index + 1], position[index + 2]) * range;
            let v = view_from_world.transform_point3(p);
            if -v.z < 0.1 {
                return Err("position behind camera".into());
            }
            checked_pixels += 1;
            let depth_error = (depth[index] + v.z).abs();
            depth_error_max = depth_error_max.max(depth_error);
            let [px, py] = calibration
                .project(v.to_array())
                .ok_or("invalid reprojected point")?;
            let reprojection_error = Vec2::new(px - x as f32 - 0.5, py - y as f32 - 0.5).length();
            reprojection_error_max = reprojection_error_max.max(reprojection_error);
            let n =
                Vec3::new(normal[index], normal[index + 1], normal[index + 2]) * 2.0 - Vec3::ONE;
            normal_error = normal_error.max((n.length() - 1.0).abs());
            if full_precision
                && !(x >= 2
                    && x < width - 2
                    && y >= 2
                    && y < height - 2
                    && x % 3 == 2
                    && y % 3 == 2)
            {
                continue;
            }
            errors.push(depth_error);
            reproject.push(reprojection_error);
            let ray = Vec3::from_array(
                calibration
                    .unproject([x as f32 + 0.5, y as f32 + 0.5], 1.)
                    .ok_or("invalid calibrated ray")?,
            );
            let expected = world_from_view.transform_point3(ray * depth[index]);
            // Conservative one-ULP error bounds for two independently quantized
            // float16 render targets, propagated into world coordinates.
            let budget = if full_precision {
                range * (8.0 * f32::EPSILON)
                    + world_from_view.transform_vector3(ray).abs()
                        * (depth[index] * 8.0 * f32::EPSILON)
                    + Vec3::splat(1e-5)
                    // Rasterized vertex positions use subpixel fixed-point precision.
                    // Bound a 0.01-pixel ray-footprint separately from f32 storage ULPs.
                    + pixel_footprint * (depth[index] * 0.01)
            } else {
                range / 2048.0
                    + world_from_view.transform_vector3(ray).abs() * (depth[index] / 1024.0)
                    + Vec3::splat(0.0005)
            };
            quantization_ratios.push(((p - expected).abs() / budget).max_element());
        }
    }
    if errors.len() < 100 {
        return Err("insufficient visible annotation pixels".into());
    }
    let report = AnnotationAlignment {
        checked_pixels,
        depth_position_p99_metres: percentile99(&mut errors),
        depth_position_max_metres: depth_error_max,
        reprojection_p99_pixels: percentile99(&mut reproject),
        reprojection_max_pixels: reprojection_error_max,
        normal_length_max_error: normal_error,
        position_quantization_budget_p99_ratio: percentile99(&mut quantization_ratios),
    };
    // The existing HDR path quantizes intermediate values to float16. Bounds
    // include the exterior, so world-position precision is centimetres here.
    if report.position_quantization_budget_p99_ratio > 1.0
        || report.normal_length_max_error > if full_precision { 2e-5 } else { 0.012 }
        || (full_precision
            && (report.depth_position_max_metres > 0.00005
                || report.reprojection_max_pixels > 0.05))
    {
        return Err(format!("annotation alignment failed: {report:?}"));
    }
    Ok(report)
}

#[derive(Debug, Serialize)]
pub struct DistributionReport {
    pub generator_version: u32,
    pub seeds: usize,
    pub first_seed: u64,
    pub cameras_per_scene: usize,
    pub camera_clearance_metres: f32,
    pub layout_counts: BTreeMap<String, usize>,
    pub architecture_counts: BTreeMap<String, usize>,
    pub plant_species_counts: BTreeMap<String, usize>,
    pub palette_counts: BTreeMap<String, usize>,
    pub floor_counts: BTreeMap<String, usize>,
    pub lighting_counts: BTreeMap<String, usize>,
    pub object_counts: BTreeMap<String, usize>,
    pub object_count_range: [usize; 2],
    pub room_size_range: [Vec3; 2],
    pub camera_height_range: [f32; 2],
    pub fov_range: [f32; 2],
    pub invalid_seeds: Vec<(u64, String)>,
}

pub fn audit(first_seed: u64, seeds: usize, cameras: usize, density: f32) -> DistributionReport {
    audit_layout(first_seed, seeds, cameras, density, IndoorLayout::Mixed)
}

pub fn audit_layout(
    first_seed: u64,
    seeds: usize,
    cameras: usize,
    density: f32,
    layout: IndoorLayout,
) -> DistributionReport {
    audit_layout_with_humans(first_seed, seeds, cameras, density, layout, 0.25)
}

pub fn audit_layout_with_humans(
    first_seed: u64,
    seeds: usize,
    cameras: usize,
    density: f32,
    layout: IndoorLayout,
    human_density: f32,
) -> DistributionReport {
    audit_layout_with_camera_settings(
        first_seed,
        seeds,
        cameras,
        density,
        layout,
        human_density,
        &default(),
        1.0,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn audit_layout_with_camera_settings(
    first_seed: u64,
    seeds: usize,
    cameras: usize,
    density: f32,
    layout: IndoorLayout,
    human_density: f32,
    camera_settings: &super::cameras::CameraSettings,
    aspect_ratio: f32,
) -> DistributionReport {
    audit_layout_with_factors(
        first_seed,
        seeds,
        cameras,
        density,
        layout,
        human_density,
        camera_settings,
        aspect_ratio,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn audit_layout_with_factors(
    first_seed: u64,
    seeds: usize,
    cameras: usize,
    density: f32,
    layout: IndoorLayout,
    human_density: f32,
    camera_settings: &super::cameras::CameraSettings,
    aspect_ratio: f32,
    appearance: Option<&super::appearance::AppearanceSettings>,
) -> DistributionReport {
    let mut report = DistributionReport {
        generator_version: GENERATOR_VERSION,
        seeds,
        first_seed,
        cameras_per_scene: cameras,
        camera_clearance_metres: CAMERA_CLEARANCE,
        layout_counts: BTreeMap::new(),
        architecture_counts: BTreeMap::new(),
        plant_species_counts: BTreeMap::new(),
        palette_counts: BTreeMap::new(),
        floor_counts: BTreeMap::new(),
        lighting_counts: BTreeMap::new(),
        object_counts: BTreeMap::new(),
        object_count_range: [usize::MAX, 0],
        room_size_range: [Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)],
        camera_height_range: [f32::INFINITY, f32::NEG_INFINITY],
        fov_range: [f32::INFINITY, f32::NEG_INFINITY],
        invalid_seeds: Vec::new(),
    };
    for offset in 0..seeds {
        let seed = first_seed.wrapping_add(offset as u64);
        let scene =
            match IndoorManifest::generate_with_humans(seed, layout, density, 0, human_density)
                .and_then(|mut scene| {
                    if let Some(settings) = appearance {
                        scene.apply_appearance(settings.clone())?;
                    }
                    scene.resample_cameras(cameras, camera_settings.clone(), aspect_ratio)?;
                    Ok(scene)
                }) {
                Ok(s) => s,
                Err(e) => {
                    report.invalid_seeds.push((seed, e));
                    continue;
                }
            };
        if let Err(e) = validate_layout(&scene) {
            report.invalid_seeds.push((seed, e));
        }
        *report
            .layout_counts
            .entry(format!("{:?}", scene.layout))
            .or_default() += 1;
        *report
            .architecture_counts
            .entry(format!("{:?}", scene.architecture_style))
            .or_default() += 1;
        for plant in scene
            .objects
            .iter()
            .filter(|o| o.kind == super::layout::ObjectKind::Plant)
        {
            *report
                .plant_species_counts
                .entry(plant.variant.to_string())
                .or_default() += 1;
        }
        *report
            .palette_counts
            .entry(scene.palette.to_string())
            .or_default() += 1;
        *report
            .floor_counts
            .entry(scene.floor_style.to_string())
            .or_default() += 1;
        *report
            .lighting_counts
            .entry(format!("{:?}", scene.lighting))
            .or_default() += 1;
        for o in &scene.objects {
            *report
                .object_counts
                .entry(format!("{:?}", o.kind))
                .or_default() += 1;
        }
        report.object_count_range[0] = report.object_count_range[0].min(scene.objects.len());
        report.object_count_range[1] = report.object_count_range[1].max(scene.objects.len());
        report.room_size_range[0] = report.room_size_range[0].min(scene.room_size);
        report.room_size_range[1] = report.room_size_range[1].max(scene.room_size);
        for c in &scene.cameras {
            report.camera_height_range[0] = report.camera_height_range[0].min(c.start.y);
            report.camera_height_range[1] = report.camera_height_range[1].max(c.start.y);
            report.fov_range[0] = report.fov_range[0].min(c.fov_degrees);
            report.fov_range[1] = report.fov_range[1].max(c.fov_degrees);
        }
    }
    report
}
