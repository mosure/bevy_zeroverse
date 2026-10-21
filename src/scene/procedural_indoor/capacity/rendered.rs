//! Completed Sample -> fixed rendered descriptor. This runs on explicitly
//! requested CPU readbacks; ordinary same-device tensor generation is unchanged.
use super::*;
use crate::sample::Sample;
use bevy::prelude::*;
use serde_json::json;
use std::collections::BTreeMap;

fn rgba(bytes: &[u8], i: usize) -> Result<[f32; 4], String> {
    let pixel = bytes
        .get(i * 16..(i + 1) * 16)
        .ok_or("incomplete annotation plane")?;
    let v =
        std::array::from_fn(|c| f32::from_ne_bytes(pixel[c * 4..c * 4 + 4].try_into().unwrap()));
    if v.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite rendered annotation".into());
    }
    Ok(v)
}
fn quantize(v: f64, step: f64) -> Result<i64, String> {
    let q = (v / step).round();
    if !q.is_finite() || q.abs() >= i64::MAX as f64 {
        return Err("rendered descriptor range overflow".into());
    }
    Ok(q as i64)
}

/// Requires actual provenance, calibration, all requested camera/time deliveries
/// and RGB/position/normal/semantic readbacks. It never accepts placement overlap
/// or an unfinished motion plan as a rendered observation.
pub fn observe_rendered(
    sample: &Sample,
    plan: &CapacityPlan,
    draw: u32,
    mesh_checked: bool,
) -> Result<CapacityRecord, String> {
    validate_plan(plan)?;
    let scene = sample
        .indoor
        .as_ref()
        .ok_or("rendered capacity needs an indoor manifest")?;
    let capture = &plan.population.capture;
    if draw >= plan.request.budget.rendered_rooms
        || sample.views.len() != capture.cameras as usize * capture.progress.len()
        || sample.view_dim != capture.cameras
        || scene.cameras.len() != capture.cameras as usize
        || scene.seed != seed_at(&plan.request.seed_policy, draw)
    {
        return Err("rendered sample has wrong draw/camera/time cardinality".into());
    }
    let metadata = sample
        .indoor_render_metadata
        .as_ref()
        .ok_or("missing renderer provenance")?;
    if sample.color_encoding != crate::render::color::ColorEncoding::TonemappedLinear {
        return Err("rendered-grid v1 expects tone-mapped linear RGB".into());
    }
    if metadata["build_provenance"]["source_sha256"] != plan.population.generator.source_sha256
        || metadata["build_provenance"]["capture_engine"] != plan.population.capture_engine
        || serde_json::to_value(sample.annotation_glass).map_err(|e| e.to_string())?
            != capture.annotation_glass
    {
        return Err("rendered population provenance/glass mismatch".into());
    }
    let n = (u64::from(capture.image_size[0]) * u64::from(capture.image_size[1])) as usize;
    let region = scene.primary_region();
    let inverse = Quat::from_rotation_y(-scene.world_yaw);
    let lo = Vec3::from_array(sample.aabb[0]);
    let size = Vec3::from_array(sample.aabb[1]) - lo;
    if !lo.is_finite() || !size.is_finite() || size.cmple(Vec3::ZERO).any() {
        return Err("invalid reconstruction normalization".into());
    }
    let mut hash_inputs = Vec::new();
    let mut descriptors = BTreeMap::new();
    let mut diagnostics = BTreeMap::new();
    for (index, view) in sample.views.iter().enumerate() {
        let step = index / capture.cameras as usize;
        let camera = index % capture.cameras as usize;
        let calibration = view
            .calibration
            .as_ref()
            .ok_or("missing full camera calibration")?;
        calibration.validate()?;
        let camera_spec = &scene.cameras[camera];
        let mut expected_k =
            bevy_zeroverse_capture::calibration::CameraCalibration::centered_pinhole(
                capture.image_size[0],
                capture.image_size[1],
                camera_spec.fov_degrees.to_radians(),
                capture.image_size[0] as f32 / capture.image_size[1] as f32,
            )?;
        expected_k.k[0][2] +=
            camera_spec.principal_point_offset.unwrap_or_default().x * capture.image_size[0] as f32;
        expected_k.k[1][2] +=
            camera_spec.principal_point_offset.unwrap_or_default().y * capture.image_size[1] as f32;
        if let Some(lens) = camera_spec.fisheye {
            expected_k = lens.calibration(
                capture.image_size,
                camera_spec
                    .principal_point_offset
                    .unwrap_or_default()
                    .to_array(),
            )?;
        }
        if calibration.lens_model != expected_k.lens_model
            || calibration.fisheye != expected_k.fisheye
        {
            return Err("rendered lens differs from accepted camera".into());
        }
        if calibration
            .k
            .iter()
            .flatten()
            .zip(expected_k.k.iter().flatten())
            .any(|(a, b)| (a - b).abs() > 1e-3)
        {
            return Err("rendered calibration differs from accepted camera".into());
        }
        if calibration.image_size != capture.image_size
            || view.trajectory_progress != Some(capture.progress[step])
            || capture.seconds.as_ref().is_some_and(|t| {
                view.time_seconds
                    .is_none_or(|s| (f64::from(s) - t[step]).abs() > 1e-5)
            })
        {
            return Err("rendered schedule/intrinsics mismatch".into());
        }
        let expected = Mat4::from_rotation_y(scene.world_yaw)
            * scene.cameras[camera]
                .transform_at(capture.progress[step])
                .to_matrix();
        if !view
            .world_from_view
            .as_flattened()
            .iter()
            .all(|v| v.is_finite())
            || expected
                .to_cols_array()
                .iter()
                .zip(view.world_from_view.as_flattened())
                .any(|(a, b)| (a - b).abs() > 2e-4)
        {
            return Err("rendered trajectory pose mismatch".into());
        }
        for mode in &capture.modes {
            let plane = match mode.as_str() {
                "color" => &view.color,
                "position" => &view.position,
                "normal" => &view.normal,
                "semantic" => &view.semantic,
                "depth" => &view.depth,
                "optical_flow" => &view.optical_flow,
                "motion_vectors" => &view.motion_vectors,
                "co_visibility" => &view.co_visibility,
                _ => return Err("unsupported captured plane".into()),
            };
            if plane.len() != n * 16 {
                return Err(
                    "rendered-grid definition needs complete RGB/position/normal/semantic planes"
                        .into(),
                );
            }
            if plane
                .as_chunks::<4>()
                .0
                .iter()
                .any(|v| !f32::from_ne_bytes(*v).is_finite())
            {
                return Err(format!("nonfinite {mode} annotation"));
            }
            hash_inputs.push(sha(plane));
        }
        if capture.modes.iter().any(|m| m == "co_visibility") {
            crate::render::co_visibility::validate_plane(
                &view.co_visibility,
                n,
                capture.cameras as usize,
                camera,
            )?;
            let mut shared = 0_u64;
            let mut valid = 0_u64;
            for i in 0..n {
                let v = rgba(&view.co_visibility, i)?;
                valid += u64::from(v[2] > 0.5);
                shared += u64::from(v[2] > 0.5 && v[0] > 0.);
            }
            diagnostics.insert(
                format!("rendered/view_{index}/shared_fraction"),
                shared as f64 / valid.max(1) as f64,
            );
        }
        hash_inputs.push(sha(&serde_json::to_vec(&(
            calibration,
            view.world_from_view,
            view.trajectory_progress,
            view.time_seconds,
        ))
        .map_err(|e| e.to_string())?));
    }
    for d in &plan.request.definitions {
        if d.descriptor != DescriptorKind::RenderedGrid {
            continue;
        }
        let side = d.grid_side as usize;
        let mut views = Vec::new();
        let (mut primary_pixels, mut valid_pixels) = (0_u64, 0_u64);
        for view in &sample.views {
            let mut sums = vec![[0_f64; 10]; side * side];
            let mut counts = vec![0_u32; side * side];
            let mut semantics = vec![BTreeMap::<[i64; 3], u32>::new(); side * side];
            for i in 0..n {
                let p = rgba(&view.position, i)?;
                let color = rgba(&view.color, i)?;
                let normal = rgba(&view.normal, i)?;
                let semantic = rgba(&view.semantic, i)?;
                if p[3] < 0.5 {
                    continue;
                }
                valid_pixels += 1;
                let world = lo + Vec3::new(p[0], p[1], p[2]) * size;
                let primary = region.contains_surface(inverse * world, 0.03);
                primary_pixels += u64::from(primary);
                if d.primary_only && !primary {
                    continue;
                }
                let x = i % capture.image_size[0] as usize;
                let y = i / capture.image_size[0] as usize;
                let cell = (y * side / capture.image_size[1] as usize) * side
                    + x * side / capture.image_size[0] as usize;
                counts[cell] += 1;
                for k in 0..3 {
                    sums[cell][k] += f64::from(color[k]);
                    sums[cell][k + 3] += f64::from(normal[k]);
                    sums[cell][k + 6] += f64::from(world[k]);
                }
                *semantics[cell]
                    .entry([
                        quantize(f64::from(semantic[0]), 0.001)?,
                        quantize(f64::from(semantic[1]), 0.001)?,
                        quantize(f64::from(semantic[2]), 0.001)?,
                    ])
                    .or_default() += 1;
            }
            let cells:Vec<_>=sums.into_iter().zip(&counts).zip(semantics).map(|((sum,&count),semantic)|{
                let denominator=f64::from(count.max(1));
                let values:Vec<_>=sum[..9].iter().enumerate().map(|(k,v)|quantize(v/denominator,if k>=6{d.metres}else{d.appearance_step})).collect::<Result<_,_>>()?;
                let fraction=f64::from(count)/(n as f64/(side*side) as f64);
                let mut histogram=Vec::new();
                for (label,n) in semantic{histogram.push((label,quantize(f64::from(n)/denominator,d.appearance_step)?));}
                Ok(json!({"occupied":quantize(fraction,d.appearance_step)?,"mean":values,"semantic":histogram}))
            }).collect::<Result<_,String>>()?;
            let calibration = view.calibration.as_ref().unwrap();
            let k: Vec<_> = calibration
                .k
                .iter()
                .enumerate()
                .map(|(r, row)| {
                    row.iter()
                        .map(|&v| {
                            quantize(
                                f64::from(v)
                                    / if r < 2 {
                                        f64::from(capture.image_size[r])
                                    } else {
                                        1.
                                    },
                                d.appearance_step,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<_, _>>()?;
            let pose: Vec<_> = view
                .world_from_view
                .iter()
                .enumerate()
                .map(|(c, col)| {
                    col.iter()
                        .map(|&v| quantize(f64::from(v), if c == 3 { d.metres } else { d.radians }))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<_, _>>()?;
            views.push(json!({"cells":cells,"normalized_k":k,"pose":pose}));
        }
        // Preserve view ordering for calibration supervision; deduplicate exact
        // repeated complete frames, not independently shuffled camera images.
        let mut frames: Vec<_> = views
            .chunks(capture.cameras as usize)
            .map(|v| json!(v))
            .collect();
        frames.dedup();
        let value = serde_json::to_vec(&frames).map_err(|e| e.to_string())?;
        let class_sha256 = sha(&value);
        descriptors.insert(
            d.id.clone(),
            Descriptor {
                class_sha256: class_sha256.clone(),
                blocks: BTreeMap::from([("rendered_joint".into(), class_sha256)]),
            },
        );
        diagnostics.insert("rendered/primary_pixels".into(), primary_pixels as f64);
        diagnostics.insert("rendered/valid_pixels".into(), valid_pixels as f64);
    }
    // A program descriptor is evaluated only when actor motion is absent. Motion
    // runs use their actual image/geometry grids and cannot claim static-program
    // evidence for moving poses.
    let mut family = super::family_identity(scene, plan);
    if plan
        .request
        .definitions
        .iter()
        .any(|d| d.descriptor == DescriptorKind::AcceptedProgram)
    {
        let mut cpu_plan = plan.clone();
        cpu_plan
            .request
            .definitions
            .retain(|d| d.descriptor == DescriptorKind::AcceptedProgram);
        cpu_plan.request.budget.rendered_rooms = 0;
        let record = super::observe(scene, &cpu_plan, draw, mesh_checked)?;
        if let SampleOutcome::Accepted {
            family_sha256,
            descriptors: programs,
            diagnostics: counts,
            ..
        } = record.outcome
        {
            family = family_sha256;
            descriptors.extend(programs);
            diagnostics.extend(counts);
        }
    }
    let record = CapacityRecord {
        draw,
        seed: scene.seed,
        mesh_checked,
        outcome: SampleOutcome::Accepted {
            family_sha256: family,
            descriptors,
            diagnostics,
            delivered_views: capture.cameras,
            delivered_progress: capture.progress.clone(),
            capture_sha256: Some(sha(
                &serde_json::to_vec(&hash_inputs).map_err(|e| e.to_string())?
            )),
        },
    };
    validate_record(plan, &record)?;
    Ok(record)
}
