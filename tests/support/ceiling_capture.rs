//! Independent CPU checks on completed ceiling-rig readbacks. Boundary pixels
//! are excluded from the co-visibility oracle, never from captures or datasets.
use bevy::prelude::*;
use bevy_zeroverse::sample::View;

fn field(bytes: &[u8], i: usize) -> [f32; 4] {
    bytemuck::pod_read_unaligned(&bytes[16 * i..16 * (i + 1)])
}

pub fn fixed_timeline(views: &[View], cameras: usize) -> serde_json::Value {
    assert!(cameras > 0 && !views.is_empty() && views.len().is_multiple_of(cameras));
    let mut valid = 0;
    let mut max_flow = 0_f32;
    for frame in views.chunks_exact(cameras) {
        for (index, view) in frame.iter().enumerate() {
            assert_eq!(
                view.world_from_view, views[index].world_from_view,
                "mounted camera drift"
            );
            assert_eq!(
                view.calibration, views[index].calibration,
                "mounted intrinsics drift"
            );
            for pixel in view.optical_flow.as_chunks::<16>().0 {
                let flow: [f32; 4] = bytemuck::pod_read_unaligned(pixel);
                if flow[2] > 0. {
                    assert!(flow.iter().all(|x| x.is_finite()));
                    valid += 1;
                    max_flow = max_flow.max(flow[0].abs().max(flow[1].abs()));
                }
            }
        }
    }
    assert!(max_flow < 0.01, "static CCTV flow {max_flow} pixels");
    serde_json::json!({"valid_flow_pixels":valid,"maximum_flow_pixels":max_flow,"extrinsics_and_intrinsics_fixed":true})
}

pub fn visibility(views: &[View], cameras: usize, aabb: [[f32; 3]; 2]) -> serde_json::Value {
    let lo = Vec3::from_array(aabb[0]);
    let range = Vec3::from_array(aabb[1]) - lo;
    let mut checked = 0;
    let mut mismatches = 0;
    for frame in views.chunks_exact(cameras) {
        let matrices: Vec<_> = frame
            .iter()
            .map(|v| Mat4::from_cols_array_2d(&v.world_from_view))
            .collect();
        for (source, v) in frame.iter().enumerate() {
            let source_lens = v.calibration.as_ref().unwrap();
            let [width, height] = source_lens.image_size.map(|v| v as usize);
            for i in (0..width * height).step_by(3) {
                let p = field(&v.position, i);
                if p[3] < 0.5 {
                    continue;
                }
                let world = lo + Vec3::new(p[0], p[1], p[2]) * range;
                let normal = field(&v.normal, i);
                let n = matrices[source]
                    .transform_vector3(Vec3::new(normal[0], normal[1], normal[2]) * 2. - Vec3::ONE)
                    .normalize();
                let bits = field(&v.co_visibility, i)[0] as u32;
                assert_eq!(bits & (1 << source), 0, "self visibility bit");
                assert_eq!(bits >> cameras, 0, "unknown camera bit");
                for (target, t) in frame
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != source)
                {
                    let calibration = t.calibration.as_ref().unwrap();
                    let local = matrices[target].inverse().transform_point3(world);
                    let Some(pixel) = calibration.project(local.to_array()) else {
                        continue;
                    };
                    let depth = calibration.depth(local.to_array());
                    let [tw, th] = calibration.image_size.map(|v| v as usize);
                    if depth < t.near
                        || depth > t.far
                        || pixel[0] < 1.
                        || pixel[1] < 1.
                        || pixel[0] >= tw as f32 - 1.
                        || pixel[1] >= th as f32 - 1.
                    {
                        continue;
                    }
                    // Avoid reprojection roundoff selecting adjacent pixel cells.
                    if pixel.iter().any(|x| x.fract() < 0.02 || x.fract() > 0.98) {
                        continue;
                    }
                    let x = pixel[0] as usize;
                    let y = pixel[1] as usize;
                    let at = y * tw + x;
                    let hit = field(&t.position, at);
                    if hit[3] < 0.5 {
                        continue;
                    }
                    let normal = field(&t.normal, at);
                    if !(-1..=1).all(|dy| {
                        (-1..=1).all(|dx| {
                            let k = ((y as i32 + dy) as usize) * tw + (x as i32 + dx) as usize;
                            let q = field(&t.normal, k);
                            (0..3).all(|i| (q[i] - normal[i]).abs() < 0.0001)
                                && field(&t.position, k)[3] > 0.5
                        })
                    }) {
                        continue;
                    }
                    let tn = matrices[target]
                        .transform_vector3(
                            Vec3::new(normal[0], normal[1], normal[2]) * 2. - Vec3::ONE,
                        )
                        .normalize();
                    let delta = lo + Vec3::new(hit[0], hit[1], hit[2]) * range - world;
                    let distance = delta.dot(n).abs().max(delta.dot(tn).abs());
                    let tolerance = 0.001_f32.max(0.0001 * depth);
                    if (distance - tolerance).abs() < 0.0002 {
                        continue;
                    }
                    checked += 1;
                    mismatches +=
                        usize::from((bits & (1 << target) != 0) != (distance < tolerance));
                }
            }
        }
    }
    assert!(
        checked > 1000,
        "insufficient interior co-visibility comparisons: {checked}"
    );
    assert_eq!(mismatches, 0, "ceiling co-visibility oracle mismatch");
    serde_json::json!({"interior_relations_checked":checked,"mismatches":mismatches,"scope":"rendered surface planes, excluding image boundaries and discontinuities from qualification only"})
}

/// Compare captured extrinsic increments against the archived runtime trajectory;
/// this is invariant to scene augmentation and checks both room and ceiling slots.
pub fn mixed_timeline(
    views: &[View],
    scene: &bevy_zeroverse::scene::procedural_indoor::layout::IndoorManifest,
) -> serde_json::Value {
    let count = scene.cameras.len();
    let mut max_error = 0_f32;
    let mut moving = 0;
    for (index, camera) in scene.cameras.iter().enumerate() {
        let first = &views[index];
        let reference = Mat4::from_cols_array_2d(&first.world_from_view).inverse();
        let start = camera
            .transform_at(first.trajectory_progress.unwrap())
            .to_matrix()
            .inverse();
        moving += usize::from(camera.path_length() > 0.001);
        for frame in views.chunks_exact(count) {
            let view = &frame[index];
            assert_eq!(
                view.calibration, first.calibration,
                "intrinsics changed along path"
            );
            let observed = reference * Mat4::from_cols_array_2d(&view.world_from_view);
            let expected = start
                * camera
                    .transform_at(view.trajectory_progress.unwrap())
                    .to_matrix();
            for (a, b) in observed
                .to_cols_array()
                .into_iter()
                .zip(expected.to_cols_array())
            {
                max_error = max_error.max((a - b).abs());
            }
        }
    }
    assert!(
        max_error < 0.0001,
        "captured trajectory mismatch {max_error}"
    );
    serde_json::json!({"matrix_increment_max_error":max_error,"moving_cameras":moving,"cameras":count,"timesteps":views.len()/count})
}

/// Static scene correspondence oracle for a moving mixed rig. A source world
/// point is projected through the next captured calibration, independent of the
/// renderer's motion-vector implementation. This fixture enables camera motion
/// only (human/furnishing trajectories are disabled).
pub fn camera_flow(views: &[View], cameras: usize, aabb: [[f32; 3]; 2]) -> serde_json::Value {
    let lo = Vec3::from_array(aabb[0]);
    let size = Vec3::from_array(aabb[1]) - lo;
    let mut errors = Vec::new();
    let frames: Vec<_> = views.chunks_exact(cameras).collect();
    for pair in frames.windows(2) {
        for (source, target) in pair[0].iter().zip(pair[1]) {
            let calibration = target.calibration.as_ref().unwrap();
            let view = Mat4::from_cols_array_2d(&target.world_from_view).inverse();
            let width = calibration.image_size[0] as usize;
            for i in (0..source.position.len() / 16).step_by(5) {
                let point = field(&source.position, i);
                let flow = field(&source.optical_flow, i);
                if point[3] < 0.5 || flow[2] < 0.5 {
                    continue;
                }
                let world = lo + Vec3::new(point[0], point[1], point[2]) * size;
                let Some(pixel) = calibration.project(view.transform_point3(world).to_array())
                else {
                    continue;
                };
                let expected = Vec2::from_array(pixel)
                    - Vec2::new((i % width) as f32 + 0.5, (i / width) as f32 + 0.5);
                errors.push(expected.distance(Vec2::new(flow[0], flow[1])));
            }
        }
    }
    errors.sort_by(f32::total_cmp);
    assert!(errors.len() > 1000, "insufficient flow comparisons");
    let p99 = errors[(errors.len() - 1) * 99 / 100];
    assert!(p99 < 0.1, "mixed camera flow p99 error {p99} pixels");
    serde_json::json!({"checked":errors.len(),"p99_error_pixels":p99,"maximum_error_pixels":errors.last()})
}
