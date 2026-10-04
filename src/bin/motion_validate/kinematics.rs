//! Physical-time diagnostics of admitted model clips. No inference-time filter.
use bevy::prelude::*;
use bevy_zeroverse::human_motion::{validation, MotionPlan};
use burn_human_motion::MotionClip;

fn quantiles(mut values: Vec<f32>) -> serde_json::Value {
    values.sort_by(f32::total_cmp);
    if values.is_empty() {
        return serde_json::Value::Null;
    }
    serde_json::json!({"p05":values[((values.len()-1) as f32 * 0.05).round() as usize],
        "median":values[(values.len()-1)/2],
        "p95":values[((values.len()-1) as f32 * 0.95).round() as usize],
        "max":values.last().unwrap()})
}
pub fn report(plan: &MotionPlan, clip: &MotionClip) -> serde_json::Value {
    let p: Vec<_> = clip.frames.iter().map(|f| f.root_translation).collect();
    let velocities: Vec<_> = p.windows(2).map(|p| (p[1] - p[0]) * clip.fps).collect();
    let angular: Vec<_> = clip
        .frames
        .windows(2)
        .flat_map(|f| {
            f[0].local_rotations
                .iter()
                .zip(&f[1].local_rotations)
                .map(|(a, b)| a.angle_between(*b) * clip.fps)
        })
        .collect();
    let path_errors: Vec<_> = p
        .iter()
        .enumerate()
        .map(|(i, p)| {
            p.with_y(0.0)
                .distance(validation::expected_position(plan, i).with_y(0.0))
        })
        .collect();
    serde_json::json!({"actor_id":plan.actor_id,"seed":plan.request.seed,"prompt":plan.request.prompt,
        "frames":p.len(),"fps":clip.fps,"duration_seconds":(p.len()-1) as f32 / clip.fps,
        "root_endpoint_baseline_m":p[0].with_y(0.0).distance(p.last().unwrap().with_y(0.0)),
        "root_path_length_m":p.windows(2).map(|p|p[0].with_y(0.0).distance(p[1].with_y(0.0))).sum::<f32>(),
        "horizontal_waypoint_error_m":quantiles(path_errors),
        "root_speed_mps":quantiles(velocities.iter().map(|v|v.length()).collect()),
        "root_acceleration_mps2":quantiles(velocities.windows(2).map(|v|(v[1]-v[0]).length()*clip.fps).collect()),
        "local_joint_angular_speed_rad_s":quantiles(angular),
        "policy":"Physical-time diagnostics of admitted raw model clips; not thresholds or perceptual motion quality evidence. Render admission additionally checks retargeted cloth/hair geometry."})
}
