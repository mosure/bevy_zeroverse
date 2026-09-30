//! Conservative geometric admission checks. Guidance is never a collision guarantee.
use super::MotionPlan;
use crate::scene::procedural_indoor::layout::IndoorManifest;
use bevy::prelude::*;

pub fn bounds_overlap(a: (Vec3, Vec3), b: (Vec3, Vec3)) -> bool {
    a.0.cmplt(b.1).all() && a.1.cmpgt(b.0).all()
}

pub fn point_inside(p: Vec3, bounds: (Vec3, Vec3), margin: f32) -> bool {
    p.cmpgt(bounds.0 - Vec3::splat(margin)).all() && p.cmplt(bounds.1 + Vec3::splat(margin)).all()
}

pub fn path_distance(p: Vec3, plan: &MotionPlan) -> f32 {
    plan.request
        .waypoints
        .windows(2)
        .map(|w| {
            let a = w[0].position.with_y(0.0);
            let b = w[1].position.with_y(0.0);
            let d = b - a;
            p.with_y(0.0).distance(
                a + d * ((p.with_y(0.0) - a).dot(d) / d.length_squared().max(1e-9)).clamp(0.0, 1.0),
            )
        })
        .fold(f32::INFINITY, f32::min)
}

pub fn expected_position(plan: &MotionPlan, frame: usize) -> Vec3 {
    let points = &plan.request.waypoints;
    for pair in points.windows(2) {
        if frame <= pair[1].frame {
            let t = (frame.saturating_sub(pair[0].frame) as f32
                / (pair[1].frame - pair[0].frame) as f32)
                .clamp(0.0, 1.0);
            return pair[0].position.lerp(pair[1].position, t);
        }
    }
    points.last().unwrap().position
}

pub fn validate_behavior(
    plan: &MotionPlan,
    clip: &burn_human_motion::MotionClip,
) -> Result<(), String> {
    let a = clip.frames.first().unwrap().root_translation;
    let b = clip.frames.last().unwrap().root_translation;
    match plan.behavior.as_str() {
        "sit" if a.y - b.y < 0.18 => Err("sit prompt did not produce a sitting transition".into()),
        "stand" if b.y - a.y < 0.18 => {
            Err("stand prompt did not produce a standing transition".into())
        }
        _ if (matches!(plan.behavior.as_str(), "enter" | "leave" | "walk")
            || plan
                .prompt_recipe
                .as_ref()
                .is_some_and(|r| r.gait.is_some()))
            && a.with_y(0.0).distance(b.with_y(0.0)) < 0.40 =>
        {
            Err("locomotion prompt produced insufficient displacement".into())
        }
        _ => Ok(()),
    }
}

pub fn validate_positions(
    scene: &IndoorManifest,
    positions: &[Vec3],
    bounds: (Vec3, Vec3),
    obstacles: &[(Vec3, Vec3)],
) -> Result<(), String> {
    if positions.iter().any(|p| !p.is_finite()) {
        return Err("non-finite deformed body".into());
    }
    if bounds.0.y < -0.025 || bounds.1.y > scene.room_size.y - 0.03 {
        return Err(format!(
            "body intersects floor or ceiling (y {:.3}..{:.3})",
            bounds.0.y, bounds.1.y
        ));
    }
    if let Some(e) = &scene.envelope {
        if positions.iter().any(|p| {
            p.z < scene.room_size.z * 0.5 - 0.02
                && (!crate::scene::procedural_indoor::envelope::polygon::contains(
                    &e.footprint,
                    p.xz(),
                    0.005,
                ) || p.y < e.floor_height(p.xz()) - 0.025
                    || p.y > scene.ceiling_height(p.xz()) - 0.03)
        }) {
            return Err("deformed body intersects architectural envelope".into());
        }
    }
    for &obstacle in obstacles {
        if bounds_overlap(bounds, obstacle)
            && positions.iter().any(|&p| point_inside(p, obstacle, 0.008))
        {
            return Err(format!(
                "deformed body intersects obstacle {:?}..{:?}",
                obstacle.0.to_array(),
                obstacle.1.to_array()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::human_motion::{planning, HumanMotionConfig};
    use crate::scene::procedural_indoor::layout::IndoorLayout;
    #[test]
    fn swept_routes_respect_glass_furniture_and_doorway() {
        let scene =
            IndoorManifest::generate_with_humans(34, IndoorLayout::Mixed, 0.5, 0, 0.0).unwrap();
        let boxes = planning::obstacles(&scene, usize::MAX, None);
        let z = scene.room_size.z * 0.5;
        let a = Vec3::new(scene.door_x, 0.0, z - 0.8);
        let b = a.with_z(z + 0.8);
        assert!(planning::segment_clear(a, b, 1.8, 0.34, &boxes));
        assert!(!planning::segment_clear(
            Vec3::new(0.0, 0.0, z - 0.8),
            Vec3::new(0.0, 0.0, z + 0.8),
            1.8,
            0.34,
            &boxes
        ));
        let block = (Vec3::new(-0.3, 0.0, -0.3), Vec3::new(0.3, 1.0, 0.3));
        let path = planning::route(
            &scene,
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.8,
            0.34,
            &[block],
        )
        .unwrap();
        assert!(path.len() > 2);
        assert!(path
            .windows(2)
            .all(|p| planning::segment_clear(p[0], p[1], 1.8, 0.34, &[block])));
    }
    #[test]
    fn policies_are_bounded_and_reproducible() {
        for bad in [
            r#"{"fraction":-1}"#,
            r#"{"batch_size":9}"#,
            r#"{"frames":41}"#,
            r#"{"unknown":1}"#,
        ] {
            assert!(HumanMotionConfig::parse(bad).is_err());
        }
        let config = HumanMotionConfig {
            fraction: 1.0,
            ..Default::default()
        };
        let mut prompts = std::collections::HashSet::new();
        let (mut plans_total, mut travel_total) = (0, 0);
        for seed in 0..128 {
            let mut scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.4, 0, 0.7)
                    .unwrap();
            planning::prepare_scene(&mut scene, &config).unwrap();
            let (a, _) = planning::plan(&scene, &config).unwrap();
            let (b, _) = planning::plan(&scene, &config).unwrap();
            assert_eq!(
                serde_json::to_string(&a).unwrap(),
                serde_json::to_string(&b).unwrap()
            );
            assert!(a.len() <= config.max_actors);
            for plan in a {
                plan.request.validate().unwrap();
                plans_total += 1;
                travel_total += usize::from(
                    plan.prompt_recipe
                        .as_ref()
                        .is_some_and(|r| r.gait.is_some()),
                );
                prompts.insert(plan.request.prompt);
            }
        }
        assert!(
            prompts.len() > 30,
            "insufficient prompt variation: {}",
            prompts.len()
        );
        assert!(travel_total * 4 > plans_total, "geometry retries collapsed weighted travel into stationary gestures: {travel_total}/{plans_total}");
    }

    #[test]
    fn explicit_prompts_preserve_contact_furniture_and_reject_invalid_ids() {
        let scene =
            IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.35, 2, 0.7).unwrap();
        let mut config = HumanMotionConfig {
            fraction: 1.0,
            frames: 120,
            ..Default::default()
        };
        let automatic = planning::plan(&scene, &config)
            .unwrap()
            .0
            .into_iter()
            .find(|p| p.support_chair.is_some())
            .unwrap();
        config.fraction = 0.0;
        config.trajectories = vec![crate::human_motion::HumanTrajectory {
            actor_id: automatic.actor_id,
            prompt: automatic.request.prompt,
            waypoints: automatic.request.waypoints,
            support_chair: automatic.support_chair,
        }];
        let (explicit, rejected) = planning::plan(&scene, &config).unwrap();
        assert!(rejected.is_empty());
        assert_eq!(explicit[0].support_chair, automatic.support_chair);
        config.trajectories[0].support_chair = Some(usize::MAX);
        assert!(planning::plan(&scene, &config).is_err());
    }
}
