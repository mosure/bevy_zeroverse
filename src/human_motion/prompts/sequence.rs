//! Preserve the planned path while inserting one or two timed, spatially checked
//! action stops. Remaining travel time determines the gait and verbal pace.
use super::*;
use crate::human_motion::planning::segment_clear;
use bevy::prelude::Vec3;

fn at(points: &[Waypoint], t: f32) -> Waypoint {
    let frame = t * points.last().unwrap().frame as f32;
    let pair = points
        .windows(2)
        .find(|p| p[1].frame as f32 >= frame)
        .unwrap();
    let alpha = (frame - pair[0].frame as f32) / (pair[1].frame - pair[0].frame) as f32;
    Waypoint {
        frame: frame.round() as usize,
        position: pair[0].position.lerp(pair[1].position, alpha),
        heading: pair[0].heading.or(pair[1].heading),
        constrain_height: true,
    }
}

pub(in crate::human_motion) fn walking_sequence(
    points: &mut Vec<Waypoint>,
    config: &HumanMotionConfig,
    stature: f32,
    headroom: f32,
    obstacles: &[(Vec3, Vec3)],
    rng: &mut impl Rng,
) -> Option<Draft> {
    if config.frames < 80 || !rng.random_bool(config.sequence_fraction as f64) {
        return None;
    }
    let distance: f32 = points
        .windows(2)
        .map(|p| {
            p[0].position
                .with_y(0.0)
                .distance(p[1].position.with_y(0.0))
        })
        .sum();
    let last = config.frames - 1;
    let max_pause = (last as f32 - distance / 1.25 * 20.0)
        .min(last as f32 * 0.50)
        .floor()
        .max(0.0) as usize;
    let count = if config.prompt_sampling.max_sequence_actions >= 2
        && max_pause >= 48
        && rng.random_bool(0.5)
    {
        2
    } else {
        1
    };
    for count in (1..=count).rev() {
        let mut actions: Vec<(f32, SampledAction)> = Vec::new();
        for i in 0..count {
            // Keep pauses away from route corners so integer frame rounding
            // cannot overwrite a necessary navigation vertex with an action key.
            let fraction = (0..16)
                .map(|_| {
                    if count == 1 {
                        rng.random_range(0.28..0.72)
                    } else if i == 0 {
                        rng.random_range(0.22..0.40)
                    } else {
                        rng.random_range(0.60..0.78)
                    }
                })
                .find(|t: &f32| {
                    points.iter().all(|w| {
                        (*t - w.frame as f32 / last as f32).abs() * (last - max_pause) as f32 > 3.0
                    })
                });
            let Some(fraction) = fraction else {
                break;
            };
            let anchor = at(points, fraction);
            for _ in 0..12 {
                let Some(action) = sample_action(
                    &config.prompt_sampling,
                    None,
                    max_pause / count,
                    headroom,
                    false,
                    rng,
                ) else {
                    break;
                };
                if actions.iter().any(|(_, a)| a.spec.id == action.spec.id) {
                    continue;
                }
                if segment_clear(
                    anchor.position,
                    anchor.position,
                    stature + action.spec.headroom + 0.05,
                    action.phase.clearance_radius,
                    obstacles,
                ) {
                    actions.push((fraction, action));
                    break;
                }
            }
        }
        if actions.len() != count {
            continue;
        }
        let pause: usize = actions.iter().map(|(_, a)| a.frames).sum();
        let travel = last - pause;
        let speed = distance / (travel as f32 / 20.0);
        let gait = gait(speed, headroom, config.energetic_fraction, rng);
        let phrases: Vec<_> = actions.iter().map(|(_, a)| a.phrase.as_str()).collect();
        let stops = if phrases.len() == 2 {
            // These actions occur at different positions, so the text must
            // describe the intervening travel as well as the two pauses.
            format!(
                "{}, {} again, stops and {}",
                phrases[0], gait.verb, phrases[1]
            )
        } else {
            phrases[0].to_owned()
        };
        let core = format!(
            "A person {} {}, stops and {}, then resumes {}",
            gait.verb,
            pace(speed),
            stops,
            gait.noun
        );
        // Reduce action count before conditioning if the essential clauses would
        // exceed the bounded encoder grammar. Never truncate an action's text.
        if !fits(&format!("{core}.")) {
            continue;
        }
        let mut base = points.clone();
        condition_gait(&mut base, gait);
        let mut timed = std::collections::BTreeMap::new();
        for mut w in base.iter().cloned() {
            let t = w.frame as f32 / last as f32;
            let elapsed: usize = actions
                .iter()
                .filter(|(fraction, _)| *fraction < t)
                .map(|(_, a)| a.frames)
                .sum();
            w.frame = (t * travel as f32).round() as usize + elapsed;
            timed.insert(w.frame, w);
        }
        let mut recipe = recipe(Family::Locomotion);
        recipe.gait = Some(gait.id.into());
        recipe.travel_speed_mps = Some(speed);
        let mut elapsed = 0;
        for (fraction, mut action) in actions {
            let anchor = at(&base, fraction);
            let start = (fraction * travel as f32).round() as usize + elapsed;
            let end = start + action.frames;
            for w in action_keys(&action, &anchor, start, end, false) {
                timed.insert(w.frame, w);
            }
            action.phase.start_frame = start;
            action.phase.end_frame = end;
            elapsed += action.frames;
            recipe.actions.push(action.phase);
        }
        let behavior = format!(
            "{}_{}_{}",
            gait.id,
            recipe
                .actions
                .iter()
                .map(|a| a.action.as_str())
                .collect::<Vec<_>>()
                .join("_"),
            gait.id
        );
        let text = finish(core, &mut recipe, &config.prompt_sampling, false, rng);
        *points = timed.into_values().collect();
        return Some(Draft {
            behavior,
            text,
            recipe,
            radius: 0.40,
            headroom: gait.headroom,
        });
    }
    None
}
