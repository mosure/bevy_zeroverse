use super::*;
use crate::scene::procedural_indoor::layout::stream;
use bevy::prelude::Vec3;
use std::collections::BTreeSet;

fn check_token_budget(texts: &BTreeSet<String>) {
    #[cfg(feature = "human_motion")]
    if let Some(path) = std::env::var_os("ZEROVERSE_PROMPT_TEST_TOKENIZER") {
        let tokenizer =
            burn_llama::tokenizer::PromptTokenizer::from_bytes(&std::fs::read(path).unwrap())
                .unwrap();
        let max_tokens = texts
            .iter()
            .map(|text| {
                tokenizer
                    .encode(text)
                    .unwrap_or_else(|e| panic!("{e}: {text}"))
                    .attention
                    .into_iter()
                    .filter(|&v| v)
                    .count()
            })
            .max()
            .unwrap();
        println!("validated {} unique prompts with pinned tokenizer; max {max_tokens}/64 tokens including header", texts.len());
    }
    #[cfg(not(feature = "human_motion"))]
    let _ = texts;
}

fn path(frames: usize) -> Vec<Waypoint> {
    vec![
        Waypoint {
            frame: 0,
            position: Vec3::ZERO,
            heading: Some(0.0),
            constrain_height: false,
        },
        Waypoint {
            frame: frames / 2,
            position: Vec3::Z,
            heading: Some(std::f32::consts::FRAC_PI_2),
            constrain_height: false,
        },
        Waypoint {
            frame: frames - 1,
            position: Vec3::Z + Vec3::X,
            heading: Some(std::f32::consts::FRAC_PI_2),
            constrain_height: false,
        },
    ]
}

#[test]
fn grammar_spans_actions_and_gaits_with_bounded_coherent_text() {
    let config = HumanMotionConfig::default();
    let mut texts = BTreeSet::new();
    let mut actions = BTreeSet::new();
    let mut gaits = BTreeSet::new();
    for seed in 0..4096 {
        let mut rng = stream(seed, 712);
        let sample =
            sample_action(&config.prompt_sampling, None, 159, 1.0, false, &mut rng).unwrap();
        let mut points = path(config.frames);
        let draft = stationary(sample, &mut points, &config, false, &mut rng);
        assert!(fits(&draft.text));
        assert!(!draft.text.contains(['{', '}']));
        assert!(!draft.text.contains("  "));
        assert!(!draft.text.contains("one jumping jacks"));
        actions.insert(draft.recipe.actions[0].action.clone());
        texts.insert(draft.text);
        let mut points = path(config.frames);
        let scale = 0.8 + (seed % 8) as f32 * 0.6;
        for w in &mut points {
            w.position *= scale;
        }
        let draft = locomotion("walk", &mut points, &config, 1.0, &mut rng);
        gaits.insert(draft.recipe.gait.unwrap());
        assert!(fits(&draft.text));
        texts.insert(draft.text);
    }
    assert_eq!(actions.len(), catalog::ACTIONS.len());
    assert_eq!(gaits.len(), GAITS.len());
    assert!(
        texts.len() > 1800,
        "insufficient compositional text diversity: {}",
        texts.len()
    );
    check_token_budget(&texts);
}

#[test]
fn sequences_preserve_routes_and_bind_actions_to_ordered_waypoints() {
    let config = HumanMotionConfig {
        frames: 240,
        sequence_fraction: 1.0,
        ..Default::default()
    };
    let mut pairs = BTreeSet::new();
    let mut texts = BTreeSet::new();
    for seed in 0..512 {
        let mut points = path(config.frames);
        let before = points.clone();
        let mut rng = stream(seed, 33);
        let draft = walking_sequence(&mut points, &config, 1.8, 1.0, &[], &mut rng).unwrap();
        assert!(fits(&draft.text));
        texts.insert(draft.text.clone());
        assert_eq!(points.first().unwrap().frame, 0);
        assert_eq!(points.last().unwrap().frame, 239);
        assert!(points.windows(2).all(|p| p[0].frame < p[1].frame));
        assert!(draft
            .recipe
            .actions
            .windows(2)
            .all(|a| a[0].end_frame < a[1].start_frame));
        for original in &before {
            assert!(points
                .iter()
                .any(|p| p.position.with_y(0.0).distance(original.position) < 1e-6));
        }
        for a in &draft.recipe.actions {
            let keys: Vec<_> = points
                .iter()
                .filter(|w| w.frame >= a.start_frame && w.frame <= a.end_frame)
                .collect();
            assert_eq!(keys.len(), a.repetitions * 2 + 1);
            assert!(keys.iter().all(|w| w
                .position
                .with_y(0.0)
                .distance(keys[0].position.with_y(0.0))
                < 1e-6));
            assert!(keys
                .iter()
                .any(|w| (w.position.y - a.peak_pelvis_height).abs() < 1e-6));
        }
        config
            .request(draft.text.clone(), seed, points.clone())
            .validate()
            .unwrap();
        pairs.insert(
            draft
                .recipe
                .actions
                .iter()
                .map(|a| a.action.as_str())
                .collect::<Vec<_>>()
                .join("/"),
        );
        let mut replay = before;
        let again =
            walking_sequence(&mut replay, &config, 1.8, 1.0, &[], &mut stream(seed, 33)).unwrap();
        assert_eq!(draft.text, again.text);
        assert_eq!(
            serde_json::to_string(&points).unwrap(),
            serde_json::to_string(&replay).unwrap()
        );
    }
    assert!(pairs.len() > 80, "sequence action coverage {}", pairs.len());
    check_token_budget(&texts);
}

#[test]
fn weights_space_support_and_duration_filter_actions() {
    let only_floor = PromptSamplingConfig {
        locomotion: 0.0,
        gesture: 0.0,
        exercise: 0.0,
        dance: 0.0,
        floor: 1.0,
        idle: 0.0,
        ..Default::default()
    };
    let config = HumanMotionConfig {
        prompt_sampling: only_floor.clone(),
        ..Default::default()
    };
    for seed in 0..128 {
        let mut rng = stream(seed, 7);
        assert_eq!(only_floor.sample_family(&mut rng), Family::Floor);
        let a = sample_action(&only_floor, None, 159, 0.0, false, &mut rng).unwrap();
        assert_eq!(a.spec.family, Family::Floor);
        assert!(sample_action(&only_floor, None, 159, 1.0, true, &mut rng).is_none());
        assert!(sample_action(&only_floor, None, 20, 1.0, false, &mut rng).is_none());
        let mut points = path(160);
        let draft = walking_sequence(
            &mut points,
            &config,
            1.8,
            0.0,
            &[(Vec3::splat(-20.0), Vec3::splat(20.0))],
            &mut rng,
        );
        assert!(
            draft.is_none(),
            "an action was placed inside a solid obstacle"
        );
        let a = sample_action(
            &PromptSamplingConfig::default(),
            None,
            80,
            0.0,
            true,
            &mut rng,
        )
        .unwrap();
        assert!(a.spec.seated && a.spec.headroom == 0.0);
    }
    for invalid in [
        r#"{"prompt_sampling":{"gesture":-1}}"#,
        r#"{"prompt_sampling":{"max_sequence_actions":3}}"#,
        r#"{"prompt_sampling":{"locomotion":0,"gesture":0,"exercise":0,"dance":0,"floor":0,"idle":0}}"#,
    ] {
        assert!(HumanMotionConfig::parse(invalid).is_err());
    }
}

#[test]
fn backward_and_sideways_gaits_condition_facing_without_reversing_the_route() {
    for g in GAITS.iter().filter(|g| g.heading_offset != 0.0) {
        let mut points = path(160);
        let before = points.clone();
        condition_gait(&mut points, g);
        for (a, b) in before.iter().zip(&points) {
            assert_eq!(a.position.with_y(0.0), b.position.with_y(0.0));
            assert!((b.heading.unwrap() - a.heading.unwrap() - g.heading_offset).abs() < 1e-6);
        }
    }
}
