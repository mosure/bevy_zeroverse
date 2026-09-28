//! Deferred, validated motion policy editing; never load models from a UI pass.
use bevy_egui::egui;

pub(super) fn model_playback(frames: usize) -> crate::camera::Playback {
    crate::camera::Playback {
        mode: crate::camera::PlaybackMode::Once,
        progress: 0.0,
        direction: 1.0,
        speed: 20.0 / frames.saturating_sub(1).max(1) as f32,
    }
}

pub(super) fn edit(ui: &mut egui::Ui, motion: &mut Option<String>) -> bool {
    let mut changed = false;
    ui.collapsing("Human motion", |ui| {
        let mut enabled = motion.is_some();
        if ui
            .checkbox(&mut enabled, "Generate motion for selected people")
            .changed()
        {
            *motion = enabled.then(|| {
                serde_json::to_string(
                    &crate::human_motion::HumanMotionConfig::default(),
                )
                .unwrap()
            });
            changed = true;
        }
        if let Some(json) = motion {
            if let Ok(mut policy) =
                crate::human_motion::HumanMotionConfig::parse(json)
            {
                let fraction = ui
                    .add(
                        egui::Slider::new(&mut policy.fraction, 0.0..=1.0)
                            .text("Moving fraction"),
                    )
                    .changed();
                let count = ui
                    .add(
                        egui::Slider::new(&mut policy.max_actors, 1..=16)
                            .text("Maximum moving people"),
                    )
                    .changed();
                let locomotion = ui
                    .add(
                        egui::Slider::new(
                            &mut policy.locomotion_fraction,
                            0.0..=1.0,
                        )
                        .text("Navigate around room"),
                    )
                    .changed();
                let sequences = ui
                    .add(
                        egui::Slider::new(
                            &mut policy.sequence_fraction,
                            0.0..=1.0,
                        )
                        .text("Walk / action / resume sequences"),
                    )
                    .changed();
                let energetic = ui
                    .add(
                        egui::Slider::new(
                            &mut policy.energetic_fraction,
                            0.0..=1.0,
                        )
                        .text("Skipping / jogging proposals"),
                    )
                    .changed();
                let mut inference_changed = false;
                ui.collapsing("ARDY generation parameters", |ui| {
                    ui.small("Settings apply on Regenerate [R]. Higher guidance does not guarantee better motion.");
                    for (name, value, range) in [
                        ("Diffusion steps", &mut policy.diffusion_steps, 1..=10),
                        ("Batch size", &mut policy.batch_size, 1..=8),
                        ("Attempts per person", &mut policy.max_attempts, 1..=3),
                    ] {
                        inference_changed |= ui.add(egui::Slider::new(value, range).text(name)).changed();
                    }
                    inference_changed |= ui.add(egui::Slider::new(&mut policy.frames, 40..=640).step_by(4.0).text("Clip frames (20 Hz)")).changed();
                    inference_changed |= ui.add(egui::Slider::new(&mut policy.history_frames, 0..=160).step_by(4.0).text("History frames")).changed();
                    ui.small(format!("Generated duration: {:.2} seconds. Playback speed sets traversal time independently.", (policy.frames-1) as f32 / 20.0));
                    for (name, value) in [
                        ("Text guidance", &mut policy.text_guidance),
                        ("Trajectory guidance", &mut policy.trajectory_guidance),
                    ] {
                        inference_changed |= ui.add(egui::Slider::new(value, 0.0..=10.0).text(name)).changed();
                    }
                    inference_changed |= ui.checkbox(&mut policy.dense_trajectory, "Constrain the complete navigation path").changed();
                    inference_changed |= ui.checkbox(&mut policy.strict, "Fail dataset capture if requested motion is rejected").changed();
                });
                let mut prompt_changed = false;
                ui.collapsing("Motion prompt sampling", |ui| {
                    let sampling = &mut policy.prompt_sampling;
                    ui.label(
                        "Relative proposal weights; zero excludes a family.",
                    );
                    for (name, weight) in [
                        (
                            "Travel and chair transitions",
                            &mut sampling.locomotion,
                        ),
                        ("Hand gestures", &mut sampling.gesture),
                        ("Exercise", &mut sampling.exercise),
                        ("Dance", &mut sampling.dance),
                        ("Floor actions", &mut sampling.floor),
                        ("Idle and looking", &mut sampling.idle),
                    ] {
                        prompt_changed |= ui
                            .add(
                                egui::Slider::new(weight, 0.0..=10.0)
                                    .text(name),
                            )
                            .changed();
                    }
                    prompt_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut sampling.style_fraction,
                                0.0..=1.0,
                            )
                            .text("Posture / gaze variation"),
                        )
                        .changed();
                    prompt_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut sampling.max_sequence_actions,
                                1..=2,
                            )
                            .text("Maximum action stops"),
                        )
                        .changed();
                });
                if fraction
                    || count
                    || locomotion
                    || sequences
                    || energetic
                    || prompt_changed
                    || inference_changed
                {
                    if policy.validate().is_ok() {
                        *json = serde_json::to_string(&policy).unwrap();
                        changed = true;
                    } else {
                        ui.label("Invalid settings: enable a motion family and use frame counts divisible by four.");
                    }
                }
            }
            ui.label("Motion models download once and use the local cache.");
            ui.label("Apply motion settings with Regenerate [R].");
        }
    });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_speed_advances_one_motion_frame_per_twentieth_second() {
        for frames in [40, 120, 160, 640] {
            let mut playback = model_playback(frames);
            let mut time = bevy::prelude::Time::default();
            time.advance_by(std::time::Duration::from_secs_f64(1.0 / 20.0));
            playback.step(&time);
            assert!((playback.progress * (frames - 1) as f32 - 1.0).abs() < 1e-6);
            assert_eq!(playback.mode, crate::camera::PlaybackMode::Once);
        }
    }
}
