//! Everyday scene controls in one place; the raw ECS inspector is optional.
use super::*;
use crate::scene::procedural_indoor::{IndoorGenerationStatus, IndoorQuality};
use bevy_egui::{egui, EguiContext, PrimaryEguiContext};

fn choice<T: PartialEq + Clone + std::fmt::Debug>(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut T,
    values: &[T],
) -> bool {
    let before = value.clone();
    egui::ComboBox::from_label(label)
        .selected_text(format!("{value:?}"))
        .show_ui(ui, |ui| {
            for option in values {
                ui.selectable_value(value, option.clone(), format!("{option:?}"));
            }
        });
    before != *value
}

pub(super) fn panel(world: &mut World) {
    let Ok(context) = world
        .query_filtered::<&mut EguiContext, With<PrimaryEguiContext>>()
        .single(world)
    else {
        return;
    };
    let mut context = context.clone();
    let mut config = world.resource::<BevyZeroverseConfig>().clone();
    let mut changed = false;
    let mut regenerate = false;
    egui::Window::new("Zeroverse")
        .default_width(300.0)
        .show(context.get_mut(), |ui| {
            egui::ScrollArea::vertical()
                .max_height(740.0)
                .show(ui, |ui| {
                    use ZeroverseSceneType::*;
                    changed |= choice(
                        ui,
                        "Scene",
                        &mut config.scene_type,
                        &[
                            Object,
                            ProceduralIndoor,
                            SemanticRoom,
                            Room,
                            CornellCube,
                            Human,
                            Custom,
                        ],
                    );
                    regenerate |= ui.button("Regenerate  [R]").clicked();
                    if let Some(crate::sample::CaptureFailure(Some(error))) =
                        world.get_resource::<crate::sample::CaptureFailure>()
                    {
                        ui.colored_label(egui::Color32::LIGHT_RED, error);
                    }
                    if let Some(status) = world.get_resource::<IndoorGenerationStatus>() {
                        if status.pending {
                            ui.label("Preparing geometry…");
                        } else if status.lighting_pending {
                            ui.label("Refining indirect lighting…");
                        }
                    }
                    if let Some(motion) =
                        world.get_resource::<crate::human_motion::HumanMotionReport>()
                    {
                        if motion.pending {
                            ui.label(&motion.stage);
                        } else if config.human_motion.is_some() {
                            ui.label(format!(
                                "{} moving people; {} requests retained static",
                                motion.accepted.len(),
                                motion.rejected.len()
                            ));
                            if !motion.rejected.is_empty() {
                                ui.collapsing("Motion rejection details", |ui| {
                                    for rejected in &motion.rejected {
                                        ui.label(format!(
                                            "Person {}: {}",
                                            rejected.actor_id, rejected.reason
                                        ));
                                    }
                                });
                            }
                        }
                    }
                    if config.scene_type == ProceduralIndoor {
                        ui.separator();
                        ui.label("Indoor generation — apply with Regenerate");
                        let mut fixed = config.indoor_seed.is_some();
                        if ui
                            .checkbox(&mut fixed, "Reproducible seed sequence")
                            .changed()
                        {
                            config.indoor_seed = fixed.then_some(0);
                            changed = true;
                        }
                        if let Some(seed) = &mut config.indoor_seed {
                            changed |= ui
                                .add(egui::DragValue::new(seed).prefix("Base seed "))
                                .changed();
                        }
                        changed |= choice(
                            ui,
                            "Activity bias",
                            &mut config.indoor_layout,
                            <IndoorLayout as clap::ValueEnum>::value_variants(),
                        );
                        ui.small("Samples a blend of work, meeting, social and learning spaces.");
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut config.indoor_density, 0.0..=1.0)
                                    .text("Furniture density"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut config.indoor_human_density, 0.0..=1.0)
                                    .text("People density"),
                            )
                            .changed();
                        ui.collapsing("Capture camera paths", |ui| {
                            let mut policy = config
                                .indoor_camera
                                .as_deref()
                                .and_then(|j| {
                                    crate::scene::procedural_indoor::cameras::CameraSettings::parse(
                                        j,
                                    )
                                    .ok()
                                })
                                .unwrap_or_default();
                            let mut edit = ui
                                .checkbox(&mut policy.primary_room, "Reconstruct primary room only")
                                .changed();
                            edit |= ui
                                .add(
                                    egui::Slider::new(&mut policy.path_length_min, 0.0..=20.0)
                                        .text("Minimum path (m)"),
                                )
                                .changed();
                            edit |= ui
                                .add(
                                    egui::Slider::new(&mut policy.path_length_max, 0.1..=30.0)
                                        .text("Maximum path (m)"),
                                )
                                .changed();
                            policy.path_length_max =
                                policy.path_length_max.max(policy.path_length_min);
                            edit |= ui
                                .add(
                                    egui::Slider::new(&mut policy.long_path_fraction, 0.0..=1.0)
                                        .text("Long route fraction"),
                                )
                                .changed();
                            let mut overlap_enabled = policy.multiview.is_some();
                            if ui
                                .checkbox(&mut overlap_enabled, "Shared geometry across views")
                                .changed()
                            {
                                policy.multiview = overlap_enabled.then(Default::default);
                                edit = true;
                            }
                            if let Some(m) = &mut policy.multiview {
                                ui.label(
                                    "Each view overlaps camera 0; estimates include occlusion.",
                                );
                                edit |= ui
                                    .add(
                                        egui::Slider::new(&mut m.min_overlap, 0.0..=1.0)
                                            .text("Minimum estimated overlap"),
                                    )
                                    .changed();
                                edit |= ui
                                    .add(
                                        egui::Slider::new(&mut m.min_baseline, 0.01..=5.0)
                                            .text("Minimum baseline (m)"),
                                    )
                                    .changed();
                                edit |= ui
                                    .add(
                                        egui::Slider::new(&mut m.max_baseline, 0.01..=10.0)
                                            .text("Maximum baseline (m)"),
                                    )
                                    .changed();
                                m.max_baseline = m.max_baseline.max(m.min_baseline);
                            }
                            if edit {
                                config.indoor_camera =
                                    Some(serde_json::to_string(&policy).unwrap());
                                changed = true;
                            }
                        });
                        #[cfg(feature = "human_motion")]
                        ui.collapsing("Human motion", |ui| {
                            let mut enabled = config.human_motion.is_some();
                            if ui
                                .checkbox(&mut enabled, "Generate motion for selected people")
                                .changed()
                            {
                                config.human_motion = enabled.then(|| {
                                    serde_json::to_string(
                                        &crate::human_motion::HumanMotionConfig::default(),
                                    )
                                    .unwrap()
                                });
                                changed = true;
                            }
                            if let Some(json) = &mut config.human_motion {
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
                                    {
                                        if policy.validate().is_ok() {
                                            *json = serde_json::to_string(&policy).unwrap();
                                            changed = true;
                                        } else {
                                            ui.label("Keep at least one motion family enabled.");
                                        }
                                    }
                                }
                                ui.label("Motion models download once and use the local cache.");
                                ui.label("Apply motion settings with Regenerate [R].");
                            }
                        });
                        changed |= choice(
                            ui,
                            "Quality",
                            &mut config.indoor_quality,
                            &[IndoorQuality::Auto, IndoorQuality::Portable],
                        );
                    }
                    ui.collapsing("Rendering and annotations", |ui| {
                        use RenderMode::*;
                        changed |= choice(
                            ui,
                            "Render",
                            &mut config.render_mode,
                            &[Color, Depth, Normal, Position, Semantic, OpticalFlow],
                        );
                        changed |= choice(
                            ui,
                            "O-voxel export",
                            &mut config.ovoxel_mode,
                            &[
                                OvoxelMode::Disabled,
                                OvoxelMode::CpuAsync,
                                OvoxelMode::GpuCompute,
                            ],
                        );
                        changed |= ui
                            .checkbox(&mut config.draw_obb_gizmo, "Bounding boxes")
                            .changed();
                        changed |= ui
                            .checkbox(&mut config.draw_pose_gizmos, "Human pose joints")
                            .changed();
                    });
                    ui.collapsing("Cameras and playback", |ui| {
                        changed |= ui
                            .checkbox(&mut config.gizmos, "Show camera gizmos")
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut config.num_cameras, 1..=16)
                                    .text("Capture cameras (regenerate)"),
                            )
                            .changed();
                        changed |= ui
                            .checkbox(&mut config.camera_grid, "Show capture camera grid")
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut config.playback_speed, 0.0..=1.0)
                                    .text("Trajectory speed"),
                            )
                            .changed();
                        changed |= choice(
                            ui,
                            "Playback",
                            &mut config.playback_mode,
                            <PlaybackMode as clap::ValueEnum>::value_variants(),
                        );
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut config.yaw_speed, -1.0..=1.0)
                                    .text("Scene rotation speed"),
                            )
                            .changed();
                    });
                    ui.collapsing("Regeneration", |ui| {
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut config.regenerate_ms)
                                    .range(0..=600000)
                                    .prefix("Interval (ms; 0 = manual) "),
                            )
                            .changed();
                        changed |= ui
                            .checkbox(&mut config.keybinds, "Keyboard shortcuts")
                            .changed();
                    });
                    ui.collapsing("Advanced scene settings", |ui| {
                        use bevy_inspector_egui::bevy_inspector::ui_for_resource;
                        match config.scene_type {
                            Object => ui_for_resource::<
                                crate::scene::object::ZeroverseObjectSceneSettings,
                            >(world, ui),
                            Room => ui_for_resource::<ZeroverseRoomSettings>(world, ui),
                            SemanticRoom => {
                                ui_for_resource::<ZeroverseSemanticRoomSettings>(world, ui)
                            }
                            _ => {
                                ui.label("Generated from the controls above and the seed.");
                            }
                        }
                    });
                    ui.collapsing("Advanced world inspector", |ui| {
                        bevy_inspector_egui::bevy_inspector::ui_for_world(world, ui);
                    });
                });
        });
    if changed {
        *world.resource_mut::<BevyZeroverseConfig>() = config;
    }
    if regenerate {
        world.write_message(RegenerateSceneEvent);
    }
}
