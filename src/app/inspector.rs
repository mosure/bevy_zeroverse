//! Everyday scene controls in one place; the raw ECS inspector is optional.
#[cfg(feature = "human_motion")]
mod motion;
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
        .default_width(340.0)
        .default_height(720.0)
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
                        {
                            changed |= motion::edit(ui, &mut config.human_motion);
                            let frames = world.get_resource::<crate::human_motion::HumanMotionReport>()
                                .and_then(|report| report.accepted.first())
                                .map(|plan| plan.request.frames);
                            if let Some(frames) = frames {
                                if ui.button("Play once at model speed (20 Hz)")
                                    .on_hover_text("Restart the active clip forward at its generated speed. Sin and PingPong reverse motion; trajectory speed can speed it up.")
                                    .clicked() {
                                    let playback = motion::model_playback(frames);
                                    config.playback_mode = playback.mode;
                                    config.playback_speed = playback.speed;
                                    *world.resource_mut::<Playback>() = playback;
                                    changed = true;
                                }
                            }
                        }
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
                            &[Color, Depth, Normal, Position, Semantic, OpticalFlow, CoVisibility],
                        );
                        if config.render_mode.is_flow() {
                            if let Some(mut preview) = world.get_resource_mut::<crate::render::optical_flow::FlowPreviewSettings>() {
                                ui.add(egui::Slider::new(&mut preview.interval_seconds, 0.005..=0.5).logarithmic(true).text("Flow preview interval (s)"));
                                ui.add(egui::Slider::new(&mut preview.full_scale_pixels, 1.0..=256.0).logarithmic(true).text("Flow color scale (pixels)"));
                                ui.small("Viewer: velocity scaled to this interval. Dataset export: exact captured-timestep displacement.");
                            }
                        }
                        if config.render_mode == CoVisibility {
                            if let Some(legend) = world.get_resource::<crate::render::co_visibility::CoVisibilityLegend>() {
                                ui.small("Capture cameras only; colors add across visible cameras. The source camera's bit is excluded.");
                                if let Some(error) = &legend.error { ui.colored_label(egui::Color32::LIGHT_RED, error); }
                                if legend.camera_indices.is_empty() { ui.label("Create capture cameras and enable the camera grid to preview."); }
                                for (bit, index) in legend.camera_indices.iter().enumerate().take(crate::render::co_visibility::MAX_CAMERAS) {
                                    let rgb = crate::render::co_visibility::camera_color(bit, legend.camera_indices.len().min(crate::render::co_visibility::MAX_CAMERAS));
                                    ui.horizontal(|ui| {
                                        let (rect, _) = ui.allocate_exact_size(egui::vec2(20.0, 14.0), egui::Sense::hover());
                                        ui.painter().rect_filled(rect, 1.0, egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
                                        ui.label(format!("Camera {index}: 0x{:04X}  RGB {rgb:?}", 1u16 << bit));
                                    });
                                }
                            }
                        }
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
                            ProceduralIndoor => {
                                changed |= ui.add(egui::Slider::new(&mut config.indoor_gi_rays, 64..=16384).logarithmic(true).text("Indirect lighting ray budget")).changed();
                                ui.small("Applies on Regenerate [R]; native Auto quality uses indirect lighting. Portable/Web builds omit this pass.");
                                changed |= ui.checkbox(&mut config.rotation_augmentation, "Randomize world orientation on regeneration").changed();
                                if let Some(scene) = world.get_resource::<crate::scene::procedural_indoor::layout::IndoorManifest>() {
                                    ui.separator();
                                    ui.label(format!("Current room: {:.1} × {:.1} × {:.1} m; seed {}", scene.room_size.x, scene.room_size.z, scene.room_size.y, scene.seed));
                                    let chairs: Vec<_> = scene.objects.iter().filter(|o| o.kind == crate::scene::procedural_indoor::layout::ObjectKind::Chair).collect();
                                    let stools = chairs.iter().filter(|o| crate::scene::procedural_indoor::objects::chairs::is_backless(o)).count();
                                    ui.label(format!("{} objects, {} people, {} seats ({} backless stools), {} capture cameras", scene.objects.len(), scene.humans.len(), chairs.len(), stools, scene.cameras.len()));
                                    if let Some(program) = &scene.program {
                                        ui.label(format!("{} activity zones, {} interior partitions; fixture spacing {:.1} × {:.1} m", program.zones.len(), program.partitions.len(), program.light_spacing.x, program.light_spacing.y));
                                        if let Some(domain) = &program.domain {
                                            ui.label(format!("Target {:.0} lux; fixtures {:.0} K; sunlight {:.0} lux; exposure EV {:.1}", domain.target_lux, domain.fixture_kelvin, domain.photometry.sun_lux, domain.photometry.ev100));
                                        }
                                    }
                                    ui.small("The values above describe the active scene. Edited generation controls take effect with Regenerate [R].");
                                }
                            }
                            CornellCube => { ui.label("Cornell cube uses fixed reference geometry and lighting."); }
                            Human => { ui.label("Body shape and pose controls are in the human entity's BurnHumanInput component in the world inspector."); }
                            Custom => { ui.label("Custom scenes are configured by their application plugin."); }
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
