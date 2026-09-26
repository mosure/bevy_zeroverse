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
                    if let Some(status) = world.get_resource::<IndoorGenerationStatus>() {
                        if status.pending {
                            ui.label("Preparing geometry…");
                        } else if status.lighting_pending {
                            ui.label("Refining indirect lighting…");
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
                            "Activity",
                            &mut config.indoor_layout,
                            &[
                                IndoorLayout::Mixed,
                                IndoorLayout::Conference,
                                IndoorLayout::OpenOffice,
                                IndoorLayout::Training,
                                IndoorLayout::Lounge,
                            ],
                        );
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
                    });
                    ui.collapsing("Cameras and playback", |ui| {
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
                            &[
                                PlaybackMode::Still,
                                PlaybackMode::Loop,
                                PlaybackMode::PingPong,
                            ],
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
