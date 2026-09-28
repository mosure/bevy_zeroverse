//! Camera controls: one baseline program, with explicit advanced overrides.
use crate::scene::procedural_indoor::cameras::{multiview::MultiViewSettings, CameraSettings};
use bevy_egui::egui;

pub(super) fn edit(ui: &mut egui::Ui, json: &mut Option<String>) -> bool {
    let mut settings = json
        .as_deref()
        .and_then(|j| CameraSettings::parse(j).ok())
        .unwrap_or_default();
    let mut enabled = settings.multiview.is_some();
    let mut edited = false;
    if ui
        .checkbox(&mut enabled, "Coordinate views for reconstruction")
        .changed()
    {
        settings.multiview = enabled.then(Default::default);
        edited = true;
    }
    if let Some(m) = &mut settings.multiview {
        let preset = m.baseline();
        let mut value =
            preset.unwrap_or(((m.min_reference_baseline - 0.06) / 2.30).clamp(0.0, 1.0));
        if ui.add(egui::Slider::new(&mut value, 0.0..=1.0).text("Camera baseline"))
            .on_hover_text("0: very close views with high overlap. 1: widely separated views with less overlap. Updates spacing, overlap, spread and path variation. Apply with Regenerate [R].")
            .changed() {
            *m = MultiViewSettings::from_baseline(value).unwrap();
            edited = true;
        }
        ui.small(if preset.is_some() {
            "0 · very narrow    →    1 · very wide"
        } else {
            "Custom advanced settings. Moving baseline replaces them."
        });
        ui.small(format!(
            "Pair separation ≥ {:.2} m · reference distance {:.2}–{:.2} m · estimated overlap ≥ {:.0}%",
            m.min_baseline,
            m.min_reference_baseline,
            m.max_baseline,
            100.0 * m.min_overlap
        ));
        ui.collapsing("Advanced multi-view constraints", |ui| {
            ui.small("Every view shares geometry with camera 0. Overlap is a proxy estimate, not a rendered-pixel guarantee.");
            for (label, field, range) in [
                ("Minimum estimated overlap", &mut m.min_overlap, 0.0..=1.0),
                ("Minimum pair separation (m)", &mut m.min_baseline, 0.01..=10.0),
                ("Minimum reference distance (m)", &mut m.min_reference_baseline, 0.0..=20.0),
                ("Maximum reference distance (m)", &mut m.max_baseline, 0.01..=30.0),
                ("Group spread (3+ views)", &mut m.min_spread, 0.0..=1.0),
                ("Independent path variation", &mut m.trajectory_variation, 0.0..=1.0),
            ] {
                edited |= ui.add(egui::Slider::new(field, range).text(label)).changed();
            }
            m.max_baseline = m.max_baseline.max(m.min_baseline).max(m.min_reference_baseline);
            ui.small("Spread: 0 allows a line, 1 requires equal horizontal extent in both directions. Path variation: 0 allows a rigid camera rig.");
        });
    }
    ui.collapsing("Trajectory length and room bounds", |ui| {
        edited |= ui.checkbox(&mut settings.primary_room, "Keep paths inside primary room").changed();
        edited |= ui.add(egui::Slider::new(&mut settings.path_length_min, 0.0..=20.0).text("Minimum travel (m)")).changed();
        edited |= ui.add(egui::Slider::new(&mut settings.path_length_max, 0.0..=30.0).text("Maximum travel (m)")).changed();
        settings.path_length_max = settings.path_length_max.max(settings.path_length_min);
        edited |= ui.add(egui::Slider::new(&mut settings.long_path_fraction, 0.0..=1.0).text("Long route proposal fraction")).changed();
        ui.small("Travel is the distance each camera moves, independent of spacing between cameras. Set both lengths to 0 for static views. Collision checks always apply.");
    });
    ui.small("Changes apply on Regenerate [R]. Impossible constraints produce an error.");
    if edited {
        *json = Some(serde_json::to_string(&settings).unwrap());
    }
    edited
}
