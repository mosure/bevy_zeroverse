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
        if !enabled {
            settings.overlap_mixture = None;
        }
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
        if settings.overlap_mixture.is_none() {
            ui.small(format!(
            "Pair separation ≥ {:.2} m · reference distance {:.2}–{:.2} m · estimated overlap ≥ {:.0}%",
            m.min_baseline,
            m.min_reference_baseline,
            m.max_baseline,
            100.0 * m.min_overlap
        ));
        }
        ui.collapsing("Advanced multi-view constraints", |ui| {
            ui.small("Reference pairs use proxy overlap estimates. An enabled mixture replaces the minimum overlap with its sampled high/low/zero band; rendered visibility is measured separately.");
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
        let mut mixture = settings.overlap_mixture.is_some();
        if ui
            .checkbox(&mut mixture, "Mix high / low / negative pairs")
            .changed()
        {
            settings.overlap_mixture = mixture.then(Default::default);
            edited = true;
        }
        if let Some(mixture) = &mut settings.overlap_mixture {
            for (label, weight) in ["High weight", "Low weight", "Negative weight"]
                .into_iter()
                .zip(&mut mixture.weights)
            {
                edited |= ui
                    .add(egui::Slider::new(weight, 0.0..=1.0).text(label))
                    .changed();
            }
            if mixture.weights.iter().sum::<f32>() == 0. {
                mixture.weights[0] = 1.;
            }
            ui.small("Requested strata are fixed per seed before placement retries. Zero proxy overlap can still have some shared rendered pixels.");
        }
    }
    ui.collapsing("Trajectory length and room bounds", |ui| {
        edited |= ui.checkbox(&mut settings.primary_room, "Keep paths inside primary room").changed();
        edited |= ui.add(egui::Slider::new(&mut settings.path_length_min, 0.0..=20.0).text("Minimum travel (m)")).changed();
        edited |= ui.add(egui::Slider::new(&mut settings.path_length_max, 0.0..=30.0).text("Maximum travel (m)")).changed();
        settings.path_length_max = settings.path_length_max.max(settings.path_length_min);
        edited |= ui.add(egui::Slider::new(&mut settings.long_path_fraction, 0.0..=1.0).text("Long route proposal fraction")).changed();
        ui.small("Travel is the distance each camera moves, independent of spacing between cameras. Set both lengths to 0 for static views. Collision checks always apply.");
    });
    ui.collapsing("Short handheld motion and capture time", |ui| {
        let mut handheld = settings.handheld.is_some();
        if ui.checkbox(&mut handheld,"Independent translation and rotation increments").changed() {
            settings.handheld = handheld.then(Default::default);
            if handheld { settings.path_length_min = 0.01; settings.path_length_max = 0.6; settings.long_path_fraction = 0.; }
            edited = true;
        }
        if let Some(handheld) = &mut settings.handheld {
            for (label, range) in ["Right (m)","Up (m)","Forward (m)"].into_iter().zip(&mut handheld.translation_m) {
                edited |= increment_range(ui,label,range,10.);
            }
            for (label,range) in ["Yaw (degrees)","Pitch (degrees)","Roll (degrees)"].into_iter().zip(&mut handheld.rotation_degrees) {
                edited |= increment_range(ui,label,range,90.);
            }
            edited |= ui.add(egui::Slider::new(&mut handheld.reverse_probability,0.0..=1.0).text("Reverse path probability")).changed();
            ui.small("Negative forward motion strides backward. End orientation is free of the look target; collision and overlap constraints still apply.");
        }
        let mut physical_time = settings.duration_seconds.is_some();
        if ui.checkbox(&mut physical_time,"Assign a capture duration").changed() {
            settings.duration_seconds = physical_time.then_some(2.);
            edited = true;
        }
        if let Some(seconds) = &mut settings.duration_seconds {
            edited |= ui.add(egui::DragValue::new(seconds).range(0.001..=3600.).speed(0.1).suffix(" s")).changed();
        }
        ui.small("Duration labels the normalized timeline in seconds and retimes its paths. Unset means physical time is unspecified.");
    });
    ui.small("Changes apply on Regenerate [R]. Impossible constraints produce an error.");
    if edited {
        *json = Some(serde_json::to_string(&settings).unwrap());
    }
    edited
}

fn increment_range(ui: &mut egui::Ui, label: &str, range: &mut [f32; 2], bound: f32) -> bool {
    let mut edited = false;
    ui.horizontal(|ui| {
        ui.label(label);
        edited |= ui
            .add(
                egui::DragValue::new(&mut range[0])
                    .range(-bound..=bound)
                    .speed(0.01),
            )
            .changed();
        ui.label("to");
        edited |= ui
            .add(
                egui::DragValue::new(&mut range[1])
                    .range(-bound..=bound)
                    .speed(0.01),
            )
            .changed();
    });
    range[1] = range[1].max(range[0]);
    edited
}
