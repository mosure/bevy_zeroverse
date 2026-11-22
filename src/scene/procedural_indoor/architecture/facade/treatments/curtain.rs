//! Sewn curtain geometry, sampled in cloth coordinates so the weave follows a
//! fold through its whole travel. Column trigonometry is shared by all rows.
use super::*;
use drapery::{CurtainHeading, Panel, TRACK_HEIGHT, TRACK_Z};

struct Column {
    head: f32,
    hem: f32,
    head_wave: f32,
    hem_wave: f32,
    top_sag: f32,
    hem_lift: f32,
}

impl Column {
    fn cut(panel: &Panel, u: f32) -> Self {
        let phase = u * panel.folds as f32 * TAU;
        let drift = ((u * TAU + panel.phase).sin() - panel.phase.sin()) / TAU;
        let secondary = ((u * 2. * TAU - panel.phase).sin() + panel.phase.sin()) / (2. * TAU);
        let (gather, spacing, carrier_count) = match panel.heading {
            CurtainHeading::Wave => (0.00, 0.055, 2.),
            CurtainHeading::PinchPleat => (0.72, 0.18, 1.),
            CurtainHeading::Gathered => (0.38, 0.32, 1.),
        };
        let shift = panel.flutter * (u * PI).sin() * (u * TAU + panel.phase).sin();
        let loose_edge = if panel.left { u } else { 1. - u };
        Self {
            head: (u + drift * spacing).clamp(0., 1.),
            hem: (u + drift * panel.arc_bias + secondary * 0.28).clamp(0., 1.),
            head_wave: (phase + gather * phase.sin()).sin(),
            hem_wave: (phase + shift + 0.10 * phase.sin()).sin(),
            top_sag: (phase * carrier_count * 0.5).sin().powi(2)
                * (panel.span / panel.folds as f32 * 0.045)
                    .min(0.015)
                    .min(panel.height * 0.05),
            hem_lift: panel.hem_lift
                * (0.35 * (0.5 + 0.5 * (phase + panel.phase).cos()) + 0.65 * loose_edge.powi(2)),
        }
    }
}

impl Builder<'_> {
    pub(super) fn drapery(&mut self, heading: CurtainHeading, o: &WindowOpening, moving: bool) {
        let held = !moving && drapery::tied(self.p, o, self.coverage);
        for panel in drapery::panels(self.p, o, heading, self.coverage, held) {
            let nx = panel.folds * 6;
            let levels = panel.levels();
            let cols: Vec<_> = (0..=nx)
                .map(|i| Column::cut(&panel, i as f32 / nx as f32))
                .collect();
            let mut points = Vec::with_capacity(cols.len() * levels.len());
            for &v in &levels {
                let profile = panel.profile_height(v);
                let width = panel.row_width(profile);
                let x0 = if panel.left {
                    panel.jamb
                } else {
                    panel.jamb - width
                };
                let hang = profile * profile * (3. - 2. * profile);
                let pull = panel.gather(profile);
                let bow = 0.006 * panel.flutter * 4. * v * (1. - v) * (1. - pull);
                let pitch = width / panel.folds as f32;
                let cut = panel.span * panel.fullness / panel.folds as f32;
                // Fold depth increases as the carriers draw together. A small
                // reveal limits the represented depth of broad decorative folds.
                let amplitude = (0.25 * (cut * cut - pitch * pitch).max(0.).sqrt())
                    .clamp(0.008, (0.040 + panel.fold_depth).min(0.075))
                    * (1. - 0.66 * pull);
                for c in &cols {
                    let scallop = if v == 0. {
                        c.hem_lift
                    } else if v == 1. {
                        -c.top_sag
                    } else {
                        0.
                    };
                    points.push(Vec3::new(
                        x0 + c.hem.lerp(c.head, hang) * width,
                        panel.top - (1. - v) * panel.height + scallop,
                        TRACK_Z + amplitude * c.hem_wave.lerp(c.head_wave, hang) + bow,
                    ));
                }
            }

            cloth::smooth_sheet(
                self.a.part(self.p.fabric, &self.cloth_label),
                self.tf,
                nx,
                &levels,
                Rect::from_corners(
                    panel.cut_origin,
                    panel.cut_origin + Vec2::new(panel.span * panel.fullness, panel.height),
                ),
                &points,
            );
            let heading_row = &points[points.len() - (nx + 1)..];
            let step = if heading == CurtainHeading::Wave {
                3
            } else {
                6
            };
            let rail_bottom = self.hi.y - 0.026 - TRACK_HEIGHT * 0.5;
            for &pin in heading_row.iter().step_by(step) {
                self.tube(
                    Surface::Metal,
                    &[pin, Vec3::new(pin.x, rail_bottom, TRACK_Z)],
                    0.0024,
                );
            }
            if panel.held {
                self.holdback(&panel);
            }
        }
    }

    fn holdback(&mut self, panel: &Panel) {
        let y = panel.top - (1. - panel.tie_at) * panel.height;
        let sign = if panel.left { 1. } else { -1. };
        let width = panel.row_width(panel.tie_at);
        let x = |offset| panel.jamb + sign * offset;
        // A cord wraps around the face and free edge of the gathered cloth
        // and returns to its fixing on the jamb, behind the fabric on both ends.
        self.block(
            Surface::Metal,
            Vec3::new(x(0.), y, 0.006),
            Vec3::new(0.012, 0.024, 0.012),
            0.002,
        );
        self.tube(
            self.p.fabric,
            &[
                Vec3::new(x(-0.003), y + 0.007, 0.011),
                Vec3::new(x(-0.003), y, TRACK_Z + 0.030),
                Vec3::new(x(width * 0.51), y - 0.014, TRACK_Z + 0.039),
                Vec3::new(x(width + 0.006), y - 0.007, TRACK_Z + 0.025),
                Vec3::new(x(width + 0.006), y, TRACK_Z - 0.032),
                Vec3::new(x(0.), y - 0.005, 0.011),
            ],
            0.0035,
        );
    }
}
