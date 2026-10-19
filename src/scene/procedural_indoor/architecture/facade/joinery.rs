//! Section dimensions shared by every light in a window installation.
use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowJoinery {
    pub identity: u64,
    pub mullion_fraction: f32,
    pub mullion_depth_fraction: f32,
    pub column_spread: f32,
    pub sash_width_m: f32,
    pub sash_depth_m: f32,
    pub seal_width_m: f32,
    pub sill_thickness_m: f32,
    pub sill_extension_m: f32,
    pub sill_surface: Surface,
    pub frame_finish_slot: usize,
    /// Applied face bead fits entirely inside the rough-opening frame.
    pub face_bead_fraction: f32,
    pub face_bead_depth_m: f32,
}

impl WindowJoinery {
    pub fn sample(seed: u64) -> Self {
        let mut r = stream(seed, 3161);
        Self {
            identity: seed,
            mullion_fraction: r.random_range(0.38..0.98),
            mullion_depth_fraction: r.random_range(0.60..0.98),
            column_spread: r.random_range(0.05..0.65),
            sash_width_m: r.random_range(0.012..0.045),
            sash_depth_m: r.random_range(0.018..0.046),
            seal_width_m: r.random_range(0.001..0.003),
            sill_thickness_m: r.random_range(0.014..0.064),
            sill_extension_m: r.random_range(0.0..0.028),
            sill_surface: [
                Surface::WoodEdge,
                Surface::Paint,
                Surface::Ceramic,
                Surface::Metal,
            ][r.random_range(0..4)],
            frame_finish_slot: r.random_range(0..super::super::super::materials::variants::COUNT),
            face_bead_fraction: if r.random_bool(0.65) {
                r.random_range(0.18..0.52)
            } else {
                0.
            },
            face_bead_depth_m: r.random_range(0.002..0.008),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(0.3..=1.).contains(&self.mullion_fraction)
            || !(0.5..=1.).contains(&self.mullion_depth_fraction)
            || !(0.0..=0.7).contains(&self.column_spread)
            || !(0.01..=0.05).contains(&self.sash_width_m)
            || !(0.015..=0.05).contains(&self.sash_depth_m)
            || !(0.0005..=0.004).contains(&self.seal_width_m)
            || !(0.01..=0.07).contains(&self.sill_thickness_m)
            || !(0.0..=0.03).contains(&self.sill_extension_m)
            || !(0.0..=0.55).contains(&self.face_bead_fraction)
            || !(0.001..=0.01).contains(&self.face_bead_depth_m)
            || self.frame_finish_slot >= super::super::super::materials::variants::COUNT
            || !matches!(
                self.sill_surface,
                Surface::WoodEdge | Surface::Paint | Surface::Ceramic | Surface::Metal
            )
        {
            return Err("invalid window joinery dimensions".into());
        }
        Ok(())
    }
}

impl WindowOpening {
    /// Sorted construction levels, excluding frame boundaries. Exported in the
    /// manifest and reused by pane construction, validation and metrics.
    pub fn rows(&self) -> Vec<f32> {
        let mut rows = self.transoms.clone();
        if self.transom > 0. {
            rows.push(self.transom);
        }
        rows.sort_by(f32::total_cmp);
        rows
    }
}

impl Facade {
    pub(crate) fn divisions(&self, o: &WindowOpening) -> (Vec<f32>, Vec<f32>) {
        let lo = o.min + Vec2::splat(self.frame_width);
        let hi = o.max - Vec2::splat(self.frame_width);
        let weights: Vec<f32> = (0..o.columns)
            .map(|i| {
                let identity = o
                    .glazing
                    .as_ref()
                    .map(|p| p.identity)
                    .or_else(|| self.joinery.as_ref().map(|j| j.identity));
                identity.map_or(1., |identity| {
                    let mut r = stream(identity, 3110 + u64::from(i));
                    let spread = self.joinery.as_ref().map_or(0.3, |j| j.column_spread);
                    r.random_range(1. - spread..1. + spread)
                })
            })
            .collect();
        let minimum = if self.joinery.is_some() {
            ((hi.x - lo.x) / o.columns as f32 * 0.8).min(0.20)
        } else {
            0.
        };
        let scale = (hi.x - lo.x - minimum * o.columns as f32) / weights.iter().sum::<f32>();
        let mut columns = vec![lo.x];
        for weight in weights {
            columns.push(columns.last().unwrap() + minimum + weight * scale);
        }
        *columns.last_mut().unwrap() = hi.x;
        let mut levels = vec![lo.y];
        levels.extend(o.rows().into_iter().map(|row| lo.y + (hi.y - lo.y) * row));
        levels.push(hi.y);
        (columns, levels)
    }

    /// Clear pane rectangles actually sent to the sash builder, in wall metres.
    pub fn pane_rectangles(&self, o: &WindowOpening) -> Vec<(Vec2, Vec2)> {
        let (columns, levels) = self.divisions(o);
        let bar = self.frame_width * self.joinery.as_ref().map_or(0.66, |j| j.mullion_fraction);
        let mut result = Vec::new();
        for (col, x) in columns.windows(2).enumerate() {
            for (row, y) in levels.windows(2).enumerate() {
                result.push((
                    Vec2::new(
                        x[0] + if col == 0 { 0. } else { bar * 0.5 },
                        y[0] + if row == 0 { 0. } else { bar * 0.5 },
                    ),
                    Vec2::new(
                        x[1] - if col + 2 == columns.len() {
                            0.
                        } else {
                            bar * 0.5
                        },
                        y[1] - if row + 2 == levels.len() {
                            0.
                        } else {
                            bar * 0.5
                        },
                    ),
                ));
            }
        }
        result
    }

    /// Applied sill boards occupy some solid wall below their opening. Trim
    /// stops around that actual volume, including its small side horns.
    pub(crate) fn trim_openings(&self) -> Vec<WindowOpening> {
        self.openings
            .iter()
            .cloned()
            .map(|mut o| {
                if o.min.y > 0.25 && self.sill_projection > 0. {
                    let (thickness, extension) =
                        self.joinery.as_ref().map_or((0.024, 0.015), |j| {
                            (j.sill_thickness_m + 0.002, j.sill_extension_m)
                        });
                    o.min -= Vec2::new(extension + 0.002, thickness + 0.002);
                    o.max.x += extension + 0.002;
                }
                o
            })
            .collect()
    }

    pub(crate) fn frame_label(&self) -> String {
        self.joinery.as_ref().map_or_else(
            || "window".into(),
            |j| format!("window#finish{}", j.frame_finish_slot),
        )
    }
}
