//! Woven-wood sliding panels hang from separate tracks. A raised blind can
//! disappear into a roll; a sliding panel must still occupy its parking width.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PanelTrack {
    pub panel_width_m: f32,
    pub reed_pitch_m: f32,
    pub openness: f32,
    /// Variation of the cut splint widths within a woven roll.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reed_variation: Option<f32>,
}

impl PanelTrack {
    pub(super) fn parks_right(stack: Stack, identity: u64) -> bool {
        stack == Stack::Right || stack == Stack::Pair && identity.is_multiple_of(2)
    }

    pub(super) fn sample(r: &mut impl Rng) -> Self {
        Self {
            panel_width_m: r.random_range(0.48..0.96),
            reed_pitch_m: r.random_range(0.019..0.043),
            openness: r.random_range(0.09..0.32),
            reed_variation: Some(0.18),
        }
    }

    pub(super) fn valid(self) -> bool {
        (0.40..=1.1).contains(&self.panel_width_m)
            && (0.016..=0.05).contains(&self.reed_pitch_m)
            && (0.06..=0.36).contains(&self.openness)
            && self
                .reed_variation
                .is_none_or(|v| (0.04..=0.28).contains(&v))
    }

    /// One piece of woven stock, cut to height. Each panel can start at a
    /// different part of the roll. Normalization preserves the actual cut
    /// length and real gaps at every extension, without another mesh or map.
    pub(super) fn reed_edges(
        self,
        identity: u64,
        panel: usize,
        count: usize,
        height: f32,
    ) -> Option<[f32; 81]> {
        let variation = self.reed_variation?;
        let mut cuts = stream(identity, 0x53504c494e5453 + panel as u64);
        let mut edges = [0.; 81];
        let mut bundle = 0.;
        for i in 0..count {
            bundle = 0.68 * bundle + 0.32 * cuts.random_range(-1.0..1.0);
            let gauge = 1. + variation * (bundle + cuts.random_range(-0.72..0.72));
            edges[i + 1] = edges[i] + gauge;
        }
        let scale = height / edges[count];
        for edge in &mut edges[1..count] {
            *edge *= scale;
        }
        edges[count] = height;
        Some(edges)
    }

    /// Clear track length excludes the aperture margins and end stops.
    pub(super) fn panels(self, aperture_width: f32) -> (usize, f32, f32) {
        let span = (aperture_width - 0.060).max(0.02);
        let n = (span / self.panel_width_m).round().clamp(2., 5.) as usize;
        let overlap = 0.024_f32.min(span * 0.04);
        let step = (span - overlap) / n as f32;
        (n, step + overlap, step)
    }

    /// Frontal clear area of uncovered space plus the open weave. Stacked
    /// panels have independently pitched reeds. Integrate each overlap interval
    /// rather than treating the parking area as another unobstructed window.
    pub(crate) fn clear_fraction(self, aperture_width: f32, coverage: f32) -> f32 {
        let (n, width, step) = self.panels(aperture_width);
        let full_span = width + step * (n - 1) as f32;
        let stride = step * coverage.clamp(0., 1.);
        let mut edges = [0.; 10];
        for i in 0..n {
            edges[i * 2] = i as f32 * stride;
            edges[i * 2 + 1] = edges[i * 2] + width;
        }
        let sorted = &mut edges[..2 * n];
        sorted.sort_unstable_by(f32::total_cmp);
        let mut clear = full_span - sorted[2 * n - 1];
        for pair in sorted.windows(2) {
            let x = (pair[0] + pair[1]) * 0.5;
            let layers = (0..n)
                .filter(|i| (0.0..width).contains(&(x - *i as f32 * stride)))
                .count();
            clear += (pair[1] - pair[0]) * self.openness.powi(layers as i32);
        }
        (clear / full_span).clamp(0., 1.)
    }
}
