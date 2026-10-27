//! Placement proposals, separate from transactional support/collision admission.
use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KitchenSampling {
    pub unit_gap_m: f32,
    pub run_offset_m: f32,
    pub appliance_coherence: f32,
}
impl KitchenSampling {
    pub(super) fn valid(&self) -> bool {
        (0.012..0.045).contains(&self.unit_gap_m)
            && self.run_offset_m.is_finite()
            && (0.0..=1.).contains(&self.appliance_coherence)
    }
}

pub(super) fn run_shifts(span: f32, total: f32, r: &mut ChaCha8Rng) -> [f32; 5] {
    let free = ((span - total - 0.85) * 0.5).max(0.);
    let mut positions =
        std::array::from_fn(|i| -free + 2. * free * (i as f32 + r.random::<f32>()) / 5.);
    positions.shuffle(r);
    positions
}

pub(super) fn island_shifts(r: &mut ChaCha8Rng) -> [Vec2; 7] {
    let phase = r.random_range(0.0..TAU);
    std::array::from_fn(|i| {
        let angle = phase + TAU * (i as f32 + r.random_range(-0.20..0.20)) / 7.;
        let radius = r.random_range(0.06..1.05);
        Vec2::new(angle.cos(), angle.sin()) * radius
    })
}
