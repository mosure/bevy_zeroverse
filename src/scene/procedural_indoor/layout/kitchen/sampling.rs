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

impl IndoorManifest {
    pub(super) fn sink_mat(&self, sink: &IndoorObject, r: &mut ChaCha8Rng) -> IndoorObject {
        let mut fit = stream(sink.seed, 0x5349_4e4b_4d41_5453);
        let mut width = r.random_range(0.70..1.30);
        if fit.random_bool(0.26) {
            width += fit.random_range(0.44..1.08);
        }
        let size = Vec3::new(
            width,
            fit.random_range(0.004..0.011),
            r.random_range(0.44..0.76),
        );
        let angle = fit.random_range(-0.20_f32..0.20) * fit.random::<f32>();
        let (sin, cos) = angle.sin_cos();
        // Measure the toe gap from the closest rotated corner, so a long runner
        // cannot swing through the cabinet. The user can still stand at the sink.
        let reach = (size.x * sin.abs() + size.z * cos) * 0.5;
        let lateral = (size.x * 0.28).min(sink.size.x * 0.40);
        let center = Vec3::new(
            fit.random_range(-lateral..lateral),
            0.,
            sink.size.z * 0.5 + reach + fit.random_range(0.035..0.16),
        );
        let mut mat = self.floor_candidate(
            ObjectKind::Rug,
            sink.transform().transform_point(center),
            size,
            sink.yaw + angle,
            r,
        );
        mat.solid = false;
        mat
    }
}
