//! Paper, canvas, paint body and varnish are independent of the depicted subject.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct ArtFinish {
    pub canvas: f32,
    pub paint_body_m: f32,
    pub paper_tooth_m: f32,
    pub roughness: f32,
    pub varnish: f32,
    pub varnish_roughness: f32,
    pub stroke_counts: [u32; 2],
    pub canvas_counts: [u32; 2],
}
impl ArtFinish {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x41525446494e4953);
        Self {
            canvas: if rng.random_bool(0.48) {
                rng.random_range(0.3..1.)
            } else {
                0.
            },
            paint_body_m: if rng.random_bool(0.48) {
                0.
            } else {
                rng.random_range(0.00002..0.00028)
            },
            paper_tooth_m: rng.random_range(0.000003..0.000025),
            roughness: rng.random_range(0.58..0.94),
            varnish: if rng.random_bool(0.45) {
                rng.random_range(0.10..0.60)
            } else {
                0.
            },
            varnish_roughness: rng.random_range(0.12..0.35),
            stroke_counts: [rng.random_range(18..=53), rng.random_range(3..=13)],
            canvas_counts: [rng.random_range(65..=103), rng.random_range(69..=109)],
        }
    }
    pub(super) fn maps(&self, seed: u64, color: Vec<u8>) -> [Image; 3] {
        let mut budget = 96 * 1024;
        let tooth = field::PreparedPeriodic::new([137, 151], seed.wrapping_add(401), &mut budget);
        let stroke =
            field::PreparedPeriodic::new(self.stroke_counts, seed.wrapping_add(409), &mut budget);
        let mut heights = vec![0.; 256 * 256];
        let mut data = Vec::with_capacity(256 * 256 * 4);
        for y in 0..256 {
            for x in 0..256 {
                let uv = [x as f32 / 256., y as f32 / 256.];
                let tooth = tooth.sample(uv[0], uv[1]) - 0.5;
                let stroke = stroke.sample(uv[0], uv[1]) - 0.5;
                // Interlaced yarn crowns, gently irregular, below a millimetre.
                let [nx, ny] = self.canvas_counts;
                let u = (uv[0] * nx as f32).fract();
                let v = (uv[1] * ny as f32).fract();
                let checker =
                    (((uv[0] * nx as f32) as u32 + (uv[1] * ny as f32) as u32) % 2) as f32;
                let crown = |t: f32| (1. - (t * 2. - 1.).powi(2)).max(0.).sqrt();
                let weave = crown(u) * (1. - checker) + crown(v) * checker;
                heights[y * 256 + x] = tooth * self.paper_tooth_m
                    + stroke * self.paint_body_m
                    + (weave - 0.5) * self.canvas * 0.00005;
                let rough = (self.roughness + tooth * 0.04 - stroke * 0.045).clamp(0.48, 0.98);
                data.extend([255, (rough * 255.).round() as u8, 0, 255]);
            }
        }
        let mut normals = Vec::with_capacity(256 * 256 * 4);
        // 60 cm reference support; no relief from image brightness/subject edges.
        for y in 0..256 {
            for x in 0..256 {
                let dx =
                    heights[y * 256 + (x + 1).min(255)] - heights[y * 256 + x.saturating_sub(1)];
                let dy =
                    heights[(y + 1).min(255) * 256 + x] - heights[y.saturating_sub(1) * 256 + x];
                normals.extend(filter::encode(
                    Vec3::new(-dx * 256. / 1.2, -dy * 256. / 1.2, 1.).normalize(),
                ));
            }
        }
        let mut images = filter::images((color, normals, data), 256);
        for image in &mut images {
            image.sampler = ImageSampler::linear();
        }
        images
    }
}

#[cfg(test)]
mod replay_tests;
