//! Frozen scalar art-finish fields and filtering for exact replay.
use super::*;
impl ArtFinish {
    fn reference_maps(&self, seed: u64, color: Vec<u8>) -> [Image; 3] {
        let mut heights = vec![0.; 256 * 256];
        let mut data = Vec::with_capacity(256 * 256 * 4);
        for y in 0..256 {
            for x in 0..256 {
                let uv = [x as f32 / 256., y as f32 / 256.];
                let noise =
                    |nx, ny, salt| periodic_noise(uv[0], uv[1], nx, ny, seed.wrapping_add(salt));
                let tooth = noise(137, 151, 401) - 0.5;
                let stroke = noise(self.stroke_counts[0], self.stroke_counts[1], 409) - 0.5;
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
#[test]
fn prepared_art_fields_preserve_every_image_and_mip() {
    for seed in [0, 7, 202, 600, u64::MAX] {
        let finish = ArtFinish::sample(seed);
        let color: Vec<_> = (0..256 * 256)
            .flat_map(|i| [i as u8, (i / 256) as u8, (i * 73) as u8, 255])
            .collect();
        let before = finish.reference_maps(seed, color.clone());
        let after = finish.maps(seed, color);
        for (before, after) in before.into_iter().zip(after) {
            assert_eq!(before.texture_descriptor, after.texture_descriptor);
            assert_eq!(before.data, after.data, "art finish seed {seed}");
            assert_eq!(
                format!("{:?}", before.sampler),
                format!("{:?}", after.sampler)
            );
        }
    }
}
