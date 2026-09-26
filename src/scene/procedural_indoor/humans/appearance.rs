//! Continuous phenotype-independent surface and garment parameters.
use super::*;
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Appearance {
    pub melanin: f32,
    pub blood_tint: f32,
    pub skin_roughness: f32,
    pub top: [f32; 3],
    pub trousers: [f32; 3],
    pub hair: [f32; 3],
    pub garment_ease: f32,
    pub fold_amplitude: f32,
    pub fold_frequency: f32,
    pub cloth_roughness: f32,
    pub hair_length: f32,
    pub hair_part: f32,
    pub hair_curl: f32,
}
impl Appearance {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 46);
        let mut color = |light_lo, light_hi| {
            let c = Color::hsl(
                rng.random_range(0.0..360.0),
                rng.random_range(0.04..0.45),
                rng.random_range(light_lo..light_hi),
            )
            .to_srgba();
            [c.red, c.green, c.blue]
        };
        let top = color(0.13, 0.79);
        let trousers = color(0.07, 0.48);
        let darkness = rng.random_range(0.045..0.56);
        let hair = if rng.random_bool(0.08) {
            [darkness; 3]
        } else {
            [
                darkness,
                darkness * rng.random_range(0.45..0.86),
                darkness * rng.random_range(0.25..0.66),
            ]
        };
        Self {
            melanin: rng.random_range(0.0..1.0),
            blood_tint: rng.random_range(0.015..0.09),
            skin_roughness: rng.random_range(0.42..0.64),
            top,
            trousers,
            hair,
            garment_ease: rng.random_range(0.008..0.025),
            fold_amplitude: rng.random_range(0.0008..0.004),
            fold_frequency: rng.random_range(25.0..65.0),
            cloth_roughness: rng.random_range(0.73..0.97),
            hair_length: rng.random_range(0.006_f32.ln()..0.11_f32.ln()).exp(),
            hair_part: rng.random_range(-0.6..0.6),
            hair_curl: rng.random_range(0.0..0.06),
        }
    }
    pub fn color(&self, surface: HumanSurface) -> Option<Color> {
        // Continuous pigmentation, in sRGB. This is an appearance prior, not a
        // spectral skin model or a claim of subsurface transport accuracy.
        let t = self.melanin;
        let skin = [
            0.92 * (1.0 - t) + 0.29 * t,
            0.74 * (1.0 - t) + 0.145 * t,
            0.61 * (1.0 - t) + 0.09 * t,
        ];
        let rgb = match surface {
            HumanSurface::Skin => [
                skin[0],
                skin[1] * (1.0 - self.blood_tint * 0.3),
                skin[2] * (1.0 - self.blood_tint * 0.1),
            ],
            HumanSurface::Lip => [skin[0] * 0.87, skin[1] * 0.69, skin[2] * 0.74],
            HumanSurface::Top => self.top,
            HumanSurface::Seam => self.top.map(|v| v * 0.86),
            HumanSurface::Shirt => [0.87, 0.88, 0.86],
            HumanSurface::Iris => [
                0.10 + self.melanin * 0.12,
                0.16 - self.melanin * 0.08,
                0.19 - self.melanin * 0.14,
            ],
            HumanSurface::Trousers => self.trousers,
            HumanSurface::Hair => self.hair,
            _ => return None,
        };
        Some(Color::srgb(rgb[0], rgb[1], rgb[2]))
    }
}
