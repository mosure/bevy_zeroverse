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
    #[serde(default = "default_sleeve")]
    pub sleeve_coverage: f32,
    #[serde(default = "default_hem")]
    pub hem_fraction: f32,
    #[serde(default = "default_weave_scale")]
    pub weave_scale: f32,
    #[serde(default)]
    pub weave_rotation: f32,
}
fn default_sleeve() -> f32 {
    0.95
}
fn default_hem() -> f32 {
    0.14
}
fn default_weave_scale() -> f32 {
    1.0
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
        let mut trousers = color(0.07, 0.48);
        // Most trousers are neutral/dark; retain a smaller chromatic component.
        if rng.random_bool(0.80) {
            let n = rng.random_range(0.06..0.36);
            let warmth = rng.random_range(-0.08..0.14);
            trousers = [n * (1.0 + warmth), n, n * (1.0 - warmth)];
        }
        let darkness = rng.random_range(0.045..0.56);
        let hair = if rng.random_bool(0.08) {
            [darkness; 3]
        } else {
            let green = darkness * rng.random_range(0.50..0.86);
            [darkness, green, green * rng.random_range(0.45..0.78)]
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
            sleeve_coverage: rng.random_range(0.0..1.0),
            hem_fraction: rng.random_range(0.06..0.23),
            weave_scale: rng.random_range(0.55..1.85),
            weave_rotation: rng.random_range(-0.4..0.4),
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
            HumanSurface::Hair | HumanSurface::Brow => self.hair,
            HumanSurface::Eyewear => self.hair.map(|v| (v * 0.6).clamp(0.025, 0.22)),
            _ => return None,
        };
        Some(Color::srgb(rgb[0], rgb[1], rgb[2]))
    }
}
