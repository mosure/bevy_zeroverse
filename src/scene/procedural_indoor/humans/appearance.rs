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
    #[serde(default)]
    pub garment: super::garments::GarmentProgram,
    #[serde(default)]
    pub face: FaceProgram,
    #[serde(default = "default_volume")]
    pub hair_volume: f32,
    #[serde(default)]
    pub hairline_raise: f32,
    #[serde(default)]
    pub hair_grey: f32,
    #[serde(default)]
    pub footwear: super::footwear::FootwearProgram,
    #[serde(default)]
    pub hair_program: super::hair::HairProgram,
    /// Optional Anny male-to-female shape anchor; not a gender identity label.
    /// Absent preserves the existing deterministic reference phenotype stream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_gender: Option<f64>,
    /// Explicit adult shape controls. None replays the legacy phenotype stream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_program: Option<super::morphology::BodyProgram>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wardrobe_and_face_replay_independently_with_legacy_defaults() {
        let mut necks = std::collections::HashSet::new();
        let mut irises = std::collections::HashSet::new();
        for seed in 0..512 {
            let a = Appearance::sample(seed);
            assert_eq!(a, Appearance::sample(seed));
            necks.insert(a.garment.neckline_depth.to_bits());
            irises.insert(a.face.iris.map(f32::to_bits));
            assert!(a.garment.section_roundness <= 2.2);
            assert!((0.0..=1.0).contains(&a.garment.trouser_coverage));
            let mut old = serde_json::to_value(&a).unwrap();
            for key in [
                "garment",
                "face",
                "hair_volume",
                "hairline_raise",
                "hair_grey",
                "footwear",
                "hair_program",
                "body_gender",
                "body_program",
            ] {
                old.as_object_mut().unwrap().remove(key);
            }
            let legacy: Appearance = serde_json::from_value(old).unwrap();
            assert_eq!(legacy.face.stubble, 0.0);
            assert!(legacy.body_program.is_none());
            assert_eq!(
                legacy.garment,
                super::super::garments::GarmentProgram::default()
            );
        }
        assert!(necks.len() > 500 && irises.len() > 500);
    }
}
fn default_volume() -> f32 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FaceProgram {
    pub iris: [f32; 3],
    pub brow_width: f32,
    pub brow_thickness: f32,
    pub brow_arch: f32,
    pub lip_tint: f32,
    pub stubble: f32,
    pub frame_metallic: f32,
    pub frame_color: [f32; 3],
}
impl Default for FaceProgram {
    fn default() -> Self {
        Self {
            iris: [0.18, 0.12, 0.07],
            brow_width: 0.041,
            brow_thickness: 0.0045,
            brow_arch: 0.004,
            lip_tint: 0.5,
            stubble: 0.0,
            frame_metallic: 0.0,
            frame_color: [0.08, 0.07, 0.06],
        }
    }
}
impl FaceProgram {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x464143454445544c);
        let iris = Color::hsl(
            rng.random_range(20.0..230.0),
            rng.random_range(0.15..0.55),
            rng.random_range(0.12..0.38),
        )
        .to_srgba();
        let frame = Color::hsl(
            rng.random_range(0.0..360.0),
            rng.random_range(0.0..0.45),
            rng.random_range(0.035..0.28),
        )
        .to_srgba();
        Self {
            iris: [iris.red, iris.green, iris.blue],
            brow_width: rng.random_range(0.034..0.050),
            brow_thickness: rng.random_range(0.0026..0.0065),
            brow_arch: rng.random_range(0.0015..0.007),
            lip_tint: rng.random_range(0.1..0.85),
            stubble: if rng.random_bool(0.25) {
                rng.random_range(0.25..0.9)
            } else {
                0.0
            },
            frame_metallic: if rng.random_bool(0.35) {
                rng.random_range(0.7..1.0)
            } else {
                0.0
            },
            frame_color: [frame.red, frame.green, frame.blue],
        }
    }
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
            garment: super::garments::GarmentProgram::sample(seed),
            face: FaceProgram::sample(seed),
            hair_volume: rng.random_range(0.55..1.65),
            hairline_raise: rng.random_range(-0.015..0.042),
            hair_grey: if rng.random_bool(0.20) {
                rng.random_range(0.12..0.85)
            } else {
                0.0
            },
            footwear: super::footwear::FootwearProgram::sample(seed),
            hair_program: super::hair::HairProgram::sample(seed),
            body_gender: None,
            body_program: Some(super::morphology::BodyProgram::sample(seed)),
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
            HumanSurface::Lip => [
                skin[0] * (0.92 - self.face.lip_tint * 0.10),
                skin[1] * (0.84 - self.face.lip_tint * 0.28),
                skin[2] * (0.86 - self.face.lip_tint * 0.22),
            ],
            HumanSurface::Top => self.top,
            HumanSurface::Seam => self.top.map(|v| v * 0.86),
            HumanSurface::Shirt => [0.87, 0.88, 0.86],
            HumanSurface::Iris => self.face.iris,
            HumanSurface::Trousers => self.trousers,
            HumanSurface::Hair | HumanSurface::Brow => self
                .hair
                .map(|v| v * (1.0 - self.hair_grey * 0.65) + 0.56 * self.hair_grey),
            HumanSurface::FacialHair => std::array::from_fn(|i| {
                skin[i] * (1.0 - self.face.stubble * 0.65) + self.hair[i] * self.face.stubble * 0.65
            }),
            HumanSurface::Eyewear => self.face.frame_color,
            HumanSurface::Shoes => self.footwear.upper_color,
            HumanSurface::Sole | HumanSurface::ShoeDetail => self.footwear.sole_color,
            _ => return None,
        };
        Some(Color::srgb(rgb[0], rgb[1], rgb[2]))
    }
}
