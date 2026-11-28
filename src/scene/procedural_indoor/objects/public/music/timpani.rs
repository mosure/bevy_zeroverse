//! A suspended concert kettle. Local +Z faces the player and the tuning pedal.
use super::*;
use crate::scene::procedural_indoor::{
    geometry::{ProfileJoints, RevolvedOutline},
    materials::drum_stock,
};
mod carriage;
#[cfg(test)]
mod tests;

pub(crate) const VERSION: u32 = 308;
pub(crate) const SIZES: [u8; 5] = [20, 23, 26, 29, 32];
const LABEL: &str = "musical_instrument";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum BowlStock {
    HammeredCopper,
    SpunCopper,
    SatinAluminium,
    LacqueredComposite,
}

impl BowlStock {
    pub(crate) fn sample(line: u64) -> Self {
        match stream(line, u64::from_le_bytes(*b"DrumLine")).random_range(0..10) {
            0..=3 => Self::HammeredCopper,
            4..=6 => Self::SpunCopper,
            7..=8 => Self::SatinAluminium,
            _ => Self::LacqueredComposite,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Timpani {
    pub nominal_inches: u8,
    pub head_radius_m: f32,
    pub head_height_m: f32,
    pub bowl_depth_m: f32,
    pub bowl_exponent: f32,
    pub bowl_center_z_m: f32,
    pub bowl_stock: BowlStock,
    pub tension_rods: usize,
    pub wheel_radius_m: f32,
    pub pedal_pitch_rad: f32,
    /// Angle around the rim from the player side. Some portable frames omit the gauge.
    pub gauge_azimuth_rad: Option<f32>,
}

pub(crate) fn dimensions(inches: u8, height: f32, r: &mut impl Rng) -> Vec3 {
    let diameter = f32::from(inches) * 0.0254;
    Vec3::new(
        diameter + r.random_range(0.121..0.150),
        height,
        diameter + r.random_range(0.178..0.246),
    )
}

pub(in super::super) fn catalog_dimensions(r: &mut impl Rng) -> Vec3 {
    let inches = SIZES[r.random_range(0..SIZES.len())];
    dimensions(inches, r.random_range(0.82..0.88), r)
}

impl Timpani {
    pub(crate) fn nominal_size(width_m: f32) -> u8 {
        // The manufacturer's nominal bore is distinct from the outside hardware.
        SIZES
            .into_iter()
            .min_by(|a, b| {
                let error = |n: u8| (f32::from(n) * 0.0254 + 0.135 - width_m).abs();
                error(*a).total_cmp(&error(*b))
            })
            .unwrap()
    }

    pub fn sample(o: &IndoorObject) -> Self {
        let line = o.finish_seed.unwrap_or(o.seed);
        let mut kit = stream(line, u64::from_le_bytes(*b"KettlKit"));
        let mut drum = stream(o.seed, u64::from_le_bytes(*b"KettlUse"));
        let nominal_inches = Self::nominal_size(o.size.x);
        let radius = (f32::from(nominal_inches) * 0.0127)
            .min(o.size.x * 0.46)
            .min(o.size.z * 0.46);
        Self {
            nominal_inches,
            head_radius_m: radius,
            head_height_m: o.size.y - 0.009,
            bowl_depth_m: (radius * kit.random_range(1.15..1.39)).min(o.size.y - 0.23),
            bowl_exponent: kit.random_range(0.43..0.75),
            bowl_center_z_m: -o.size.z * 0.5 + radius + 0.047,
            bowl_stock: BowlStock::sample(line),
            tension_rods: if nominal_inches >= 29 || (nominal_inches == 26 && kit.random_bool(0.38))
            {
                8
            } else {
                6
            },
            wheel_radius_m: kit.random_range(0.043..0.057),
            pedal_pitch_rad: drum.random_range(0.08..0.29),
            gauge_azimuth_rad: kit.random_bool(0.84).then(|| {
                let angle = kit.random_range(0.61..0.92);
                if kit.random_bool(0.5) {
                    angle
                } else {
                    -angle
                }
            }),
        }
    }

    fn point(self, angle: f32, radius: f32, y: f32) -> Vec3 {
        Vec3::new(
            angle.sin() * radius,
            y,
            self.bowl_center_z_m + angle.cos() * radius,
        )
    }
}

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let t = Timpani::sample(o);
    let r = t.head_radius_m;
    let h = t.head_height_m;
    let tf = Transform::from_xyz(0., 0., t.bowl_center_z_m);
    let outline = RevolvedOutline::round(40, 0., 0);
    let bottom = h - t.bowl_depth_m;
    // The kettle is vented underneath. Carry the sheet around the aperture;
    // a separate tensioned film closes only its upper bearing edge.
    let mut profile = [(0., 0.); 11];
    profile[0] = (r * 0.085, bottom + 0.009);
    profile[1] = (r * 0.085, bottom);
    for (i, point) in profile[2..].iter_mut().enumerate() {
        let u = (i + 1) as f32 / 9.;
        *point = (
            r * u.powf(t.bowl_exponent),
            bottom + (h - bottom - 0.003) * u,
        );
    }
    let (surface, label) = drum_stock::bowl_label(t.bowl_stock);
    outline.sweep(a.part(surface, label), &profile, ProfileJoints::Smooth, tf);
    // Radial strips share the exact vertices at the skin/counterhoop junction.
    a.part(Surface::Plastic, drum_stock::HEAD_LABEL).lathe(
        &[(0., h - 0.003), (r, h - 0.003), (r, h), (0., h)],
        40,
        tf,
    );
    a.part(Surface::Chrome, LABEL).lathe(
        &[
            (r, h - 0.035),
            (r + 0.018, h - 0.033),
            (r + 0.018, h + 0.003),
            (r + 0.012, h + 0.005),
            (r, h),
        ],
        40,
        tf,
    );
    carriage::build(a, o, t);
    if let Some(azimuth) = t.gauge_azimuth_rad {
        carriage::gauge(a, t, azimuth);
    }
}
