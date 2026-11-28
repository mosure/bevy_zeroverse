//! A solid carving with one or two bores. Dimensions with an _m suffix are local
//! metres. Profile and bore coordinates are unitless. The mount yaw is independent
//! of the plinth's pose and the finish stream.
use super::*;
use rand_chacha::ChaCha8Rng;

pub(super) const VERSION: u32 = 306;

#[derive(Clone, Debug, serde::Serialize)]
pub struct Aperture {
    /// Start and end along the carving axis, measured from 0 to 1.
    pub bottom: f32,
    pub top: f32,
    /// Position and radius relative to the half-width at this elevation.
    pub center: f32,
    pub radius: f32,
    pub drift: f32,
    /// Fraction by which the bore narrows between the two lips.
    pub throat: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum CarvingAxis {
    Vertical,
    Horizontal,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Carving {
    pub axis: CarvingAxis,
    pub profile_exponent: f32,
    /// Half of the flat bearing span as a fraction of the horizontal run.
    pub bearing_half_span: f32,
    pub half_width_m: f32,
    pub half_depth_m: f32,
    pub height_m: f32,
    pub foot_y_m: f32,
    pub mount_yaw_radians: f32,
    pub apertures: Vec<Aperture>,
    pub bore_surface: Surface,
    pub lean: f32,
    pub shoulder: f32,
    pub waist: f32,
    pub waist_height: f32,
    pub end_latitudes: [f32; 2],
    pub face_inset: f32,
    pub bore_shear: f32,
    pub bow: f32,
}

impl Carving {
    pub(super) fn sample(o: &IndoorObject, version: u32, p: &mut SculptureProgram) -> Option<Self> {
        if version < VERSION {
            return None;
        }
        let mut r = stream(o.seed, u64::from_le_bytes(*b"PiercArt"));
        if !r.random_bool(0.38) {
            return None;
        }
        let mut finish = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"CastSkin"),
        );
        p.surface = [
            Surface::Metal,
            Surface::Concrete,
            Surface::Wood,
            Surface::Ceramic,
            Surface::Terracotta,
        ][finish.random_range(0..5)];
        // Paint or glaze may terminate at the lip. A polished metal bore
        // remains metallic and an unglazed ceramic bore exposes its clay.
        let bore_surface = if finish.random_bool(0.55) {
            match p.surface {
                Surface::Metal => Surface::Chrome,
                Surface::Ceramic => Surface::Terracotta,
                Surface::Terracotta => Surface::Ceramic,
                _ => p.surface,
            }
        } else {
            p.surface
        };
        let aperture = |r: &mut ChaCha8Rng, bottom, top| Aperture {
            bottom,
            top,
            center: r.random_range(-0.26..0.26),
            radius: r.random_range(0.29..0.62),
            drift: r.random_range(-0.12..0.12),
            throat: r.random_range(0.06..0.27),
        };
        let apertures = if r.random_bool(0.46) {
            let divide = r.random_range(0.40..0.60);
            let bottom = r.random_range(0.08..0.15);
            let lo = divide - r.random_range(0.04..0.105);
            let hi = divide + r.random_range(0.04..0.105);
            let top = r.random_range(0.81..0.93);
            vec![aperture(&mut r, bottom, lo), aperture(&mut r, hi, top)]
        } else {
            let bottom = r.random_range(0.13..0.37);
            let top = r.random_range(0.65..0.90);
            vec![aperture(&mut r, bottom, top)]
        };
        // Fit the projected width and depth jointly to the admitted rectangle.
        // Choosing the longer axis first also works on narrow vitrine shelves.
        let size = o.size;
        let yaw = if size.x >= size.z { 0. } else { FRAC_PI_2 }
            + r.random_range(-0.86..0.86)
            + if r.random() { PI } else { 0. };
        let (sin, cos) = yaw.sin_cos();
        let depth_ratio = r.random_range(0.28..0.67);
        let bore_shear: f32 = r.random_range(-0.09..0.09);
        let wx = 1. + bore_shear.abs();
        let wz = depth_ratio * 1.12;
        let width = r.random_range(0.83..0.97)
            * 0.5
            * (size.x / (cos.abs() * wx + sin.abs() * wz))
                .min(size.z / (sin.abs() * wx + cos.abs() * wz));
        let base_top = size.y * p.base_height;
        let joint = (base_top * 0.05).min(0.0006);
        let horizontal_probability = (0.12 + 0.60 * (2. * width / size.y - 0.45)).clamp(0.04, 0.87);
        Some(Self {
            axis: if r.random_bool(horizontal_probability.into()) {
                CarvingAxis::Horizontal
            } else {
                CarvingAxis::Vertical
            },
            profile_exponent: r.random_range(0.48..1.18),
            bearing_half_span: r.random_range(0.135..0.195),
            half_width_m: width,
            half_depth_m: width * depth_ratio,
            foot_y_m: base_top - joint,
            height_m: (size.y - base_top) * r.random_range(0.80..0.98),
            mount_yaw_radians: yaw,
            apertures,
            bore_surface,
            lean: r.random_range(-0.32..0.32),
            shoulder: r.random_range(-0.19..0.19),
            waist: r.random_range(0.02..0.34),
            waist_height: r.random_range(0.27..0.72),
            end_latitudes: [r.random_range(-0.96..-0.86), r.random_range(0.88..0.984)],
            face_inset: r.random_range(0.035..0.115),
            bore_shear,
            bow: r.random_range(-0.11..0.11),
        })
    }

    pub fn bore_span_m(&self, eye: &Aperture) -> f32 {
        let span = match self.axis {
            CarvingAxis::Vertical => self.height_m,
            CarvingAxis::Horizontal => 2. * self.half_width_m,
        };
        (eye.top - eye.bottom) * (1. - self.face_inset) * span
    }

    /// Section perpendicular to the carving axis. The two faces are ruled
    /// across each section so triangles remain well conditioned at a bore tip,
    /// including on wide, shallow pieces with a narrow gap between bores.
    pub(super) fn section(&self, u: f32, v: f32) -> (Vec3, f32) {
        let latitude = self.end_latitudes[0].lerp(self.end_latitudes[1], v);
        let arch = 4. * v * (1. - v);
        let center = self.lean * v + self.shoulder * arch;
        let distance = 3. * (v - self.waist_height);
        let waist = (1. - distance * distance).max(0.);
        let half = (1. - latitude * latitude).powf(self.profile_exponent)
            * (1. - self.waist * waist * waist)
            * (1. - center.abs());
        let (x, y) = match self.axis {
            CarvingAxis::Vertical => (center + half * u, v),
            CarvingAxis::Horizontal => {
                // A landscape carving rests on a broad, flat central foot and
                // rises towards both ends. The X/Y chart keeps outward winding.
                let end = ((v - 0.5).abs() - self.bearing_half_span).max(0.)
                    / (0.5 - self.bearing_half_span);
                let bottom = 0.40 * end * end;
                // Leave a positive end section even on a deeply tapered,
                // leaning piece. Its crown must stay above the raised foot.
                let crown = 0.52 + 0.42 * half + 0.12 * center;
                (2. * v - 1., crown.lerp(bottom, (u + 1.) * 0.5))
            }
        };
        let mid = Vec3::new(
            x * self.half_width_m,
            self.foot_y_m + y * self.height_m,
            self.bow * arch * self.half_depth_m,
        );
        let swell = (0.21 * self.half_depth_m).min(0.10 * self.height_m);
        (mid, 0.72 * self.half_depth_m + swell * arch)
    }

    pub(super) fn build(&self, assembly: &mut Assembly, surface: Surface) {
        super::carved_mesh::build(assembly, self, surface);
    }
}
