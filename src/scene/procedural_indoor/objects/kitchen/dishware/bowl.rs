//! A single hollow section supplies the fired body, dipped glaze and serving cavity.
use super::*;
use crate::scene::procedural_indoor::{geometry::ProfileJoints, materials::variants};

pub(crate) const INTRODUCED: u32 = 258;
const STEPS: usize = 6;
const HEEL: usize = 6;
const OUTER_LIP: usize = HEEL + STEPS;
const INNER_LIP: usize = OUTER_LIP + 4;
const INNER_HEEL: usize = INNER_LIP + STEPS;
const FILLET_STEPS: usize = 4;
const WELL: usize = INNER_HEEL + FILLET_STEPS;
type Section = [(f32, f32); WELL + 2];

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BowlForm {
    pub outline_exponent: f32,
    pub foot_height_m: f32,
    pub foot_radius_m: f32,
    pub foot_half_width_m: f32,
    pub lip_radius_m: f32,
    /// Cubic control points in normalized (foot-to-maximum-radius, floor-to-rim) coordinates.
    pub lower_control: Vec2,
    pub upper_control: Vec2,
    /// Curve parameter at the exterior glaze boundary.
    pub dip: f32,
}

impl BowlForm {
    pub(super) fn sample(p: &mut DishProgram, seed: u64) -> Self {
        let mut r = stream(seed, 0x424f_574c_5348_454c);
        // In a microwave or cupboard the available vertical space can far
        // exceed the diameter of the vessel that fits on the shelf.
        let diameter = 2. * p.radius * p.aspect.y.sqrt();
        p.height = p.height.min(diameter * r.random_range(0.34..0.56));
        p.wall = p.wall.min(p.height * 0.12).min(p.radius * 0.075);
        p.mouth = if r.random_bool(0.34) {
            r.random_range(0.78..0.96)
        } else {
            r.random_range(0.96..1.)
        };
        let base = p.radius * p.foot;
        let fullness = r.random::<f32>();
        let outline_exponent = if r.random_bool(0.29) {
            r.random_range(2.5..4.3)
        } else {
            2.
        };
        p.segments = if outline_exponent > 2. {
            32
        } else {
            p.segments.max(24)
        };
        let foot_height_m = if r.random_bool(0.22) {
            0.
        } else {
            (p.wall * r.random_range(0.55..1.90))
                .min(0.006)
                .min(p.height * 0.11)
        };
        Self {
            outline_exponent,
            foot_height_m,
            foot_radius_m: base * r.random_range(0.56..0.77),
            foot_half_width_m: (p.wall * r.random_range(0.40..0.92)).min(base * 0.09),
            lip_radius_m: (p.wall * r.random_range(0.46..1.03)).min(p.height * 0.065),
            lower_control: Vec2::new(
                0.025 + 0.65 * fullness * fullness,
                (p.shoulder_height - 0.26) * 0.80 + 0.075,
            ),
            upper_control: Vec2::new(0.38 + 0.62 * fullness, r.random_range(0.51..0.84)),
            dip: r.random_range(0.40..0.83),
        }
    }

    /// Fit the rounded-square diagonal into the same circle used for placement.
    pub(super) fn outline_scale(&self) -> f32 {
        2_f32.powf(self.outline_exponent.recip() - 0.5)
    }

    /// (outer radius, inner radius, elevation). Paired stations share a height,
    /// so the triangulated sidewalls stay ordered even below a returning lip.
    fn station(&self, p: &DishProgram, t: f32) -> (f32, f32, f32) {
        let base = p.radius * p.foot;
        let span = Vec2::new(
            p.radius - base,
            p.height - self.lip_radius_m - p.interior_floor(),
        );
        let a = span * self.lower_control;
        let b = span * self.upper_control;
        let end = Vec2::new(p.radius * p.mouth - base, span.y);
        let u = 1. - t;
        let curve = (3. * u * u * t) * a + (3. * u * t * t) * b + (t * t * t) * end;
        let tangent = u * u * a + 2. * u * t * (b - a) + t * t * (end - b);
        let radial = base + curve.x;
        // Account for the wall slope when converting stock gauge to a radial separation.
        let side = (p.wall * tangent.length() / tangent.y).min(base * 0.27);
        let gauge = side + (2. * self.lip_radius_m - side) * t.powi(4);
        (radial, radial - gauge, p.interior_floor() + curve.y)
    }

    /// Trim the corner where the interior side meets the flat food support.
    /// The quadratic is tangent to both incident mesh edges.
    fn inner_fillet(p: &DishProgram, corner: Vec2, next: Vec2) -> (Vec2, Vec2) {
        let edge = next - corner;
        let setback = (0.18 * p.radius)
            .min(0.82 * edge.length())
            .min(0.28 * corner.x);
        (
            corner - Vec2::X * setback,
            corner + edge.normalize() * setback,
        )
    }

    pub(super) fn floor_radius(&self, p: &DishProgram) -> f32 {
        let inner = |t| {
            let (_, r, y) = self.station(p, t);
            Vec2::new(r, y)
        };
        Self::inner_fillet(p, inner(0.), inner(self.dip / 3.)).0.x
    }

    fn section(&self, p: &DishProgram) -> Section {
        let mut section = [(0., 0.); WELL + 2];
        let (rf, w, lift) = (
            self.foot_radius_m,
            self.foot_half_width_m,
            self.foot_height_m,
        );
        if lift > 0. {
            section[..5].copy_from_slice(&[
                (0., lift),
                (rf - 1.22 * w, lift),
                (rf - w, 0.),
                (rf + w, 0.),
                (rf + 1.22 * w, lift),
            ]);
        }
        let base = p.radius * p.foot;
        section[HEEL - 1] = (base - p.wall.min(base * 0.08), lift);
        for j in 0..=STEPS {
            let s = j as f32 / 3.;
            let t = if j <= 3 {
                self.dip * s
            } else {
                self.dip + (1. - self.dip) * (s - 1.)
            };
            let (outer, inner, y) = self.station(p, t);
            section[HEEL + j] = (outer, y);
            section[INNER_HEEL - j] = (inner, y);
        }
        let corner = Vec2::from(section[INNER_HEEL]);
        let next = Vec2::from(section[INNER_HEEL - 1]);
        let (flat, side) = Self::inner_fillet(p, corner, next);
        for (i, point) in section[INNER_HEEL..=WELL].iter_mut().enumerate() {
            let t = i as f32 / FILLET_STEPS as f32;
            let p = side.lerp(corner, t).lerp(corner.lerp(flat, t), t);
            *point = (p.x, p.y);
        }
        // Keep the floor planar to the last bit for the planar UV chart.
        section[WELL] = (flat.x, flat.y);
        let edge = self.lip_radius_m;
        let (outside, y) = section[OUTER_LIP];
        let mid = outside - edge;
        let arc = edge * std::f32::consts::FRAC_1_SQRT_2;
        section[OUTER_LIP + 1..INNER_LIP].copy_from_slice(&[
            (mid + arc, y + arc),
            (mid, p.height),
            (mid - arc, y + arc),
        ]);
        section[WELL + 1] = (0., p.interior_floor());
        section
    }

    fn first(&self) -> usize {
        if self.foot_height_m == 0. {
            HEEL - 2
        } else {
            0
        }
    }

    pub(super) fn profile(&self, p: &DishProgram) -> Vec<(f32, f32)> {
        self.section(p)[self.first()..].to_vec()
    }

    /// Overall width along the vessel's major axis, including returning shoulders.
    pub(crate) fn width_m(&self, p: &DishProgram) -> f32 {
        let radius = self.section(p).iter().map(|&(r, _)| r).fold(0., f32::max);
        2. * radius * self.outline_scale() * p.aspect.x
    }

    /// Intersect the horizontal serving plane with the same straight mesh edges.
    pub(super) fn cavity_radius(&self, p: &DishProgram, fill: f32) -> f32 {
        let section = self.section(p);
        let inner = &section[INNER_LIP..=WELL];
        for edge in inner.windows(2) {
            let [(hi, top), (lo, bottom)] = [edge[0], edge[1]];
            if fill >= bottom {
                return lo + (hi - lo) * ((fill - bottom) / (top - bottom)).clamp(0., 1.);
            }
        }
        section[WELL].0
    }

    pub(super) fn build(&self, a: &mut Assembly, tf: Transform, p: &DishProgram, seed: u64) {
        let outline = p.outline();
        let section = self.section(p);
        let first = self.first();
        let profile = &section[first..];
        let base = variants::object_slot(p.surface, seed, None);
        let body = a.part(p.surface, &format!("dishware#finish{base}"));
        if let Some(glaze) = PlateForm::rim_slot(p.surface, seed) {
            let split = HEEL + 3 - first;
            outline.sweep_range(body, profile, 0..split, ProfileJoints::Smooth, tf);
            let inside = a.part(p.surface, &format!("dishware#finish{glaze}"));
            outline.sweep_range(
                inside,
                profile,
                split..profile.len() - 1,
                ProfileJoints::Smooth,
                tf,
            );
        } else {
            outline.sweep(body, profile, ProfileJoints::Smooth, tf);
        }
    }
}
