//! Nested supermarket handbaskets. Dimensions describe the basket stock at
//! delivery. Every layer in a stack uses the same taper, rim and bearing pitch.
use super::*;

mod shell;
mod stack;
#[cfg(test)]
mod tests;

pub(crate) const VERSION: u32 = 303;
pub(crate) const CLASS: &str = "shopping_basket";
const PLATING: &str = "shopping_basket#finish111#uv_local";
const CADDY: &str = "retail_fixture";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum BasketShell {
    Molded,
    WeldedWire,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Carrier {
    Floor,
    Dolly,
    PushDolly,
}

#[derive(Clone, Debug, Serialize)]
pub struct BasketStack {
    pub shell: BasketShell,
    pub carrier: Carrier,
    pub count: usize,
    /// Outside half-extents of the upper rim in X/Z, before any custom object scaling.
    pub rim_half_m: Vec2,
    pub bottom_half_m: Vec2,
    pub body_height_m: f32,
    pub nesting_pitch_m: f32,
    pub wall_m: f32,
    pub chamfer_fraction: f32,
    pub openings: [usize; 2],
    pub two_slot_rows: bool,
    pub slat_fraction: f32,
    pub coated_wire: bool,
    pub dye_slot: usize,
    pub top_handle_angles_rad: [f32; 2],
    pub wheel_radius_m: f32,
    pub rail_top_m: f32,
    pub push_handle_height_m: f32,
}

impl BasketStack {
    pub fn for_object(o: &IndoorObject) -> Self {
        Self::sample(o.seed, o.finish_seed.unwrap_or(o.seed))
    }

    fn sample(seed: u64, finish: u64) -> Self {
        let mut r = stream(seed, u64::from_le_bytes(*b"HandBskt"));
        let mut paint = stream(finish, u64::from_le_bytes(*b"CarryDye"));
        let shell = if r.random_bool(0.72) {
            BasketShell::Molded
        } else {
            BasketShell::WeldedWire
        };
        let carrier = match r.random_range(0..100) {
            0..=39 => Carrier::Floor,
            40..=74 => Carrier::Dolly,
            _ => Carrier::PushDolly,
        };
        let length = r.random_range(0.38..0.52);
        let half = Vec2::new(length, length * r.random_range(0.57..0.74)) * 0.5;
        let bottom = half * Vec2::new(r.random_range(0.71..0.81), r.random_range(0.68..0.79));
        let height = r.random_range(0.164..0.242);
        let wall = if shell == BasketShell::Molded {
            r.random_range(0.0020..0.0032)
        } else {
            r.random_range(0.0028..0.0040)
        };
        // The inserted body must fit inside the preceding basket even at its
        // shallowest taper. Allow for ribs on both faces of a welded basket.
        let stock_clearance = if shell == BasketShell::WeldedWire {
            2.2 * wall
        } else {
            wall + 0.0028
        };
        let pitch = r
            .random_range(0.029_f32..0.052)
            .max(stock_clearance * height / (half - bottom).min_element());
        let count = if r.random_bool(0.19) {
            1
        } else {
            r.random_range(2..=7)
        };
        let raised = r.random_bool(0.28);
        let angles = if raised {
            let a = r.random_range(1.26..1.48);
            [
                a,
                if r.random_bool(0.65) {
                    r.random_range(1.26..1.48)
                } else {
                    0.
                },
            ]
        } else {
            [0.; 2]
        };
        let wheel = if carrier == Carrier::Floor {
            0.
        } else {
            r.random_range(0.030..0.044)
        };
        Self {
            shell,
            carrier,
            count,
            rim_half_m: half,
            bottom_half_m: bottom,
            body_height_m: height,
            nesting_pitch_m: pitch,
            wall_m: wall,
            chamfer_fraction: r.random_range(0.10..0.21),
            openings: [r.random_range(5..=9), r.random_range(3..=6)],
            two_slot_rows: r.random_bool(0.52),
            slat_fraction: r.random_range(0.20..0.39),
            coated_wire: paint.random_bool(0.23),
            dye_slot: crate::scene::procedural_indoor::materials::basket_resin::sample_slot(
                &mut paint,
            ),
            top_handle_angles_rad: angles,
            wheel_radius_m: wheel,
            rail_top_m: if wheel > 0. { 2. * wheel + 0.030 } else { 0. },
            push_handle_height_m: if carrier == Carrier::PushDolly {
                r.random_range(0.77..0.96)
            } else {
                0.
            },
        }
    }

    pub fn bottom_m(&self, layer: usize) -> f32 {
        self.rail_top_m + layer as f32 * self.nesting_pitch_m
    }

    fn native_size(&self) -> Vec3 {
        let top = self.bottom_m(self.count - 1) + self.body_height_m;
        let rise = self
            .top_handle_angles_rad
            .into_iter()
            .fold(0., f32::max)
            .sin()
            * self.handle_reach();
        let extra = if self.carrier == Carrier::Floor {
            Vec2::new(0.013, 0.013)
        } else {
            Vec2::new(0.026, 0.038)
        };
        let half = self.rim_half_m + extra;
        Vec3::new(
            2. * half.x,
            (top + rise + 0.006).max(self.push_handle_height_m),
            2. * half.y,
        )
    }

    /// Proxy boxes for camera placement. Rays can pass through the pickup
    /// opening, above the stack and between the uprights of a push trolley.
    pub(crate) fn view_boxes(&self, size: Vec3, mut emit: impl FnMut(Vec3, Vec3)) {
        let scale = size / self.native_size();
        let mut put = |c: Vec3, h: Vec3| emit(c * scale, h * scale);
        let base = self.bottom_m(self.count - 1);
        let half = self.rim_half_m;
        if self.shell == BasketShell::Molded {
            let h = self.wall_m * 0.5;
            put(
                Vec3::Y * (base + h),
                Vec3::new(
                    self.bottom_half_m.x * (1. - self.chamfer_fraction),
                    h,
                    self.bottom_half_m.y,
                ),
            );
        }
        for sign in [-1., 1.] {
            let y = base + self.body_height_m - 0.003;
            put(
                Vec3::new(0., y, sign * (half.y - 0.002)),
                Vec3::new(half.x * (1. - self.chamfer_fraction), 0.003, 0.002),
            );
            put(
                Vec3::new(sign * (half.x - 0.002), y, 0.),
                Vec3::new(0.002, 0.003, half.y * (1. - self.chamfer_fraction)),
            );
            if self.carrier == Carrier::PushDolly {
                let height = self.push_handle_height_m - self.rail_top_m;
                put(
                    Vec3::new(
                        sign * (half.x + 0.014),
                        self.rail_top_m + height * 0.5,
                        -half.y - 0.025,
                    ),
                    Vec3::new(0.007, height * 0.5, 0.007),
                );
            }
        }
    }

    fn handle_reach(&self) -> f32 {
        self.rim_half_m.y - 0.005
    }
}

pub(super) fn dimensions(seed: u64) -> Vec3 {
    BasketStack::sample(seed, seed).native_size()
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = BasketStack::for_object(o);
    let scale = o.size / p.native_size();
    if scale == Vec3::ONE {
        stack::build(a, &p);
    } else {
        // Custom envelopes and fixed-size catalog audits use a nonuniform fit.
        // Apply the inverse transpose to normals while preserving metric UVs.
        let mut local = Assembly::default();
        stack::build(&mut local, &p);
        for ((s, k), mut g) in local.parts {
            for v in &mut g.positions {
                *v = (Vec3::from_array(*v) * scale).to_array();
            }
            for n in &mut g.normals {
                *n = (Vec3::from_array(*n) / scale).normalize().to_array();
            }
            a.part(s, &k).append(g);
        }
    }
}
