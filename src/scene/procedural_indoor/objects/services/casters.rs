//! A matched set of load-bearing casters, installed beneath a cart or pedestal.
use super::*;

mod mesh;
#[cfg(test)]
mod tests;

pub(crate) const INTRODUCED: u32 = 208;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) enum Wheel {
    Solid,
    Spoked(u8),
    Twin,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Mount {
    pub pivot_xz_m: Vec2,
    pub heading_rad: f32,
    pub fixed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spoke_phase_rad: Option<f32>,
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct CasterSet {
    pub wheel: Wheel,
    pub radius_m: f32,
    pub tread_width_m: f32,
    pub core_fraction: f32,
    pub swivel_trail_m: f32,
    pub mount_height_m: f32,
    pub frame: Surface,
    pub core: Surface,
    pub mounts: [Mount; 4],
    #[serde(skip)]
    post_mount: bool,
}

impl CasterSet {
    pub(crate) fn cart(o: &IndoorObject, p: &cart::CartProgram) -> Self {
        Self::fit(o, p.post_span * o.size.xz(), p.wheel_radius, None)
    }

    pub(crate) fn pedestal(o: &IndoorObject, p: &drawers::DrawerProgram) -> Self {
        Self::fit(o, p.foot_span * o.size.xz(), p.foot / 2.65, Some(p.foot))
    }

    pub(in super::super) fn laundry_cart(o: &IndoorObject, span: Vec2, radius: f32) -> Self {
        Self::fit(o, span, radius, None)
    }

    fn fit(o: &IndoorObject, span: Vec2, requested: f32, cabinet_floor: Option<f32>) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"CastrSet"));
        let mut f = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"CastrMat"),
        );
        let cart = cabinet_floor.is_none();
        let (twin, spoked) = if cart { (20, 58) } else { (62, 75) };
        let wheel = match r.random_range(0..100) {
            n if n < twin => Wheel::Twin,
            n if n < spoked => Wheel::Spoked(r.random_range(4..=6)),
            _ => Wheel::Solid,
        };
        let width = if wheel == Wheel::Twin {
            r.random_range(0.28..0.42)
        } else {
            r.random_range(0.48..0.78)
        };
        let trail: f32 = r.random_range(0.26..0.64);
        let core_fraction = r.random_range(0.59..0.79);
        // A complete turn must fit the placement footprint even if a swivel is
        // parked sideways. Include the outside axle ends as well as the tyre.
        let half_axle = Self::axle_half_width(wheel, width);
        let sweep = (1. + trail).hypot(half_axle);
        let half = o.size.xz() * 0.5;
        let (radius_m, span, center, mount_height_m) = if let Some(height) = cabinet_floor {
            // Pedestal floors are shallower than their overhanging top and are
            // set back 25 mm. Bolt all four plates to the actual carcass floor.
            let floor = Vec2::new(o.size.x * 0.98, o.size.z - 0.065) * 0.5;
            let center = Vec2::new(0., -0.025);
            let room = half - center.abs();
            let radius = requested.min(room.min_element() * 0.45 / sweep);
            let limit =
                (floor - Vec2::splat(radius * 0.45)).min(room - Vec2::splat(radius * sweep));
            (radius, span.min(limit), center, height)
        } else {
            // Cart post positions also define the cargo bays. Match the wheels
            // to those posts without moving a shelf, its contents or its handle.
            let radius = requested.min((half - span).min_element() / sweep);
            (
                radius,
                span,
                Vec2::ZERO,
                radius * r.random_range(2.48..2.78),
            )
        };
        let long_x = half.x >= half.y;
        let axial = if long_x { FRAC_PI_2 } else { 0. };
        let rigid_end = cart && r.random_bool(0.38);
        let positive = r.random_bool(0.5);
        let heading = if rigid_end {
            axial + if r.random_bool(0.5) { PI } else { 0. } + r.random_range(-0.35..0.35)
        } else {
            r.random_range(-PI..PI)
        };
        let mut spokes = match wheel {
            Wheel::Spoked(count) => Some((
                stream(o.seed, u64::from_le_bytes(*b"SpokPhas")),
                2. * PI / f32::from(count),
            )),
            _ => None,
        };
        let mounts = std::array::from_fn(|i| {
            let sign = Vec2::new(
                if i < 2 { -1. } else { 1. },
                if i % 2 == 0 { -1. } else { 1. },
            );
            let fixed = rigid_end && ((if long_x { sign.x } else { sign.y }) > 0.) == positive;
            let yaw = if fixed {
                axial
            } else if r.random_bool(0.23) {
                r.random_range(-PI..PI)
            } else {
                heading + r.random_range(-0.48..0.48)
            };
            Mount {
                pivot_xz_m: center + sign * span,
                heading_rad: yaw,
                fixed,
                spoke_phase_rad: spokes
                    .as_mut()
                    .map(|(r, period)| r.random_range(0.0..*period)),
            }
        });
        Self {
            wheel,
            radius_m,
            tread_width_m: radius_m * width,
            core_fraction,
            swivel_trail_m: radius_m * trail,
            mount_height_m,
            mounts,
            post_mount: cart,
            frame: if f.random_bool(0.75) {
                Surface::Metal
            } else {
                Surface::Chrome
            },
            core: if f.random_bool(0.70) {
                Surface::Plastic
            } else {
                Surface::Metal
            },
        }
    }

    /// Insert the upright into the bearing so its bottom cap lies inside
    /// the housing, even when both components have the same nominal radius.
    pub(in super::super) fn post_bottom_m(&self) -> f32 {
        self.mount_height_m - 0.05 * self.radius_m
    }

    // Dimensions in wheel radii. Twin treads flank a central carrier; a
    // single wheel runs between two outside fork cheeks.
    fn axle_half_width(wheel: Wheel, width: f32) -> f32 {
        if wheel == Wheel::Twin {
            0.19 + width + 0.07
        } else {
            width * 0.5 + 0.24
        }
    }
}
