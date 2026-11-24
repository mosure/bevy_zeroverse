//! Shared seat, rail and back dimensions in the chair's local frame (front is -Z).
use super::*;

pub(crate) const FIT_VERSION: u32 = 217;

/// Movable seating and its table need a common floor for knee access and egress.
pub(crate) fn same_floor(chair: &IndoorObject, table: &IndoorObject) -> bool {
    (chair.position.y - table.position.y).abs() < 0.015
}

pub(crate) fn fitted_envelope(chair: &IndoorObject, surface: Option<&IndoorObject>) -> f32 {
    let mut instance = stream(chair.seed, 0x0053_5446_4954);
    let key = surface.map_or(chair.seed, |o| o.seed);
    let mut product = stream(key, 0x0053_5447_4150);
    let low = product.random_range(0.37..0.424);
    let high = product.random_range(0.765..0.823);
    let gap = product.random_range(0.242..0.317);
    let target = surface.map_or_else(
        || {
            if product.random_bool(0.24) {
                product.random_range(0.57..0.78)
            } else {
                product.random_range(0.405..0.51)
            }
        },
        |top| top.position.y + top.size.y - chair.position.y - gap,
    );
    let height = target.clamp(low, high) + instance.random_range(-0.008..0.008);
    height + allowance(chair)
}

fn allowance(o: &IndoorObject) -> f32 {
    if is_backless(o) {
        0.01
    } else {
        0.32
    }
}

pub(super) fn seat(o: &IndoorObject) -> f32 {
    (o.size.y - allowance(o)).clamp(0.36, 0.835)
}

/// Front rail segment or front arc. Distances are metres from the chair origin.
#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct SeatedSupport {
    pub seat_height_m: f32,
    pub rail_top_m: f32,
    pub rail_half_width_m: f32,
    pub rail_depth_m: f32,
    pub curved_rail: bool,
}

impl SeatedSupport {
    pub(crate) fn matches(self, other: Self) -> bool {
        self.curved_rail == other.curved_rail
            && [
                (self.seat_height_m, other.seat_height_m),
                (self.rail_top_m, other.rail_top_m),
                (self.rail_half_width_m, other.rail_half_width_m),
                (self.rail_depth_m, other.rail_depth_m),
            ]
            .iter()
            .all(|(a, b)| (a - b).abs() < 0.0001)
    }

    pub fn uses_rail(self, stature: f32) -> bool {
        self.seat_height_m > 0.51 * (stature / 1.75)
    }
    pub(crate) fn foot_goal(self, x: f32, stature: f32) -> Vec3 {
        let x = x.clamp(
            -0.76 * self.rail_half_width_m,
            0.76 * self.rail_half_width_m,
        );
        let front = if self.curved_rail {
            (self.rail_depth_m.powi(2) - x.powi(2)).max(0.).sqrt()
        } else {
            self.rail_depth_m
        };
        Vec3::new(x, self.rail_top_m + 0.105, 0.065 * stature / 1.75 - front)
    }
}

impl StoolProgram {
    fn radius(&self, o: &IndoorObject) -> f32 {
        o.size.x.min(o.size.z) * self.radius_fraction
    }
    pub fn support(&self, o: &IndoorObject) -> SeatedSupport {
        let radius = self.radius(o);
        let (width, depth, thickness) = match self.base {
            0 => {
                let t = (self.footrest_height_m - 0.016)
                    / (self.seat_height_m - self.padding_m - 0.016);
                let rr = radius * (self.foot_spread * (1. - t) + 0.60 * t);
                (rr, rr, 0.009)
            }
            1 => (radius * 0.76, radius * 0.76, 0.009),
            _ => (radius * 0.66, radius * self.seat_aspect * 0.65, 0.012),
        };
        SeatedSupport {
            seat_height_m: self.seat_height_m,
            rail_top_m: self.footrest_height_m + thickness,
            rail_half_width_m: width,
            rail_depth_m: depth,
            curved_rail: self.base != 2,
        }
    }

    pub(crate) fn view_boxes(&self, o: &IndoorObject, mut add: impl FnMut(Vec3, Vec3)) {
        let r = self.radius(o);
        let top = self.seat_height_m;
        let under = top - self.padding_m;
        add(
            Vec3::Y * (top - self.padding_m * 0.5),
            Vec3::new(r, self.padding_m * 0.5, r * self.seat_aspect),
        );
        match self.base {
            0 => {
                for i in 0..self.legs {
                    let t = self.phase + i as f32 * TAU / self.legs as f32;
                    let dir = Vec3::new(t.cos(), 0., t.sin());
                    let foot = dir * r * self.foot_spread;
                    let socket = dir * r * 0.60 + Vec3::Y * under;
                    add(
                        (foot + socket) * 0.5,
                        (foot - socket).abs() * 0.5 + Vec3::new(0.016, 0., 0.016),
                    );
                }
            }
            1 => {
                add(Vec3::Y * under * 0.5, Vec3::new(0.032, under * 0.5, 0.032));
                add(
                    Vec3::Y * 0.018,
                    Vec3::new(r * self.foot_spread, 0.018, r * self.foot_spread),
                );
            }
            _ => {
                for x in [-1., 1.] {
                    for z in [-1., 1.] {
                        add(
                            Vec3::new(x * r * 0.66, under * 0.5, z * r * self.seat_aspect * 0.65),
                            Vec3::new(0.016, under * 0.5, 0.016),
                        );
                    }
                }
            }
        }
        if !is_backless(o) {
            let z = r * self.seat_aspect * 0.70;
            add(
                Vec3::new(
                    0.,
                    top + self.back_gap_m + self.back_height_m * 0.5,
                    z + self.back_curve_m * 0.5,
                ),
                Vec3::new(
                    r * 0.83,
                    self.back_height_m * 0.5,
                    self.back_curve_m * 0.5 + 0.028,
                ),
            );
        }
    }
}

pub(crate) fn clearance_boxes(o: &IndoorObject) -> Vec<(Vec3, Vec3)> {
    let seat = seat(o);
    let half = o.size * 0.5;
    let mut boxes = vec![(
        Vec3::new(-half.x, 0., -half.z),
        Vec3::new(half.x, seat + 0.008, half.z),
    )];
    if !is_backless(o) {
        boxes.push((
            Vec3::new(-half.x, seat - 0.06, 0.),
            Vec3::new(half.x, o.size.y, half.z),
        ));
    }
    boxes
}
