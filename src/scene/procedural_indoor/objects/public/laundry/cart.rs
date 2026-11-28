//! Rolling linen stock. The manufacturer's basket and chassis are shared within
//! a fleet. Contents and swivel positions belong to the individual cart.
use super::*;
use crate::scene::procedural_indoor::materials::garment_stock;
use crate::scene::procedural_indoor::objects::services::casters;
mod basket;
mod linen;
#[cfg(test)]
mod tests;

pub(crate) const VERSION: u32 = 314;
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum Basket {
    Wire,
    Canvas,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum Load {
    Empty,
    LooseLaundry,
    FoldedLinen,
}

struct Product {
    basket: Basket,
    size: Vec3,
    rim: f32,
    hanging_rail: bool,
}

fn product(seed: u64) -> Product {
    let mut r = stream(seed, u64::from_le_bytes(*b"WashCart"));
    let (basket, length, width, rim) = if r.random_bool(0.58) {
        (
            Basket::Wire,
            r.random_range(0.61..0.91),
            r.random_range(0.46..0.65),
            r.random_range(0.65..0.88),
        )
    } else {
        (
            Basket::Canvas,
            r.random_range(0.53..1.04),
            r.random_range(0.46..0.67),
            r.random_range(0.69..0.98),
        )
    };
    let hanging_rail = basket == Basket::Wire && r.random_bool(0.28);
    let height = if hanging_rail {
        rim + r.random_range(0.59..0.89)
    } else {
        rim
    };
    Product {
        basket,
        size: Vec3::new(length, height, width),
        rim,
        hanging_rail,
    }
}

pub fn dimensions(stock: u64) -> Vec3 {
    product(stock).size
}

#[derive(Debug, serde::Serialize)]
pub struct LaundryCart {
    pub basket: Basket,
    pub hanging_rail: bool,
    pub rim_height_m: f32,
    pub base_height_m: f32,
    pub rim_half_m: Vec2,
    pub base_half_m: Vec2,
    pub frame: Surface,
    pub rim_radius_m: f32,
    pub wire_pitch_m: f32,
    pub liner_slot: usize,
    pub load: Load,
    pub fill_fraction: f32,
    pub(crate) casters: casters::CasterSet,
}

impl LaundryCart {
    pub fn for_object(o: &IndoorObject) -> Self {
        let seed = o.finish_seed.unwrap_or(o.seed);
        let stock = product(seed);
        let mut r = stream(seed, u64::from_le_bytes(*b"HamprKit"));
        let mut contents = stream(o.seed, u64::from_le_bytes(*b"LinenUse"));
        let rim_radius_m = r.random_range(0.009..0.016);
        let rim_half_m = o.size.xz() * 0.5 - Vec2::splat(rim_radius_m + 0.002);
        let base_half_m =
            rim_half_m * Vec2::new(r.random_range(0.74..0.88), r.random_range(0.69..0.84));
        let span = o.size.xz() * Vec2::new(r.random_range(0.31..0.37), r.random_range(0.28..0.34));
        let casters = casters::CasterSet::laundry_cart(o, span, r.random_range(0.042..0.068));
        let rim_height_m = o.size.y * (stock.rim / stock.size.y) - rim_radius_m;
        let base_height_m = casters.mount_height_m
            + rim_height_m
                * if stock.basket == Basket::Wire {
                    r.random_range(0.12..0.24)
                } else {
                    r.random_range(0.048..0.11)
                };
        let load = match contents.random_range(0..100) {
            0..=31 => Load::Empty,
            32..=69 => Load::LooseLaundry,
            _ => Load::FoldedLinen,
        };
        Self {
            basket: stock.basket,
            hanging_rail: stock.hanging_rail,
            rim_height_m,
            base_height_m,
            rim_half_m,
            base_half_m,
            rim_radius_m,
            casters,
            frame: if r.random_bool(0.61) {
                Surface::Chrome
            } else {
                Surface::Metal
            },
            wire_pitch_m: r.random_range(0.043..0.078),
            liner_slot: garment_stock::sample_slot(&mut r),
            load,
            fill_fraction: if load == Load::Empty {
                0.
            } else {
                contents.random_range(0.19..0.90)
            },
        }
    }
}

const LABEL: &str = "mobile_cart";
pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = LaundryCart::for_object(o);
    p.casters.build(a, LABEL);
    let frame = a.part(p.frame, LABEL);
    let bottom = p.casters.post_bottom_m();
    let radius = p.rim_radius_m;
    let rim = p.rim_height_m;
    // Short corner chamfers keep the top open and provide a continuous push rail.
    basket::ring(frame, p.rim_half_m, rim, radius);
    for m in &p.casters.mounts {
        let v = m.pivot_xz_m;
        let sign = v.signum();
        let pivot = Vec3::new(v.x, bottom, v.y);
        let foot = p.base_half_m * sign * Vec2::new(0.88, 1.);
        match p.basket {
            Basket::Wire => {
                frame.tube(
                    &[pivot, Vec3::new(foot.x, p.base_height_m, foot.y)],
                    radius * 0.82,
                    8,
                );
            }
            Basket::Canvas => {
                // Crossed scissors run outside the bag. The ends meet the rail on
                // its straight sides and separated sidemembers pass the pivot.
                let end = p.rim_half_m * Vec2::new(-0.88 * sign.x, sign.y);
                let lower = Vec3::new(
                    v.x,
                    bottom + 0.023,
                    end.y - sign.y * if sign.x > 0. { 0.018 } else { 0. },
                );
                let upper = Vec3::new(end.x, rim, lower.z);
                if sign.x > 0. {
                    let t = v.x / (v.x - end.x);
                    let center = lower.lerp(upper, t);
                    frame.tube(
                        &[
                            center.with_z((p.rim_half_m.y - 0.029) * sign.y),
                            center.with_z((p.rim_half_m.y + 0.006) * sign.y),
                        ],
                        radius * 0.94,
                        8,
                    );
                }
                frame.tube(&[pivot, lower, upper], radius * 0.73, 8);
                if sign.x > 0. {
                    // The inboard scissor attaches to the rim through a transverse
                    // pin. A tube bent through a sub-radius elbow would self-fold.
                    frame.tube(&[upper, upper.with_z(end.y)], radius * 0.64, 8);
                }
            }
        }
    }
    if p.hanging_rail {
        let x = p.rim_half_m.x;
        let top = o.size.y - radius;
        frame.tube(
            &[
                Vec3::new(-x, rim, 0.),
                Vec3::new(-x, top - 0.045, 0.),
                Vec3::new(-x + 0.045, top, 0.),
                Vec3::new(x - 0.045, top, 0.),
                Vec3::new(x, top - 0.045, 0.),
                Vec3::new(x, rim, 0.),
            ],
            radius * 0.88,
            10,
        );
    }
    let floor = match p.basket {
        Basket::Wire => basket::wire(frame, &p),
        Basket::Canvas => basket::canvas(a, &p),
    };
    linen::build(a, o, &p, floor);
}
