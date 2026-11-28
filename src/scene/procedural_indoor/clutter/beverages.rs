//! Hollow vessels with separate rims, closures, liquids and printed sleeves.
use super::*;
pub(crate) mod bottle;
pub mod can;
mod end_plate;
pub(crate) mod glassware;
mod wrap;
use crate::scene::procedural_indoor::geometry::{ProfileJoints, RevolvedOutline};
use std::sync::OnceLock;
pub(crate) mod cups;
use std::f32::consts::{PI, TAU};

#[derive(Debug, serde::Serialize)]
pub struct BeverageProgram {
    pub style: u32,
    pub taper: f32,
    pub fill_fraction: f32,
    pub wall_m: f32,
    pub lid: bool,
    pub shoulder_fraction: f32,
    pub neck_fraction: f32,
    pub heel_fraction: f32,
    pub belly_height: f32,
    pub belly_fraction: f32,
    pub handle_bottom: f32,
    pub handle_top: f32,
    pub handle_shape: f32,
    pub cup_radius_fraction: f32,
    pub cup_sides: u32,
    pub empty: bool,
    pub waist: f32,
    pub waist_height: f32,
    pub ribs: usize,
    pub rib_depth: f32,
}
pub fn parameters(o: &IndoorObject) -> BeverageProgram {
    let mut rng = stream(o.seed, 831);
    let mut profile = stream(o.seed, 7455);
    BeverageProgram {
        belly_height: profile.random_range(0.30..0.68),
        belly_fraction: profile.random_range(0.76..1.20),
        handle_bottom: profile.random_range(0.12..0.30),
        handle_top: profile.random_range(0.72..0.90),
        handle_shape: profile.random_range(0.45..1.80),
        cup_radius_fraction: profile.random_range(0.29..0.39),
        cup_sides: if profile.random_bool(0.20) {
            profile.random_range(8..=12)
        } else {
            32
        },
        empty: profile.random_bool(0.32),
        waist: profile.random_range(0.76..1.0),
        waist_height: profile.random_range(0.32..0.70),
        ribs: profile.random_range(3..=10),
        rib_depth: profile.random_range(0.008..0.055),
        shoulder_fraction: profile.random_range(0.54..0.80),
        neck_fraction: profile.random_range(0.24..0.80),
        heel_fraction: profile.random_range(0.035..0.10),
        style: rng.random_range(0..6),
        taper: if o.kind == ObjectKind::CoffeeCup {
            rng.random_range(1.10..1.28)
        } else {
            rng.random_range(0.70..1.22)
        },
        fill_fraction: rng.random_range(0.32..0.85),
        wall_m: rng.random_range(0.0014..0.0034),
        lid: rng.random_bool(0.72),
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    if o.kind == ObjectKind::SodaCan {
        can::build(a, o);
        return;
    }
    let p = parameters(o);
    match o.kind {
        ObjectKind::Mug | ObjectKind::CoffeeCup => cups::build(a, o, &p, version),
        ObjectKind::WaterBottle => bottle::build(a, o, &p),
        _ => unreachable!(),
    }
}

fn outline(sides: u32) -> &'static RevolvedOutline {
    // Only the five ceramic facets and the round section are sampled.
    static STOCK: [OnceLock<RevolvedOutline>; 6] = [const { OnceLock::new() }; 6];
    let i = if sides == 32 { 5 } else { (sides - 8) as usize };
    STOCK[i].get_or_init(|| {
        if sides == 32 {
            RevolvedOutline::round(32, 0., 0)
        } else {
            RevolvedOutline::polygon(sides)
        }
    })
}

#[cfg(test)]
mod package_tests;
#[cfg(test)]
mod test_geometry;
