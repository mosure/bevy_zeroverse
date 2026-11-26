//! Crowd-control products: webbing, rope, rigid rails and inset infill panels.
use super::*;
mod fitted;
mod legacy;
mod spans;
#[cfg(test)]
mod tests;
pub use fitted::{Infill, Installation};
pub(crate) const FITTED_VERSION: u32 = 259;

#[derive(Clone, Debug, serde::Serialize)]
pub struct BarrierProgram {
    pub construction: u8,
    pub base: u8,
    pub base_radius: f32,
    pub post_radius: f32,
    pub rail_height: f32,
    pub sag: f32,
    pub belt_width: f32,
    pub frame: Surface,
    pub rail: Surface,
}
pub fn parameters(o: &IndoorObject) -> BarrierProgram {
    let mut r = stream(o.seed, 0x5155455545424152);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x515545554546494e);
    BarrierProgram {
        construction: r.random_range(0..4),
        base: r.random_range(0..3),
        base_radius: r.random_range(0.31..0.46),
        post_radius: r.random_range(0.040..0.075),
        rail_height: r.random_range(0.77..0.92),
        sag: r.random_range(0.035..0.22),
        belt_width: r.random_range(0.035..0.065),
        frame: [
            Surface::Chrome,
            Surface::Metal,
            Surface::Plastic,
            Surface::Wood,
        ][f.random_range(0..4)],
        rail: [Surface::Fabric, Surface::FabricAlt, Surface::Rubber][f.random_range(0..3)],
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    if version < FITTED_VERSION {
        return legacy::build(a, o);
    }
    let product = parameters(o);
    let installation = Installation::sample(o, &product);
    installation.build(a, o, &product);
}
