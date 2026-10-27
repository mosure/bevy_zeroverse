//! Refreshment equipment and hollow vessels; no opaque caps across vessel mouths.
use super::*;
pub mod coffee;
pub mod kettle;

#[derive(Debug, Clone, serde::Serialize)]
pub struct HospitalityProgram {
    pub style: u8,
    pub belly: f32,
    pub neck: f32,
    pub taper: f32,
    pub flutes: u32,
    pub finish: Surface,
}
pub fn parameters(o: &IndoorObject) -> HospitalityProgram {
    let mut rng = stream(o.seed, 5400);
    HospitalityProgram {
        style: rng.random_range(0..4),
        belly: rng.random_range(0.68..0.96),
        neck: rng.random_range(0.28..0.67),
        taper: rng.random_range(0.50..0.90),
        flutes: rng.random_range(12..=32),
        finish: if rng.random_bool(0.45) {
            Surface::Metal
        } else {
            Surface::Plastic
        },
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    match o.kind {
        ObjectKind::CoffeeMaker => coffee::build(a, o),
        ObjectKind::Kettle => kettle::build(a, o),
        ObjectKind::Vase | ObjectKind::Bowl => super::vessels::build(a, o),
        _ => unreachable!("non-hospitality object"),
    }
}
