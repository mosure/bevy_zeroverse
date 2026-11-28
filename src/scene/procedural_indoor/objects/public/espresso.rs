//! Commercial espresso stations. Local +Z faces the operator. Both machines
//! have independent metric envelopes, poses and instance IDs on the counter.
use super::*;

mod grinder;
mod machine;
#[cfg(test)]
mod tests;

pub(super) use grinder::build as build_grinder;
pub(crate) use grinder::power_port;
pub use grinder::Grinder;
pub(super) use machine::build as build_machine;
pub use machine::Machine;
pub(crate) const VERSION: u32 = 301;
const CLASS: &str = "appliance";
const STEEL: &str = "appliance#finish111#uv_local";

pub(crate) fn dimensions(kind: ObjectKind, seed: u64) -> Vec3 {
    let mut r = stream(seed, u64::from_le_bytes(*b"BarStock"));
    if kind == ObjectKind::CoffeeGrinder {
        let width = r.random_range(0.167..0.239);
        return Vec3::new(
            width,
            r.random_range(0.415..0.635),
            width * r.random_range(1.30..1.56),
        );
    }
    let width = match machine::nominal_groups(seed) {
        1 => r.random_range(0.465..0.592),
        2 => r.random_range(0.648..0.853),
        _ => r.random_range(0.882..1.046),
    };
    Vec3::new(
        width,
        r.random_range(0.435..0.555),
        r.random_range(0.490..0.614),
    )
}

fn button(a: &mut Assembly, at: Vec3, radius: f32, stock: Surface) {
    a.part(stock, CLASS).cylinder(
        radius,
        0.004,
        Transform::from_translation(at).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
    );
}
