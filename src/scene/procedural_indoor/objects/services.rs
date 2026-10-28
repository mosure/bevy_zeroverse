//! Dimensioned service furniture. Contact, tray tops and outer bounds are exact.
use super::*;
pub mod bench;
pub mod cart;
pub mod divider;
pub mod drawers;
pub mod heater;
pub mod ottoman;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ServiceProgram {
    pub levels: usize,
    pub drawers: usize,
    pub frame: Surface,
    pub upholstery: Surface,
    pub roundness: f32,
    pub feet_height: f32,
    pub slats: bool,
    pub backed: bool,
    pub panel_fraction: f32,
    pub drawer_bias: f32,
    pub handle_fraction: f32,
}

pub fn parameters(o: &IndoorObject) -> ServiceProgram {
    let mut rng = stream(o.seed, 5200);
    let mut detail = stream(o.seed, 7495);
    ServiceProgram {
        drawer_bias: detail.random_range(-0.55..0.55),
        handle_fraction: detail.random_range(0.13..0.33),
        levels: rng.random_range(2..=3),
        drawers: rng.random_range(2..=5),
        frame: if rng.random_bool(0.45) {
            Surface::Wood
        } else {
            Surface::Metal
        },
        upholstery: if rng.random_bool(0.28) {
            Surface::Leather
        } else {
            Surface::FabricAlt
        },
        roundness: rng.random_range(2.0..8.0),
        feet_height: rng.random_range(0.055..0.12),
        slats: rng.random_bool(0.50),
        backed: rng.random_bool(0.45),
        panel_fraction: rng.random_range(0.58..0.90),
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    match o.kind {
        ObjectKind::DrawerUnit => return drawers::build(a, o),
        ObjectKind::BaseboardHeater => return heater::build(a, o),
        ObjectKind::Bench => return bench::build(a, o),
        ObjectKind::MobileCart => return cart::build(a, o),
        ObjectKind::RoomDivider => return divider::build(a, o),
        _ => {}
    }
    match o.kind {
        ObjectKind::Ottoman => ottoman::build(a, o),
        _ => unreachable!("non-service furniture"),
    }
}

/// Categorical construction signature includes only parameters used by this
/// object's geometry, never inactive draws from another object's program.
pub fn construction(o: &IndoorObject) -> String {
    match o.kind {
        ObjectKind::BaseboardHeater => {
            let p = heater::parameters(o);
            format!(
                "heater/{:?}/hydronic={}/left={}",
                p.surface, p.hydronic, p.control_left
            )
        }
        ObjectKind::MobileCart => {
            let p = cart::parameters(o);
            format!(
                "cart/{:?}/{:?}/{}-levels/handles={}",
                p.frame,
                p.trays,
                p.shelf_tops.len(),
                p.handles
            )
        }
        ObjectKind::DrawerUnit => {
            let p = drawers::parameters(o);
            format!(
                "drawers/{:?}/{}-drawers/{}-banks/base={}/hardware={}/profile={}",
                p.front,
                p.weights.len(),
                p.banks.len(),
                p.base,
                p.hardware,
                p.profile
            )
        }
        ObjectKind::Bench => {
            let p = bench::parameters(o);
            format!(
                "bench/{:?}/base={}/slats={}/back={}/arms={}",
                p.frame, p.base, p.slats, p.backed, p.arms
            )
        }
        ObjectKind::Ottoman => {
            let p = ottoman::parameters(o);
            format!(
                "ottoman/{:?}/{:?}/base={:?}/cap={}",
                p.body,
                p.top,
                p.base,
                p.cap_fraction > 0.
            )
        }
        ObjectKind::RoomDivider => {
            let p = divider::parameters(o);
            format!(
                "divider/{:?}/{}-panels/base={}/slats={}",
                p.panel,
                p.weights.len(),
                p.base,
                p.slats
            )
        }
        _ => unreachable!("non-service construction"),
    }
}
