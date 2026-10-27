//! Dimensioned service furniture. Contact, tray tops and outer bounds are exact.
use super::*;
pub mod bench;
pub mod cart;
pub mod divider;
pub mod drawers;
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
        ObjectKind::Bench => return bench::build(a, o),
        ObjectKind::MobileCart => return cart::build(a, o),
        ObjectKind::RoomDivider => return divider::build(a, o),
        _ => {}
    }
    let p = parameters(o);
    match o.kind {
        ObjectKind::Ottoman => ottoman(a, o, &p),
        _ => unreachable!("non-service furniture"),
    }
}

fn feet(a: &mut Assembly, o: &IndoorObject, height: f32, caster: bool) {
    let s = o.size;
    for x in [-1., 1.] {
        for z in [-1., 1.] {
            let pos = Vec3::new(x * s.x * 0.36, 0., z * s.z * 0.34);
            if caster {
                a.part(Surface::Rubber, o.kind.class_name()).ellipsoid(
                    Vec3::new(height * 0.38, height * 0.38, height * 0.22),
                    Transform::from_translation(pos + Vec3::Y * height * 0.38),
                );
                a.box_part(
                    Surface::Metal,
                    o.kind.class_name(),
                    pos + Vec3::Y * height * 0.78,
                    Vec3::new(height * 0.36, height * 0.44, height * 0.32),
                    0.003,
                );
            } else {
                a.box_part(
                    Surface::WoodEdge,
                    o.kind.class_name(),
                    pos + Vec3::Y * height * 0.5,
                    Vec3::new(0.038, height, 0.038),
                    0.004,
                );
            }
        }
    }
}

/// Categorical construction signature includes only parameters used by this
/// object's geometry, never inactive draws from another object's program.
pub fn construction(o: &IndoorObject) -> String {
    let p = parameters(o);
    match o.kind {
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
                "drawers/{:?}/{}-drawers/base={}/hardware={}/profile={}",
                p.front,
                p.weights.len(),
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
        ObjectKind::Ottoman => format!(
            "ottoman/{:?}/{}",
            p.upholstery,
            if p.roundness < 3. {
                "round"
            } else if p.roundness < 5. {
                "soft-square"
            } else {
                "square"
            }
        ),
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

fn ottoman(a: &mut Assembly, o: &IndoorObject, p: &ServiceProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    let leg = p.feet_height.min(s.y * 0.24);
    feet(a, o, leg, false);
    let cushion = super::super::geometry::CushionProfile {
        roundness: p.roundness,
        crown: 4.2,
        dish: 0.002,
    };
    let top = (s.y - leg) * 0.32;
    a.part(p.upholstery, label).cushion(
        Vec3::new(s.x * 0.98, s.y - leg - top, s.z * 0.98),
        cushion,
        Transform::from_xyz(0., (s.y - top + leg) * 0.5, 0.),
    );
    a.part(p.upholstery, label).cushion(
        Vec3::new(s.x, top, s.z),
        cushion,
        Transform::from_xyz(0., s.y - top * 0.5, 0.),
    );
}
