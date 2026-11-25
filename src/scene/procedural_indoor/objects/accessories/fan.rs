//! Desk fans vary head pose, pedestal/yoke construction and physical guard/blades.
use super::*;
mod grille;
mod legacy;
mod product;
pub use product::FanHead;
pub(crate) const FORMED_ROTOR_VERSION: u32 = 243;

#[derive(Debug, Clone, serde::Serialize)]
pub struct FanProgram {
    pub radius: f32,
    pub depth: f32,
    pub center: Vec3,
    pub pitch: f32,
    pub yaw: f32,
    pub blades: usize,
    pub blade_phase: f32,
    pub blade_chord: f32,
    pub blade_sweep: f32,
    pub spokes: usize,
    pub rings: usize,
    pub wire: f32,
    pub base: Vec3,
    pub yoke: bool,
    pub finish: Surface,
    pub blade_finish: Surface,
    pub guard_finish: Surface,
}
pub fn parameters(o: &IndoorObject) -> FanProgram {
    parameters_at(o, bevy_zeroverse_capture::GENERATOR_VERSION)
}

fn parameters_at(o: &IndoorObject, version: u32) -> FanProgram {
    let mut r = stream(o.seed, 0x4445534b46414e50);
    let s = o.size;
    let pitch = r.random_range(-0.32_f32..0.48);
    let yaw = r.random_range(-0.34_f32..0.34);
    let radius = (s.x * r.random_range(0.32_f32..0.43)).min(s.y * r.random_range(0.23..0.34));
    let depth = s.z * r.random_range(0.24..0.43);
    let wire = s.x.min(s.z) * r.random_range(0.005..0.009);
    let q = Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch);
    // A conservative rotated head box fits before geometry is built.
    let ext = q.mul_vec3(Vec3::X).abs() * radius
        + q.mul_vec3(Vec3::Y).abs() * radius
        + q.mul_vec3(Vec3::Z).abs() * depth * 0.5
        + Vec3::splat(wire);
    let scale = ((s.x * 0.48) / ext.x).min((s.z * 0.48) / ext.z).min(1.);
    let radius = radius * scale;
    let depth = depth * scale;
    let ey = ext.y * scale + wire;
    let cy = s.y * r.random_range(0.61..0.72);
    let center = Vec3::new(0., cy.min(s.y - ey - 0.002), 0.);
    let mut p = FanProgram {
        radius,
        depth,
        center,
        pitch,
        yaw,
        blades: r.random_range(3..=7),
        blade_phase: r.random_range(-PI..PI),
        blade_chord: r.random_range(0.13..0.24),
        blade_sweep: r.random_range(0.12..0.42),
        spokes: r.random_range(12..=24),
        rings: r.random_range(1..=3),
        wire,
        base: Vec3::new(
            s.x * r.random_range(0.54..0.83),
            s.y * r.random_range(0.035..0.075),
            s.z * r.random_range(0.60..0.94),
        ),
        yoke: r.random_bool(0.42),
        finish: [Surface::Metal, Surface::Plastic, Surface::Chrome][r.random_range(0..3)],
        blade_finish: [Surface::Plastic, Surface::Metal][r.random_range(0..2)],
        guard_finish: Surface::Metal,
    };
    if version >= FORMED_ROTOR_VERSION {
        let mut finish = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"FanCoats"),
        );
        p.finish = if finish.random_bool(0.31) {
            Surface::Metal
        } else {
            Surface::Plastic
        };
        p.guard_finish =
            [Surface::Metal, Surface::Chrome, Surface::Plastic][finish.random_range(0..3)];
        p.blade_finish = if finish.random_bool(0.67) {
            Surface::Plastic
        } else {
            Surface::Metal
        };
    } else {
        p.guard_finish = p.finish;
    }
    p
}
impl FanProgram {
    pub(crate) fn head_transform(&self) -> Transform {
        Transform::from_translation(self.center)
            .with_rotation(Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch))
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    if version < FORMED_ROTOR_VERSION {
        legacy::build(a, o);
        return;
    }
    let p = parameters(o);
    FanHead::sample(o, &p).build(a, o, &p);
}
#[cfg(test)]
mod tests;
