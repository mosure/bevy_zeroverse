//! Low wall convectors: a closed cover with real inlet/outlet gaps, end caps,
//! brackets and optional hydronic connections. No displacement-only detail.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct HeaterProgram {
    pub surface: Surface,
    pub grille_fraction: f32,
    pub slat_pitch_m: f32,
    pub slat_tilt: f32,
    pub top_slope: f32,
    pub hydronic: bool,
    pub control_left: bool,
}

pub fn dimensions(seed: u64) -> Vec3 {
    let mut r = stream(seed, u64::from_le_bytes(*b"HeatDims"));
    Vec3::new(
        r.random_range(0.55_f32.ln()..3.20_f32.ln()).exp(),
        r.random_range(0.13_f32.ln()..0.46_f32.ln()).exp(),
        r.random_range(0.075..0.19),
    )
}

pub fn parameters(o: &IndoorObject) -> HeaterProgram {
    let mut r = stream(o.seed, u64::from_le_bytes(*b"HeatForm"));
    HeaterProgram {
        surface: [
            Surface::Paint,
            Surface::Paint,
            Surface::Metal,
            Surface::Plastic,
        ][r.random_range(0..4)],
        grille_fraction: r.random_range(0.10..0.35),
        slat_pitch_m: r.random_range(0.018..0.052),
        slat_tilt: r.random_range(-0.45..0.45),
        top_slope: r.random_range(0.02..0.18),
        hydronic: r.random_bool(0.6),
        control_left: r.random_bool(0.5),
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let Vec3 { x: w, y: h, z: d } = o.size;
    let cap = (w * 0.04).clamp(0.018, 0.045);
    let body = w - 2. * cap - 0.004;
    let skin = (d * 0.07).clamp(0.004, 0.012);
    let metal = "appliance";
    // Floor contact brackets support the enclosure, whose underside is open to air.
    for x in [-w * 0.32, w * 0.32] {
        a.box_part(
            Surface::Metal,
            metal,
            Vec3::new(x, h * 0.12, -d * 0.25),
            Vec3::new(cap * 0.65, h * 0.24, d * 0.45),
            0.002,
        );
    }
    for x in [-w * 0.5 + cap * 0.5, w * 0.5 - cap * 0.5] {
        a.box_part(
            p.surface,
            metal,
            Vec3::new(x, h * 0.55, 0.),
            Vec3::new(cap, h * 0.90, d - 2. * skin),
            0.004,
        );
    }
    let inlet = h * 0.13;
    let outlet = h * p.grille_fraction;
    let cover_top = h - outlet;
    a.box_part(
        p.surface,
        metal,
        Vec3::new(0., (inlet + cover_top) * 0.5, d * 0.5 - skin * 0.5),
        Vec3::new(body, cover_top - inlet, skin),
        0.002,
    );
    a.box_part(
        p.surface,
        metal,
        Vec3::new(0., h * 0.55, -d * 0.5 + skin * 0.5),
        Vec3::new(body, h * 0.88, skin),
        0.002,
    );
    // Real spaced slats instead of dark painted stripes; each is bounded by the cover.
    let count = (body / p.slat_pitch_m).floor().clamp(4., 96.) as usize;
    let pitch = body / count as f32;
    for i in 0..count {
        let x = -body * 0.5 + pitch * (i as f32 + 0.5);
        a.part(p.surface, metal).cuboid(
            Vec3::new(pitch * 0.28, outlet * 0.86, d * 0.60),
            0.,
            Transform::from_xyz(x, cover_top + outlet * 0.47, -d * 0.015).with_rotation(
                Quat::from_rotation_y(p.slat_tilt * (pitch / (d * 0.6)).min(1.)),
            ),
        );
    }
    // A recessed heat exchanger can be seen through both vents.
    a.box_part(
        Surface::Metal,
        metal,
        Vec3::new(0., h * 0.48, -d * 0.05),
        Vec3::new(body - 0.004, h * 0.54, d * 0.28),
        0.002,
    );
    let control_x = if p.control_left { -1. } else { 1. } * (w * 0.5 - cap * 0.5);
    a.part(Surface::Plastic, metal).lathe(
        &[
            (0., 0.),
            (cap * 0.26, 0.),
            (cap * 0.26, skin * 0.65),
            (0., skin * 0.65),
        ],
        16,
        Transform::from_xyz(control_x, h * 0.75, d * 0.5 - skin * 1.1)
            .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
    );
    if p.hydronic {
        for x in [-w * 0.32, w * 0.32] {
            a.part(Surface::Chrome, metal).rod(
                Vec3::new(x, 0., d * 0.16),
                Vec3::new(x, h * 0.27, d * 0.16),
                d * 0.045,
            );
        }
    }
    // Header slope is authored independently of the cover and stays below its top.
    a.part(p.surface, metal).cuboid(
        Vec3::new(body, skin, d * 0.30),
        0.,
        Transform::from_xyz(0., h - skin * 1.5 - d * p.top_slope * 0.15, -d * 0.24)
            .with_rotation(Quat::from_rotation_x(p.top_slope)),
    );
}

#[cfg(test)]
mod tests;
