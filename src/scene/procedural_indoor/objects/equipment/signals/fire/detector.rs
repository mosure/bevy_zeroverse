//! Ceiling detector with open radial inlets and a separate inner sensing chamber.
//! Y increases from the protected face toward the ceiling mounting plate.
use super::*;

pub(crate) const CHAMBER: usize = FINISH_BASE + 5;
pub(crate) const FIRST_LENS: usize = CHAMBER + 1;
pub(crate) const LAST_LENS: usize = FIRST_LENS + 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum DetectorFace {
    CompactButton,
    HushPlate,
    GuardedProbe,
    Flush,
}

#[derive(Debug, serde::Serialize)]
pub struct DetectorProgram {
    pub face: DetectorFace,
    pub finish: usize,
    pub lens_finish: usize,
    pub radius_m: f32,
    pub height_m: f32,
    pub mount_y_m: f32,
    pub skirt_y_m: f32,
    pub face_radius_m: f32,
    pub face_y_m: f32,
    pub inlet_bottom_m: f32,
    pub inlet_top_m: f32,
    pub lower_lip_radius_m: f32,
    pub upper_lip_radius_m: f32,
    pub chamber_radius_m: f32,
    pub inlets: usize,
    /// Free angular fraction at either end of each bay.
    pub opening_fraction: f32,
    pub rib_rake_rad: f32,
    pub phase_rad: f32,
    pub button_radius_m: f32,
    pub button_center_m: Vec2,
    pub lens_center_m: Vec2,
    pub lens_radius_m: f32,
    pub guard_arms: usize,
}

/// Allowance for the concealed straight mounting sleeve in plaster. Even a
/// shallow housing retains a distinct faceted body below this section.
pub(crate) fn sleeve_depth(size: Vec3) -> f32 {
    (size.x.min(size.z) * 0.075).min(size.y * 0.24)
}

pub fn parameters(o: &IndoorObject) -> DetectorProgram {
    let case = super::program(o);
    let s = o.size * case.body_fraction;
    let r = s.x.min(s.z) * 0.49;
    let base = o.size.y - s.y;
    let mut rng = stream(o.seed, u64::from_le_bytes(*b"AirInlet"));
    let face = [
        DetectorFace::CompactButton,
        DetectorFace::HushPlate,
        DetectorFace::GuardedProbe,
        DetectorFace::Flush,
    ][case.style as usize];
    let deck = if face == DetectorFace::GuardedProbe {
        rng.random_range(0.30..0.42)
    } else {
        rng.random_range(0.065..0.11)
    };
    let skirt = case.skirt_height.min(1. - sleeve_depth(o.size) / s.y);
    let lower = deck + (skirt - deck) * rng.random_range(0.23..0.40);
    let upper = lower + (skirt - lower) * rng.random_range(0.52..0.82);
    let inlets = case.slots + 4;
    let face_r = r * case.sensor_face_fraction;
    let phase_rad = rng.random_range(0.0..TAU);
    let guard_arms = if face == DetectorFace::GuardedProbe {
        rng.random_range(3..=5)
    } else {
        0
    };
    // Put the lens between the guard anchors on a heat detector.
    let axis = Vec2::from_angle(if guard_arms > 0 {
        phase_rad + PI / guard_arms as f32
    } else {
        rng.random_range(0.0..TAU)
    });
    let lower_r = r * case.crown_fraction;
    DetectorProgram {
        face,
        finish: case.finish,
        lens_finish: FIRST_LENS + rng.random_range(0..3),
        radius_m: r,
        height_m: s.y,
        mount_y_m: o.size.y,
        skirt_y_m: base + s.y * skirt,
        face_radius_m: face_r,
        face_y_m: base + s.y * deck,
        inlet_bottom_m: base + s.y * lower,
        inlet_top_m: base + s.y * upper,
        lower_lip_radius_m: lower_r,
        upper_lip_radius_m: r * rng.random_range(0.93..0.98),
        chamber_radius_m: lower_r * rng.random_range(0.67..0.83),
        inlets,
        opening_fraction: rng.random_range(0.49..0.83),
        rib_rake_rad: (TAU / inlets as f32) * rng.random_range(-0.27..0.27),
        phase_rad,
        button_radius_m: face_r
            * match face {
                DetectorFace::CompactButton => rng.random_range(0.25..0.44),
                DetectorFace::HushPlate => rng.random_range(0.52..0.68),
                _ => 0.,
            },
        button_center_m: -axis * face_r * rng.random_range(0.12..0.22),
        lens_center_m: axis * face_r * rng.random_range(0.67..0.83),
        lens_radius_m: face_r * rng.random_range(0.065..0.12),
        guard_arms,
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let label = o.kind.class_name();
    let body = a.part(Surface::Plastic, &format!("{label}#finish{}", p.finish));
    let segments = 2 * p.inlets as u32;
    let (lo, hi) = (p.inlet_bottom_m, p.inlet_top_m);
    let (inner, lower, upper) = (
        p.chamber_radius_m,
        p.lower_lip_radius_m,
        p.upper_lip_radius_m,
    );
    // Each lip turns inward to meet the drum. Skip the skin across the inlets.
    body.lathe(
        &[
            (0., p.face_y_m),
            (p.face_radius_m, p.face_y_m),
            (lower * 0.97, lo - p.height_m * 0.055),
            (lower, lo),
            (inner, lo),
        ],
        segments,
        Transform::IDENTITY,
    );
    body.lathe(
        &[
            (inner, hi),
            (upper, hi),
            (p.radius_m, p.skirt_y_m),
            (p.radius_m, p.mount_y_m),
            (0., p.mount_y_m),
        ],
        segments,
        Transform::IDENTITY,
    );
    if p.face == DetectorFace::HushPlate {
        let mid = (lo + hi) * 0.5;
        let lip = (hi - lo) * 0.074;
        body.lathe(
            &[
                (inner * 0.99, mid - lip),
                (lower * 0.96, mid - lip),
                (lower * 0.98, mid),
                (lower * 0.95, mid + lip),
                (inner * 0.99, mid + lip),
            ],
            segments,
            Transform::IDENTITY,
        );
    }
    for bay in 0..p.inlets {
        inlet_rib(body, &p, bay);
    }
    if p.button_radius_m > 0. {
        let r = p.button_radius_m;
        let y = p.face_y_m;
        let depth = (p.height_m * 0.044).min(r * 0.19);
        let at = p.button_center_m;
        body.lathe(
            &[
                (0., y - depth),
                (r * 0.78, y - depth),
                (r, y - depth * 0.38),
                (r, y + depth * 0.12),
            ],
            16,
            Transform::from_xyz(at.x, 0., at.y),
        );
    }
    if p.face == DetectorFace::GuardedProbe {
        heat_guard(
            a.part(
                Surface::Metal,
                &format!("{label}#finish{}", FINISH_BASE + 4),
            ),
            &p,
        );
    }
    a.part(Surface::Plastic, &format!("{label}#finish{CHAMBER}"))
        .lathe(&[(inner, lo), (inner, hi)], segments, Transform::IDENTITY);
    // A pigmented indicator is seated inside the flat portion of every face.
    // Its mild emission is independent of the room's ceiling photometry.
    let lens = p.lens_radius_m;
    let at = p.lens_center_m;
    let protrusion = (lens * 0.45).min(p.height_m * 0.044);
    a.part(
        Surface::Plastic,
        &format!("{label}#finish{}", p.lens_finish),
    )
    .lathe(
        &[
            (0., -protrusion),
            (lens * 0.64, -protrusion * 0.84),
            (lens, -protrusion * 0.18),
            (lens, protrusion * 0.16),
        ],
        12,
        Transform::from_xyz(at.x, p.face_y_m, at.y),
    );
}

fn radial(angle: f32) -> Vec3 {
    let (s, c) = angle.sin_cos();
    Vec3::new(c, 0., s)
}

fn inlet_rib(body: &mut Geometry, p: &DetectorProgram, bay: usize) {
    let pitch = TAU / p.inlets as f32;
    let center = p.phase_rad + pitch * bay as f32;
    let width = pitch * (1. - p.opening_fraction) * 0.5;
    let foot = [radial(center - width), radial(center + width)];
    let head = [
        radial(center - width + p.rib_rake_rad),
        radial(center + width + p.rib_rake_rad),
    ];
    let ring = |dirs: [Vec3; 2], radius: f32, y: f32| dirs.map(|d| (d * radius).with_y(y));
    // Step the ribs inside the polygonal lip's apothem even at the coarsest size.
    let f = ring(foot, p.lower_lip_radius_m * 0.973, p.inlet_bottom_m);
    let h = ring(head, p.upper_lip_radius_m * 0.973, p.inlet_top_m);
    let inside = p.chamber_radius_m * 0.96;
    let fi = ring(foot, inside, p.inlet_bottom_m);
    let hi = ring(head, inside, p.inlet_top_m);
    for points in [
        [f[0], h[0], h[1], f[1]],
        [fi[0], hi[0], h[0], f[0]],
        [f[1], h[1], hi[1], fi[1]],
    ] {
        let [a, b, c, d] = points;
        let n0 = (b - a).cross(c - a).normalize();
        let n1 = (c - a).cross(d - a).normalize();
        let joint = (n0 + n1).normalize();
        let (width, rise) = (d.distance(a), b.distance(a));
        body.vertex_quad(
            points,
            [joint, n0, joint, n1],
            [[0., 0.], [0., rise], [width, rise], [width, 0.]],
        );
    }
}

fn heat_guard(g: &mut Geometry, p: &DetectorProgram) {
    let deck = p.face_y_m;
    let gap = deck - (p.mount_y_m - p.height_m);
    let floor = deck - gap * 0.94;
    let radius = p.face_radius_m * 0.47;
    let wire = (radius * 0.105).min(gap * 0.078);
    g.lathe(
        &[
            (radius - wire, floor),
            (radius + wire, floor),
            (radius + wire, floor + wire * 1.5),
            (radius - wire, floor + wire * 1.5),
            (radius - wire, floor),
        ],
        20,
        Transform::IDENTITY,
    );
    for i in 0..p.guard_arms {
        let d = radial(p.phase_rad + TAU * i as f32 / p.guard_arms as f32);
        g.tube(
            &[
                (d * (radius * 1.60)).with_y(deck + wire),
                (d * (radius * 1.47)).with_y(deck - gap * 0.52),
                (d * radius).with_y(floor + wire * 0.76),
            ],
            wire * 0.64,
            6,
        );
    }
    let tip = deck - gap * 0.64;
    g.lathe(
        &[
            (0., tip),
            (radius * 0.10, tip),
            (radius * 0.23, tip + gap * 0.14),
            (radius * 0.28, deck + wire),
        ],
        12,
        Transform::IDENTITY,
    );
}
