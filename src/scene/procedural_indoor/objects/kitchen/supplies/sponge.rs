//! Cellulose/foam bodies and optional nonwoven scouring pads, with a shared
//! physical silhouette. Profile variation remains supported and envelope-safe.
use super::*;
#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct SpongeProgram {
    pub footprint: Vec2,
    pub height_fraction: f32,
    pub roundness: f32,
    pub waist: f32,
    pub core_fraction: f32,
    pub scouring: bool,
    pub color: usize,
}
pub fn parameters(o: &IndoorObject) -> SpongeProgram {
    let mut r = stream(o.seed, 0x53504f4e4745464d);
    SpongeProgram {
        footprint: Vec2::new(r.random_range(0.74..1.), r.random_range(0.68..1.)),
        height_fraction: r.random_range(0.78..1.),
        roundness: r.random_range(2.0..12.),
        waist: r.random_range(0.83..1.),
        core_fraction: r.random_range(0.65..0.88),
        scouring: r.random_bool(0.72),
        color: r.random_range(0..8),
    }
}
pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let half = o.size.xz() * p.footprint * 0.5;
    let h = o.size.y * p.height_fraction;
    let core = if p.scouring { h * p.core_fraction } else { h };
    let label = format!("{}#finish{}", o.kind.class_name(), 320 + p.color);
    a.part(Surface::Plastic, &label).housing(
        &[
            (0., half * 0.90),
            (core * 0.15, half),
            (core * 0.50, half * p.waist),
            (core * 0.85, half),
            (core, half * 0.90),
        ],
        p.roundness,
        Transform::IDENTITY,
    );
    if p.scouring {
        let pad = half * 0.89;
        a.part(Surface::Fabric, &label).housing(
            &[
                (core - 0.0001, pad),
                (core + (h - core) * 0.5, pad),
                (h, pad * 0.96),
            ],
            p.roundness,
            Transform::IDENTITY,
        );
    }
}
