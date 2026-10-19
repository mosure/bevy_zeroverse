//! Continuous neighborhood and construction controls, isolated from room RNG.
use super::*;

#[derive(Debug, serde::Serialize)]
pub(super) struct District {
    pub building_fraction: f32,
    pub parcel_scale: f32,
    pub street_yaw: f32,
    pub paving_width: f32,
    pub paving_pitch: f32,
    pub railing_probability: f32,
}
impl District {
    pub fn sample(seed: u64) -> Self {
        let mut r = stream(seed, 6380);
        Self {
            building_fraction: r.random_range(0.25..0.88),
            parcel_scale: r.random_range(0.9..1.85),
            street_yaw: r.random_range(-std::f32::consts::PI..std::f32::consts::PI),
            paving_width: r.random_range(0.65..2.05),
            paving_pitch: r.random_range(0.28..1.30),
            railing_probability: r.random_range(0.0..0.70),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub(super) struct Building {
    pub levels: usize,
    pub pitch: f32,
    /// Only a selected upper portion steps back; most stories share a footprint.
    pub terrace_level: usize,
    pub setback: f32,
    pub shift: Vec2,
    pub split: f32,
    pub wing_depth: f32,
    pub wing_levels: usize,
    /// 0 flat/parapet, 1 pitched, 2 shed.
    pub roof: u8,
    pub roof_rise: f32,
    pub parapet: f32,
    pub wall: Surface,
    pub roof_material: Surface,
    pub frame: Surface,
    pub bay_pitch: f32,
    pub window_fraction: f32,
    pub window_height: f32,
    pub sill: f32,
    pub frame_width: f32,
    pub stagger: f32,
    pub fins: bool,
    pub hood: f32,
}
impl Building {
    pub fn sample(p: &Parcel, sky_radius: f32) -> Self {
        let mut r = stream(p.seed, 6381);
        let pitch = r.random_range(2.65_f32..3.65);
        // All projections, roofs and services fit within this conservative ball.
        let horizontal = p.center.length() + p.size.length() * 0.5 + 0.30;
        let available = (sky_radius.powi(2) - horizontal.powi(2)).max(0.).sqrt();
        let requested: usize = r.random_range(1..=5);
        let slenderness_limit = (p.size.min_element() * 3.8 / pitch).floor().max(1.) as usize;
        let levels = requested
            .min(slenderness_limit)
            .min(((available - 1.7) / pitch).floor().max(1.) as usize);
        let terrace_level = r.random_range(1..=levels);
        let roof = r.random_range(0..3);
        Self {
            levels,
            pitch,
            terrace_level,
            setback: r.random_range(0.08..0.30),
            shift: Vec2::new(r.random_range(-0.8..0.8), r.random_range(-0.8..0.8)),
            split: if levels > 1 && p.size.x >= 5.8 && r.random_bool(0.60) {
                r.random_range(0.38..0.64)
            } else {
                1.
            },
            wing_depth: r.random_range(0.55..0.96),
            wing_levels: r.random_range(1..=levels.saturating_sub(1).max(1)),
            roof,
            roof_rise: (p.size.x * r.random_range(0.12..0.34)).min(1.45),
            parapet: r.random_range(0.10..0.42),
            wall: [
                Surface::Concrete,
                Surface::Paint,
                Surface::Terracotta,
                Surface::WoodEdge,
                Surface::Metal,
            ][r.random_range(0..5)],
            roof_material: [Surface::Metal, Surface::Terracotta, Surface::Concrete]
                [r.random_range(0..3)],
            frame: if r.random_bool(0.8) {
                Surface::Metal
            } else {
                Surface::WoodEdge
            },
            bay_pitch: r.random_range(0.72..2.35),
            window_fraction: r.random_range(0.36..0.91),
            window_height: r.random_range(0.32..0.76),
            sill: r.random_range(0.10..0.18),
            frame_width: r.random_range(0.018..0.068),
            stagger: if r.random_bool(0.25) {
                r.random_range(-0.18..0.18)
            } else {
                0.
            },
            fins: r.random_bool(0.34),
            hood: if r.random_bool(0.48) {
                r.random_range(0.07..0.22)
            } else {
                0.
            },
        }
    }
}
