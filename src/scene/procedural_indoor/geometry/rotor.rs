//! One indexed blade blank per rotor. Its copies share tooling and balance;
//! the metric material chart follows each blade's span through the rotation.
use super::{stock::TimberCuts, Geometry};
use bevy::prelude::*;
use std::f32::consts::{PI, TAU};

/// Metric dimensions of one balanced rotor. A blade is cut once and instanced
/// around its axle into the destination mesh.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct RotorBlades {
    pub blades: u8,
    pub phase: f32,
    pub hub_radius: f32,
    pub radius_m: f32,
    pub rotor_y: f32,
    pub blade_width_m: f32,
    pub root_chord_ratio: f32,
    pub shoulder_span: f32,
    pub tip_chord_ratio: f32,
    pub pitch_radians: f32,
    pub tip_pitch_radians: f32,
    pub sweep_m: f32,
    pub thickness_m: f32,
    pub camber_m: f32,
    pub tip_rise_m: f32,
}

const ROWS: usize = 8;
const COLS: usize = 5;
type Skin = [[Vec3; COLS]; ROWS];

fn blade(p: &RotorBlades) -> Geometry {
    let shoulder = p.shoulder_span;
    let stations = [
        0.,
        0.11,
        shoulder * 0.72,
        shoulder,
        (shoulder + 0.86) * 0.5,
        0.86,
        0.95,
        1.,
    ];
    let chord = [0., 0.17, 0.43, 0.77, 1.];
    let root = p.hub_radius * 0.76;
    let span = p.radius_m - root;
    let mut upper = Skin::default();
    let mut lower = Skin::default();
    let mut uv = [[[0.; 2]; COLS]; ROWS];
    let sector = 1.80 * (PI / p.blades as f32).sin();
    for (i, t) in stations.into_iter().enumerate() {
        let r = root + t * span;
        let easing = |x: f32| x * (2. - x);
        let width = if t < shoulder {
            p.root_chord_ratio + (1. - p.root_chord_ratio) * easing(t / shoulder)
        } else {
            let u = (t - shoulder) / (1. - shoulder);
            1. - (1. - p.tip_chord_ratio) * u * u
        } * p.blade_width_m;
        // Even high-solidity rotors must leave a gap between adjacent roots.
        let width = width.min(r * sector);
        let (sin, cos) = p.pitch_radians.lerp(p.tip_pitch_radians, t).sin_cos();
        for (j, v) in chord.into_iter().enumerate() {
            let across = (v - 0.5) * width;
            let crown = 4. * v * (1. - v);
            let camber = p.camber_m * crown * (0.55 + 0.45 * (1. - t));
            let thickness = p.thickness_m * (0.32 + 0.68 * crown) * (1. - 0.28 * t);
            for (skin, side) in [(&mut upper, 0.5), (&mut lower, -0.5)] {
                let lift = camber + side * thickness;
                let z = across * cos + lift * sin + p.sweep_m * t * t;
                let y = p.rotor_y - across * sin + lift * cos + p.tip_rise_m * t * t;
                // Put tip and root sections on concentric arcs; chord and sweep
                // cannot push an outer corner outside the advertised rotor disk.
                skin[i][j] = Vec3::new((r * r - z * z).sqrt(), y, z);
            }
            // The procedural timber shader runs its fibres along V.
            uv[i][j] = [across, t * span];
        }
    }
    let mut g = Geometry::default();
    for (skin, top) in [(&upper, true), (&lower, false)] {
        let base = g.positions.len() as u32;
        for i in 0..ROWS {
            for j in 0..COLS {
                let along = skin[(i + 1).min(ROWS - 1)][j] - skin[i.saturating_sub(1)][j];
                let across = skin[i][(j + 1).min(COLS - 1)] - skin[i][j.saturating_sub(1)];
                let normal = across.cross(along).normalize();
                g.positions.push(skin[i][j].to_array());
                g.normals
                    .push(if top { normal } else { -normal }.to_array());
                g.uvs.push(uv[i][j]);
            }
        }
        for i in 0..ROWS - 1 {
            for j in 0..COLS - 1 {
                let a = base + (i * COLS + j) as u32;
                let b = a + COLS as u32;
                g.indices.extend(if top {
                    [a, a + 1, b + 1, a, b + 1, b]
                } else {
                    [a, b + 1, a + 1, a, b, b + 1]
                });
            }
        }
    }
    for (j, outside) in [(0, Vec3::NEG_Z), (COLS - 1, Vec3::Z)] {
        for i in 0..ROWS - 1 {
            g.stock_quad(
                [upper[i][j], upper[i + 1][j], lower[i + 1][j], lower[i][j]],
                outside,
                (upper[i + 1][j] - upper[i][j]).normalize(),
                Some(Vec2::ZERO),
            );
        }
    }
    for (i, outside) in [(0, Vec3::NEG_X), (ROWS - 1, Vec3::X)] {
        for j in 0..COLS - 1 {
            g.metric_quad(
                [upper[i][j], lower[i][j], lower[i][j + 1], upper[i][j + 1]],
                outside,
            );
        }
    }
    g
}

impl RotorBlades {
    /// `orient` rotates the Y-axis stock into the fixture frame, identically
    /// for positions and normals. Identity preserves ceiling-fan replay.
    pub(crate) fn append(
        &self,
        g: &mut Geometry,
        seed: u64,
        orient: impl Fn([f32; 3]) -> [f32; 3],
    ) {
        let p = self;
        let blank = blade(p);
        let blades = usize::from(p.blades);
        g.positions.reserve(blank.positions.len() * blades);
        g.normals.reserve(blank.normals.len() * blades);
        g.uvs.reserve(blank.uvs.len() * blades);
        g.indices.reserve(blank.indices.len() * blades);
        let mut cuts = TimberCuts::new(seed, u64::from_le_bytes(*b"BladeUVs"));
        for i in 0..p.blades {
            let (s, c) = (p.phase + TAU * f32::from(i) / f32::from(p.blades)).sin_cos();
            let turn = |&[x, y, z]: &[f32; 3]| orient([c * x + s * z, y, c * z - s * x]);
            let offset = g.positions.len() as u32;
            g.positions.extend(blank.positions.iter().map(turn));
            g.normals.extend(blank.normals.iter().map(turn));
            let Vec2 { x: u, y: v } = cuts.origin();
            g.uvs
                .extend(blank.uvs.iter().map(|uv| [uv[0] + u, uv[1] + v]));
            g.indices.extend(blank.indices.iter().map(|i| i + offset));
        }
    }
}
