//! Exterior walls share one aperture program for meshes, attachments and audits.
mod geometry;
mod legacy;
#[cfg(test)]
mod tests;

use super::super::{
    layout::{stream, IndoorManifest},
    materials::Surface,
    objects::Assembly,
};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub(super) use geometry::shell;
pub(crate) use geometry::{backdrop, glazing_at};
pub use geometry::{clipped_wall_box, solid_rectangles};

/// The +Z wall adjoins the furnished neighboring room and remains internal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FacadeSide {
    Left,
    Right,
    Rear,
}
impl FacadeSide {
    pub const ALL: [Self; 3] = [Self::Left, Self::Right, Self::Rear];
    pub fn span(self, size: Vec3) -> f32 {
        if self == Self::Rear {
            size.x
        } else {
            size.z
        }
    }
    pub fn yaw(self) -> f32 {
        match self {
            Self::Left => std::f32::consts::FRAC_PI_2,
            Self::Right => -std::f32::consts::FRAC_PI_2,
            Self::Rear => 0.,
        }
    }
    /// Local X follows the wall, Y is up, and +Z points into the primary room.
    pub fn transform(self, size: Vec3) -> Transform {
        Transform::from_translation(match self {
            Self::Left => Vec3::new(-size.x * 0.5, 0., 0.),
            Self::Right => Vec3::new(size.x * 0.5, 0., 0.),
            Self::Rear => Vec3::new(0., 0., -size.z * 0.5),
        })
        .with_rotation(Quat::from_rotation_y(self.yaw()))
    }
    pub fn local(self, size: Vec3, point: Vec3) -> Vec3 {
        let tf = self.transform(size);
        tf.rotation.inverse() * (point - tf.translation)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowOpening {
    /// Rough opening in wall-local horizontal/up coordinates, metres.
    pub min: Vec2,
    pub max: Vec2,
    pub columns: u32,
    /// Internal crossbar height relative to this opening (0 means none).
    pub transom: f32,
    pub operable: bool,
}
impl WindowOpening {
    pub fn area(&self) -> f32 {
        (self.max - self.min).element_product()
    }
    pub fn full_height(&self, height: f32) -> bool {
        self.min.y <= 0.16 && height - self.max.y <= 0.18
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shade {
    None,
    Venetian,
    Roller,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Facade {
    pub side: FacadeSide,
    pub openings: Vec<WindowOpening>,
    pub frame: Surface,
    pub frame_width: f32,
    pub frame_depth: f32,
    pub recess: f32,
    pub sill_projection: f32,
    pub shade: Shade,
    pub shade_coverage: f32,
    pub shade_tilt: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExteriorProgram {
    pub exposure_probability: f32,
    pub facades: Vec<Facade>,
}
impl ExteriorProgram {
    pub fn sample(seed: u64, size: Vec3, column_width: f32) -> Self {
        let mut rng = stream(seed, 3090);
        let exposure_probability = rng.random_range(0.35..0.85);
        let mut sides: Vec<_> = FacadeSide::ALL
            .into_iter()
            .filter(|_| rng.random_bool(exposure_probability))
            .collect();
        if sides.is_empty() {
            sides.push(FacadeSide::ALL[rng.random_range(0..3)]);
        }
        let facades = sides
            .into_iter()
            .map(|side| {
                let mut r = stream(seed, 3091 + side as u64);
                let span = side.span(size);
                let edge = column_width + r.random_range(0.04..0.24);
                let usable = span - 2. * edge;
                let count = (usable / r.random_range(1.1..5.8)).round().clamp(1., 8.) as usize;
                let gap = (usable / count as f32 * r.random_range(0.035..0.38)).clamp(0.08, 0.9);
                let weights: Vec<f32> = (0..count).map(|_| r.random_range(0.72..1.30)).collect();
                let scale = (usable - gap * (count - 1) as f32) / weights.iter().sum::<f32>();
                // Continuous skew gives both low sills/full-height glazing and high
                // punched/clerestory openings, without choosing one of a few sizes.
                let bottom = 0.018 + size.y * 0.60 * r.random_range(0.0_f32..1.0).powf(3.2);
                let top = (size.y - 0.035 - size.y * 0.22 * r.random_range(0.0_f32..1.0).powf(3.0))
                    .max(bottom + 0.50);
                let mut bands = vec![(bottom, top)];
                if top - bottom > 1.7 && r.random_bool(0.22) {
                    let spandrel = r.random_range(0.12..0.34);
                    let split = (bottom + (top - bottom) * r.random_range(0.54..0.72))
                        .clamp(bottom + 0.45 + spandrel * 0.5, top - 0.45 - spandrel * 0.5);
                    bands = vec![
                        (bottom, split - spandrel * 0.5),
                        (split + spandrel * 0.5, top),
                    ];
                }
                let frame_width = r.random_range(0.028..0.075);
                let mullion_pitch = r.random_range(0.65..2.6);
                let mut openings = Vec::new();
                let mut u = -span * 0.5 + edge;
                for weight in weights {
                    let width = weight * scale;
                    for &(bottom, top) in &bands {
                        openings.push(WindowOpening {
                            min: Vec2::new(u, bottom),
                            max: Vec2::new(u + width, top),
                            columns: (width / mullion_pitch).ceil().clamp(1., 6.) as u32,
                            transom: if top - bottom > 1.25 && r.random_bool(0.52) {
                                r.random_range(0.27..0.78)
                            } else {
                                0.
                            },
                            operable: r.random_bool(0.35),
                        });
                    }
                    u += width + gap;
                }
                Facade {
                    side,
                    openings,
                    frame: [Surface::Metal, Surface::WoodEdge, Surface::Plastic]
                        [r.random_range(0..3)],
                    frame_width,
                    frame_depth: r.random_range(0.065..0.12),
                    recess: r.random_range(0.04..0.12),
                    sill_projection: r.random_range(0.05..0.18),
                    shade: [
                        Shade::None,
                        Shade::None,
                        Shade::None,
                        Shade::Venetian,
                        Shade::Roller,
                    ][r.random_range(0..5)],
                    shade_coverage: r.random_range(0.0..0.95),
                    shade_tilt: r.random_range(-1.0..0.8),
                }
            })
            .collect();
        Self {
            exposure_probability: exposure_probability as f32,
            facades,
        }
    }
    pub fn facade(&self, side: FacadeSide) -> Option<&Facade> {
        self.facades.iter().find(|f| f.side == side)
    }
    pub fn validate(&self, size: Vec3) -> Result<(), String> {
        if self.facades.is_empty()
            || self.facades.len() > 3
            || !(0.0..=1.0).contains(&self.exposure_probability)
        {
            return Err("invalid exterior exposure".into());
        }
        let mut sides = std::collections::BTreeSet::new();
        for f in &self.facades {
            if !sides.insert(f.side)
                || f.openings.is_empty()
                || f.openings.len() > 16
                || !matches!(
                    f.frame,
                    Surface::Metal | Surface::WoodEdge | Surface::Plastic
                )
                || !(0.02..=0.08).contains(&f.frame_width)
                || !(0.04..=0.14).contains(&f.frame_depth)
                || !(0.03..=0.14).contains(&f.recess)
                || !(0.0..=0.20).contains(&f.sill_projection)
                || !(0.0..=1.0).contains(&f.shade_coverage)
                || !(-1.2..=1.2).contains(&f.shade_tilt)
            {
                return Err("invalid exterior frame or duplicate wall".into());
            }
            for (i, o) in f.openings.iter().enumerate() {
                if !o.min.is_finite()
                    || !o.max.is_finite()
                    || (o.max - o.min).min_element() < 0.35
                    || o.min.x < -f.side.span(size) * 0.5 + 0.01
                    || o.max.x > f.side.span(size) * 0.5 - 0.01
                    || o.min.y < 0.01
                    || o.max.y > size.y - 0.01
                    || !(1..=6).contains(&o.columns)
                    || (o.max.x - o.min.x - 2. * f.frame_width) / (o.columns as f32) < 0.10
                    || !(o.transom == 0. || (0.20..=0.85).contains(&o.transom))
                {
                    return Err("invalid exterior opening dimensions".into());
                }
                for other in &f.openings[i + 1..] {
                    if o.min.cmplt(other.max + Vec2::splat(0.005)).all()
                        && o.max.cmpgt(other.min - Vec2::splat(0.005)).all()
                    {
                        return Err("exterior apertures overlap".into());
                    }
                }
            }
        }
        Ok(())
    }
}

/// Attachments require solid backing, not just a clear furniture footprint.
pub fn overlaps_opening(scene: &IndoorManifest, lo: Vec3, hi: Vec3, margin: f32) -> bool {
    if let Some(envelope) = &scene.envelope {
        return envelope.walls.iter().any(|wall| {
            let Some(f) = &wall.facade else {
                return false;
            };
            let tf = envelope.wall_transform(wall.edge);
            let mut min = Vec3::splat(f32::INFINITY);
            let mut max = Vec3::splat(f32::NEG_INFINITY);
            for x in [lo.x, hi.x] {
                for y in [lo.y, hi.y] {
                    for z in [lo.z, hi.z] {
                        let p = tf.rotation.inverse() * (Vec3::new(x, y, z) - tf.translation);
                        min = min.min(p);
                        max = max.max(p);
                    }
                }
            }
            min.z < 0.30
                && max.z > -0.24
                && f.openings.iter().any(|o| {
                    min.x < o.max.x + margin
                        && max.x > o.min.x - margin
                        && min.y < o.max.y + margin
                        && max.y > o.min.y - margin
                })
        });
    }
    let Some(exterior) = &scene.exterior else {
        return false;
    };
    exterior.facades.iter().any(|f| {
        let a = f.side.local(scene.room_size, lo);
        let b = f.side.local(scene.room_size, hi);
        let min = a.min(b);
        let max = a.max(b);
        min.z < 0.30
            && max.z > -0.24
            && f.openings.iter().any(|o| {
                min.x < o.max.x + margin
                    && max.x > o.min.x - margin
                    && min.y < o.max.y + margin
                    && max.y > o.min.y - margin
            })
    })
}
