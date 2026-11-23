//! Metric charts for cut timber. The procedural finish runs its fibres along V.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, materials::Surface};
use rand::Rng;
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Copy)]
pub(crate) enum GrainAxis {
    X,
    Y,
    Z,
}

impl GrainAxis {
    fn frame(self) -> [Vec3; 3] {
        match self {
            Self::X => [Vec3::X, Vec3::Y, Vec3::Z],
            Self::Y => [Vec3::Y, Vec3::Z, Vec3::X],
            Self::Z => [Vec3::Z, Vec3::X, Vec3::Y],
        }
    }
}

pub(crate) struct TimberCuts(ChaCha8Rng);

impl TimberCuts {
    pub(crate) fn new(seed: u64, member: u64) -> Self {
        Self(stream(seed, 0x5449_4d42_4552_0000 ^ member))
    }

    pub(crate) fn origin(&mut self) -> Vec2 {
        Vec2::new(self.0.random_range(0.0..4.0), self.0.random_range(0.0..4.0))
    }

    /// Siblings sample different cuts of the same material. Charts are local
    /// to the member, including when the assembled object moves or rotates.
    pub(crate) fn cuboid(
        &mut self,
        g: &mut Geometry,
        surface: Surface,
        size: Vec3,
        bevel: f32,
        tf: Transform,
        grain: GrainAxis,
    ) {
        // Consume one cut per authored member, including painted or metal
        // members. Refinishing an upright must not move the knots on its shelves.
        let origin = self.origin();
        if !surface.is_cut_timber() {
            g.cuboid(size, bevel, tf);
            return;
        }
        let [length, a, b] = grain.frame();
        let (da, db) = (size.dot(a), size.dot(b));
        g.mapped_cuboid(size, bevel, tf, |n, u, v, _, _, dn| {
            // Unroll the four long faces around the section, with one UV seam.
            // End caps have their own two-dimensional chart, including on thin boards.
            let (across, along, offset) = if n.dot(length) != 0. {
                (a * n.dot(length), -b, 0.)
            } else if n == a {
                (b, -length, db * 0.5)
            } else if n == b {
                (-a, -length, db + da * 0.5)
            } else if n == -a {
                (-b, -length, db * 1.5 + da)
            } else {
                (a, -length, db * 2. + da * 1.5)
            };
            let project = |p: Vec3| Vec2::new(across.dot(p), along.dot(p));
            let start = origin + Vec2::X * offset + project(n * dn);
            let (dx, dy) = (project(u), project(v));
            move |x, y| start + dx * x + dy * y
        });
    }
}

impl Geometry {
    /// Flat returns on a moulding use the same running coordinate as its face.
    /// End cuts have a separate planar chart and should call metric_quad.
    pub(crate) fn stock_quad(
        &mut self,
        points: [Vec3; 4],
        outward: Vec3,
        length: Vec3,
        origin: Option<Vec2>,
    ) {
        if let Some(origin) = origin {
            self.planar_quad(points, outward, |p, normal| {
                let along = length.reject_from_normalized(normal).normalize();
                let across = normal.cross(along);
                p.map(|point| (origin + Vec2::new(point.dot(across), -point.dot(along))).to_array())
            });
        } else {
            self.metric_quad(points, outward);
        }
    }
}

#[cfg(test)]
mod tests;
