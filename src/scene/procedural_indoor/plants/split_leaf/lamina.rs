//! Each lateral vein bounds a strip of lamina. The two sides share a midrib.
//! A fenestrated strip is a ring of triangles, so every render pass sees the hole.
use super::{Growth, Leaf, Opening};
use crate::scene::procedural_indoor::geometry::Geometry;
use bevy::prelude::*;
use std::f32::consts::{FRAC_1_SQRT_2, PI};

struct Blade<'a> {
    base: Vec3,
    delta: Vec3,
    axis: Vec3,
    side: Vec3,
    up: Vec3,
    width: f32,
    growth: &'a Growth,
    leaf: &'a Leaf,
}

struct Section {
    t: f32,
    center: Vec3,
    tangent: Vec3,
    envelope: f32,
    sweep: f32,
    fold: f32,
}

impl Blade<'_> {
    fn section(&self, t: f32) -> Section {
        let (wave, slope) = (PI * t).sin_cos();
        let sag = self.growth.droop;
        Section {
            t,
            center: self.base
                + self.delta * t
                + self.up * self.width * (self.leaf.arch * wave - sag * t * t),
            tangent: self.delta
                + self.up * self.width * (PI * self.leaf.arch * slope - 2. * sag * t),
            // Broad basal shoulders narrow to a pointed drip tip.
            envelope: if t == 1. {
                0.
            } else {
                wave.powf(0.52 + 0.19 * (1. - self.leaf.age)) * (1. - 0.26 * t)
            },
            sweep: -0.30 + (0.40 + self.leaf.rake * 3.) * t,
            fold: self.leaf.twist * (t - 0.25),
        }
    }

    fn vertex(&self, g: &mut Geometry, s: &Section, q: f32) -> u32 {
        let w = self.width * s.envelope * q * (1. + q * self.leaf.asymmetry);
        let sign = if q == 0. { 0. } else { q.signum() };
        let curl = self.growth.leaf_curl;
        let lateral = self.side + self.axis * (sign * s.sweep) + self.up * (s.fold - sign * curl);
        let longitudinal = s.tangent
            + self.axis * (w.abs() * (0.40 + self.leaf.rake * 3.))
            + self.up * (w * self.leaf.twist);
        let p = s.center + lateral * w;
        let i = g.positions.len() as u32;
        g.positions.push(p.to_array());
        g.normals
            .push(lateral.cross(longitudinal).normalize().to_array());
        // Keep the vein atlas centered on the physical midrib as its density and phase vary per leaf.
        g.uvs.push([
            0.5 + 0.5 * s.envelope * q,
            self.leaf.vein_offset + self.leaf.vein_span * s.t,
        ]);
        i
    }

    fn row(&self, g: &mut Geometry, t: f32) -> [u32; 5] {
        let section = self.section(t);
        [-1., -0.5, 0., 0.5, 1.].map(|q| self.vertex(g, &section, q))
    }
}

// Indices around a half strip start at the midrib on its basal edge.
fn strip(
    g: &mut Geometry,
    blade: &Blade,
    ring: [u32; 8],
    range: [f32; 2],
    sign: f32,
    opening: Opening,
) {
    let [r0, h0, m0, mid, m1, h1, r1, rib] = ring;
    let start = g.indices.len();
    match opening {
        Opening::Window { center, radius } => {
            const C: f32 = FRAC_1_SQRT_2;
            const OVAL: [[f32; 2]; 8] = [
                [-C, -C],
                [0., -1.],
                [C, -C],
                [1., 0.],
                [C, C],
                [0., 1.],
                [-C, C],
                [-1., 0.],
            ];
            let hole = OVAL.map(|[x, y]| {
                let v = 0.50 + y * 0.30 + x * 0.12;
                let s = blade.section(range[0] + (range[1] - range[0]) * v);
                blade.vertex(g, &s, sign * (center + x * radius))
            });
            for i in 0..8 {
                let j = (i + 1) % 8;
                g.indices
                    .extend([ring[i], ring[j], hole[i], ring[j], hole[j], hole[i]]);
            }
        }
        Opening::Cleft { .. } => {
            // Mid is the recessed tip of the cleft. Retain rounded shoulders at
            // the outer margin on either side, leaving a clear path to the edge.
            let [lower, upper] = [0.20, 0.75].map(|v| {
                let section = blade.section(range[0] + (range[1] - range[0]) * v);
                blade.vertex(g, &section, sign * 0.96)
            });
            g.indices.extend([
                r0, h0, rib, h0, mid, rib, h0, lower, mid, h0, m0, lower, rib, mid, h1, rib, h1,
                r1, mid, upper, h1, upper, m1, h1,
            ]);
        }
        Opening::Entire => {
            g.indices.extend([
                r0, h0, rib, h0, mid, rib, h0, m0, mid, rib, mid, h1, rib, h1, r1, mid, m1, h1,
            ]);
        }
    }
    if sign < 0. {
        for tri in g.indices[start..].as_chunks_mut::<3>().0 {
            tri.swap(1, 2);
        }
    }
}

pub(super) fn append(
    g: &mut Geometry,
    base: Vec3,
    delta: Vec3,
    width: f32,
    growth: &Growth,
    leaf: &Leaf,
) {
    let axis = delta.normalize();
    let side = axis.cross(Vec3::Y).normalize();
    let blade = Blade {
        base,
        delta,
        axis,
        side,
        up: side.cross(axis),
        width,
        growth,
        leaf,
    };
    let root = blade.vertex(g, &blade.section(0.), 0.);
    let mut last = blade.row(g, leaf.rows[0]);
    for edge in last.windows(2) {
        g.indices.extend([root, edge[1], edge[0]]);
    }
    for k in 0..leaf.count {
        let range = [leaf.rows[k], leaf.rows[k + 1]];
        let next = blade.row(g, range[1]);
        let mid = blade.section((range[0] + range[1]) * 0.5);
        let rib = blade.vertex(g, &mid, 0.);
        for (side, sign) in [-1., 1.].into_iter().enumerate() {
            let opening = leaf.bays[k][side];
            let extent = match opening {
                Opening::Cleft { depth } => depth,
                _ => 1.,
            };
            let margin = blade.vertex(g, &mid, sign * extent);
            let outer = side * 4;
            let half = side * 2 + 1;
            strip(
                g,
                &blade,
                [
                    last[2],
                    last[half],
                    last[outer],
                    margin,
                    next[outer],
                    next[half],
                    next[2],
                    rib,
                ],
                range,
                sign,
                opening,
            );
        }
        last = next;
    }
    let tip = blade.vertex(g, &blade.section(1.), 0.);
    for edge in last.windows(2) {
        g.indices.extend([edge[0], edge[1], tip]);
    }
}
