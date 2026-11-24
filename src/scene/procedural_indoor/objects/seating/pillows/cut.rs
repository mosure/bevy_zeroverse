//! Two cut cloth panels joined at the perimeter. Boxing, welting and flanges
//! have a physical cross-section on the same closed surface as the faces.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;
use std::sync::OnceLock;

const N: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Edge {
    Knife,
    Welt,
    Boxed,
    Flanged,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(super) struct Cut {
    pub edge: Edge,
    pub size: Vec3,
    pub corner_exponent: f32,
    pub border_m: f32,
    pub boxing: f32,
    pub waist: f32,
    pub corner_pinch: f32,
    pub fill_bias: Vec2,
    pub dimple: f32,
    pub grain_radians: f32,
}

impl Cut {
    pub(super) fn sample(size: Vec3, r: &mut impl Rng) -> Self {
        let edge = match r.random_range(0..100) {
            0..=30 => Edge::Knife,
            31..=57 => Edge::Welt,
            58..=85 => Edge::Boxed,
            _ => Edge::Flanged,
        };
        let border_m: f32 = match edge {
            Edge::Knife => 0.,
            Edge::Welt | Edge::Boxed => r.random_range(0.0018..0.0040),
            Edge::Flanged => r.random_range(0.009..0.027),
        };
        Self {
            edge,
            size,
            corner_exponent: r.random_range(3.2..7.4),
            border_m: border_m.min(size.x.min(size.z) * 0.085),
            boxing: r.random_range(0.28..0.68),
            waist: r.random_range(0.015..0.11),
            corner_pinch: r.random_range(0.22..0.70),
            fill_bias: Vec2::new(r.random_range(-0.13..0.13), r.random_range(-0.17..0.17)),
            dimple: if r.random_bool(0.23) {
                r.random_range(0.15..0.37)
            } else {
                0.
            },
            grain_radians: r.random_range(-0.14..0.14)
                + if r.random_bool(0.26) { FRAC_PI_2 } else { 0. },
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Panel {
    Face,
    Edge,
    Hem,
    Reverse,
}

pub(super) struct Form {
    pub points: Vec<Vec3>,
    // The transition below each ring determines which piece of cloth it uses.
    panels: Vec<Panel>,
    cut: Cut,
}

impl Form {
    pub fn new(cut: Cut) -> Self {
        static ANGLES: OnceLock<[Vec2; N]> = OnceLock::new();
        let dirs = ANGLES
            .get_or_init(|| std::array::from_fn(|i| Vec2::from_angle(TAU * i as f32 / N as f32)));
        let half = cut.size * 0.5;
        let outline = dirs.map(|v| {
            let convert = |k: f32| {
                if k.abs() < 1e-6 {
                    0.
                } else {
                    k.signum() * k.abs().powf(2. / cut.corner_exponent)
                }
            };
            let corner = 2. * (v.x * v.y).abs();
            let plan =
                Vec2::new(convert(v.x), convert(v.y)) * (1. - cut.waist * (1. - corner).powi(2));
            (plan, corner)
        });
        let side = cut.size.x.min(cut.size.z);
        let border = (2. * cut.border_m / side).min(0.18);
        let band = cut.boxing;
        let mut sections = Vec::with_capacity(13);
        let mut add = |radius: f32, loft: f32, next: Panel| sections.push((radius, loft, next));
        let shoulder = match cut.edge {
            Edge::Boxed => band + (1. - band) * 0.38,
            Edge::Flanged => 0.38,
            _ => 0.48,
        };
        add(0.34, 0.965, Panel::Face);
        add(0.65, 0.85, Panel::Face);
        add((1. - border) * 0.89, shoulder, Panel::Face);
        match cut.edge {
            Edge::Knife => add(1., 0., Panel::Reverse),
            Edge::Welt => {
                let ridge = (cut.border_m / half.y).min(0.18);
                add(1. - border, ridge, Panel::Edge);
                add(1., 0., Panel::Edge);
                add(1. - border, -ridge, Panel::Reverse);
            }
            Edge::Boxed => {
                add(1. - border, band, Panel::Edge);
                add(1., band * 0.84, Panel::Edge);
                add(1., -band * 0.84, Panel::Edge);
                add(1. - border, -band, Panel::Reverse);
            }
            Edge::Flanged => {
                let lip = (0.0009 / half.y).min(0.05);
                add(1. - border, lip * 2., Panel::Edge);
                add(1., lip, Panel::Hem);
                add(1., -lip, Panel::Edge);
                add(1. - border, -lip * 2., Panel::Reverse);
            }
        }
        add((1. - border) * 0.89, -shoulder, Panel::Reverse);
        add(0.65, -0.85, Panel::Reverse);
        add(0.34, -0.965, Panel::Reverse);

        let point = |r: f32, loft: f32, plan: Vec2, corner: f32| {
            let xz = (plan * r + cut.fill_bias * (1. - r) * (1. - r)) * half.xz();
            let pinch = 1. - cut.corner_pinch * corner.powi(2) * r * r;
            let crown = loft * pinch - cut.dimple * (1. - r * 2.).max(0.).powi(2) * loft.signum();
            // The seam rises gently towards alternating ears on the same cut.
            let bias = 0.10 * r * r * plan.x * plan.y * (1. - loft.abs());
            Vec3::new(xz.x, (crown + bias) * half.y, xz.y)
        };
        let mut points = Vec::with_capacity(sections.len() * N + 2);
        points.push(point(0., 1., Vec2::ZERO, 0.));
        for &(r, h, _) in &sections {
            points.extend(outline.map(|(v, k)| point(r, h, v, k)));
        }
        points.push(point(0., -1., Vec2::ZERO, 0.));
        Self {
            points,
            panels: sections.iter().map(|s| s.2).collect(),
            cut,
        }
    }
    // A ring has one geometric vertex at its seam. The boxing chart splits
    // that vertex only for UVs, keeping shading continuous around the cut.
    pub(super) fn triangles(&self, mut emit: impl FnMut([usize; 3])) {
        let start = |row: usize| 1 + row * N;
        for i in 0..N {
            emit([0, 1 + (i + 1) % N, 1 + i]);
        }
        for row in 0..self.panels.len() - 1 {
            for i in 0..N {
                let j = (i + 1) % N;
                let (a, b) = (start(row), start(row + 1));
                emit([a + i, a + j, b + j]);
                emit([a + i, b + j, b + i]);
            }
        }
        let last = start(self.panels.len() - 1);
        for i in 0..N {
            emit([last + i, last + (i + 1) % N, self.points.len() - 1]);
        }
    }
    pub(super) fn sew(&self, a: &mut Assembly, cover: Stock, tf: Transform) {
        let mut normals = vec![Vec3::ZERO; self.points.len()];
        self.triangles(|t| {
            let [p, q, r] = t.map(|i| self.points[i]);
            let n = (q - p).cross(r - p);
            for i in t {
                normals[i] += n;
            }
        });
        let normals: Vec<_> = normals
            .into_iter()
            .map(|n| tf.rotation * n.normalize())
            .collect();
        let positions: Vec<_> = self
            .points
            .iter()
            .map(|p| tf.transform_point(*p).to_array())
            .collect();
        let grain = Mat2::from_angle(self.cut.grain_radians);
        let top_uv = |i: usize| grain * self.points[i].xz() + self.cut.size.xz() * 0.5;
        let ring = |row: usize| 1 + row * N;
        let mut from = 0;
        while from <= self.panels.len() {
            let material = |span: usize| {
                if span == 0 {
                    Panel::Face
                } else {
                    self.panels[span - 1]
                }
            };
            let panel = material(from);
            let mut to = from;
            while to < self.panels.len() && material(to + 1) == panel {
                to += 1;
            }
            let (key, origin) = match panel {
                Panel::Face => (cover.face, cover.face_origin_m),
                Panel::Edge | Panel::Hem => (cover.edge, cover.edge_origin_m),
                Panel::Reverse => (cover.reverse, cover.reverse_origin_m),
            };
            let g = a.part(key.surface, key.label());
            let vertex = |g: &mut Geometry, i: usize, uv: Vec2| {
                g.positions.push(positions[i]);
                g.normals.push(normals[i].to_array());
                g.uvs.push((uv + origin).to_array());
            };
            let mut prev = 0;
            if from == 0 {
                prev = g.positions.len() as u32;
                vertex(g, 0, top_uv(0));
            }
            let first = from.saturating_sub(1);
            let last = to.min(self.panels.len() - 1);
            let boxed =
                panel == Panel::Hem || panel == Panel::Edge && self.cut.edge != Edge::Flanged;
            let cols = if boxed { N + 1 } else { N };
            let mut arclength = [0.; N + 1];
            if boxed {
                for i in 1..=N {
                    arclength[i] = arclength[i - 1]
                        + self.points[ring(first) + i % N]
                            .distance(self.points[ring(first) + i - 1]);
                }
            }
            for row in first..=last {
                let offset = g.positions.len() as u32;
                for (i, &u) in arclength[..cols].iter().enumerate() {
                    let ix = ring(row) + i % N;
                    let uv = if boxed {
                        Vec2::new(u, self.points[ix].y - self.points[ring(first) + i % N].y)
                    } else {
                        top_uv(ix)
                    };
                    vertex(g, ix, uv);
                }
                for i in 0..N {
                    let j = if boxed { i + 1 } else { (i + 1) % N };
                    let (i, j) = (i as u32, j as u32);
                    if row == 0 && from == 0 {
                        g.indices.extend([prev, offset + j, offset + i]);
                    } else if row > first {
                        g.indices.extend([
                            prev + i,
                            prev + j,
                            offset + j,
                            prev + i,
                            offset + j,
                            offset + i,
                        ]);
                    }
                }
                prev = offset;
            }
            if to == self.panels.len() {
                let end = g.positions.len() as u32;
                vertex(g, self.points.len() - 1, top_uv(self.points.len() - 1));
                for i in 0..N as u32 {
                    g.indices.extend([prev + i, prev + (i + 1) % N as u32, end]);
                }
            }
            from = to + 1;
        }
    }
}
