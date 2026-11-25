//! Metric texture charts for fluted and polygonal stock. An outline can also
//! cut the contents to the exact cross section of a hollow vessel.
use super::*;

#[derive(Clone, Copy)]
struct Meridian {
    point: Vec2,
    outward: Vec2,
    apothem: f32,
    along: f32,
}

pub(crate) struct RevolvedOutline {
    columns: Vec<Meridian>,
    sides: usize,
    stride: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProfileJoints {
    Creased,
    Smooth,
}

impl RevolvedOutline {
    /// A round-to-soft-square cross section with the supplied axial dimensions.
    /// The gradient of its implicit curve gives the transverse shading normal.
    pub(crate) fn superellipse(sides: u32, exponent: f32, aspect: Vec2) -> Self {
        let mut outline = Self::round(sides, 0., 0);
        if exponent == 2. && aspect == Vec2::ONE {
            return outline;
        }
        let mut distance = 0.;
        let mut previous = None;
        for m in &mut outline.columns {
            let unit = m.point;
            if exponent != 2. {
                let abs = unit.abs();
                let gradient = Vec2::new(abs.x.powf(exponent - 1.), abs.y.powf(exponent - 1.));
                m.point = unit * abs.dot(gradient).powf(-1. / exponent);
                m.outward = unit.signum() * gradient;
            }
            // The outline is physical before charting it. Glaze texels keep
            // their metric footprint across an oval face and around its rim.
            m.point *= aspect;
            m.outward /= aspect;
            m.apothem = m.outward.dot(m.point);
            if let Some(p) = previous {
                distance += m.point.distance(p);
            }
            m.along = distance;
            previous = Some(m.point);
        }
        outline
    }

    /// Support along an arbitrary axis, including asymmetric polygon tessellations.
    pub(crate) fn projected_range(&self, direction: Vec2) -> (f32, f32) {
        self.columns
            .iter()
            .map(|m| direction.dot(m.point))
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), d| {
                (lo.min(d), hi.max(d))
            })
    }

    /// A point on the actual mesh edge, including between the corners of flat stock.
    pub(crate) fn perimeter_point(&self, turn: f32) -> Vec2 {
        let t = turn.rem_euclid(1.) * self.sides as f32;
        let i = (t as usize % self.sides) * self.stride;
        self.columns[i]
            .point
            .lerp(self.columns[i + 1].point, t.fract())
    }

    pub(crate) fn round(sides: u32, depth: f32, lobes: u32) -> Self {
        let mut columns = Vec::with_capacity(sides as usize + 1);
        let mut previous = Vec2::ZERO;
        let mut along = 0.;
        for i in 0..=sides {
            let angle = TAU * (i % sides) as f32 / sides as f32;
            let (sin, cos) = angle.sin_cos();
            let unit = Vec2::new(cos, sin);
            let (wave, slope) = if depth > 0. {
                let (s, c) = (lobes as f32 * angle).sin_cos();
                (1. - depth * (0.5 + 0.5 * c), depth * 0.5 * lobes as f32 * s)
            } else {
                (1., 0.)
            };
            let point = wave * unit;
            let outward = wave * unit + slope * Vec2::new(sin, -cos);
            if i > 0 {
                along += previous.distance(point);
            }
            columns.push(Meridian {
                point,
                outward,
                apothem: outward.dot(point),
                along,
            });
            previous = point;
        }
        Self {
            columns,
            sides: sides as usize,
            stride: 1,
        }
    }

    pub(crate) fn polygon(sides: u32) -> Self {
        Self::round(sides, 0., 0).faceted()
    }

    pub(crate) fn faceted(self) -> Self {
        let ring = self;
        let mut columns = Vec::with_capacity(2 * ring.sides);
        for edge in ring.columns.windows(2) {
            let chord = edge[1].point - edge[0].point;
            let outward = Vec2::new(chord.y, -chord.x).normalize();
            let apothem = outward.dot(edge[0].point);
            for &p in edge {
                columns.push(Meridian {
                    outward,
                    apothem,
                    ..p
                });
            }
        }
        Self {
            columns,
            sides: ring.sides,
            stride: 2,
        }
    }

    /// Unit outlines and their gradients in metric section coordinates, with
    /// the final seam repeated for the cylindrical edge chart.
    pub(crate) fn meridians(&self) -> impl ExactSizeIterator<Item = (Vec2, Vec2, f32)> + '_ {
        self.columns
            .iter()
            .map(|m| (m.point, m.outward / m.apothem, m.along))
    }

    pub(crate) fn sweep(
        &self,
        g: &mut Geometry,
        profile: &[(f32, f32)],
        joints: ProfileJoints,
        tf: Transform,
    ) {
        let radius = profile.iter().map(|p| p.0).fold(0., f32::max);
        let edge = |i: usize| {
            let (r, y) = profile[i];
            let (next_r, next_y) = profile[i + 1];
            Vec2::new(next_r - r, next_y - y)
        };
        let join = |a: Vec2, b: Vec2| {
            // A horizontal lip, foot or floor owns a separate planar chart.
            (joints == ProfileJoints::Smooth && a.y != 0. && b.y != 0. && a.dot(b) > -0.8)
                .then(|| (a + b).normalize())
        };
        let ring = |g: &mut Geometry, (r, y), tangent: Vec2, v, planar| {
            let first = g.positions.len() as u32;
            for m in &self.columns {
                let xz = m.point * r;
                let n = m.outward * tangent.y;
                let uv = if planar {
                    xz
                } else {
                    Vec2::new(m.along * radius, v)
                };
                g.vertex(
                    Vec3::new(xz.x, y, xz.y),
                    Vec3::new(n.x, -m.apothem * tangent.x, n.y),
                    uv,
                    &tf,
                );
            }
            first
        };

        let mut v = 0.;
        let mut last_upper = None;
        for (i, pair) in profile.windows(2).enumerate() {
            let step = edge(i);
            let length = step.length();
            if length == 0. || pair[0].0.max(pair[1].0) <= 0. {
                last_upper = None;
                continue;
            }
            let tangent = step / length;
            let planar = step.y == 0.;
            let lower = last_upper
                .take()
                .unwrap_or_else(|| ring(g, pair[0], tangent, v, planar));
            let next = (i + 2 < profile.len())
                .then(|| edge(i + 1).normalize_or_zero())
                .and_then(|next| join(tangent, next));
            v += length;
            let upper = ring(g, pair[1], next.unwrap_or(tangent), v, planar);
            if next.is_some() {
                last_upper = Some(upper);
            }
            for side in 0..self.sides {
                let k = (side * self.stride) as u32;
                let (a, b) = (lower + k, upper + k);
                if pair[0].0 > 0. {
                    g.indices.extend([a, b, a + 1]);
                }
                if pair[1].0 > 0. {
                    g.indices.extend([a + 1, b, b + 1]);
                }
            }
        }
    }
}
