//! A formed substrate and its occupant-facing surface share a closed perimeter.
//! All dimensions are in metres; a fully folded seat clears the front skin.
use super::*;

#[derive(Debug, serde::Serialize)]
pub(super) struct Backrest {
    pub(super) z: f32,
    bottom: f32,
    height: f32,
    width: f32,
    corner: f32,
    waist: f32,
    taper: f32,
    cup: f32,
    rake: f32,
    pad: f32,
    gauge: f32,
}

impl Backrest {
    pub(super) fn sample(o: &IndoorObject, w: f32, z: f32, fixed: bool, r: &mut impl Rng) -> Self {
        let bottom = r.random_range(0.545..0.615);
        let available = o.size.z * 0.5 - z - 0.006;
        let cup = available * r.random_range(0.26..0.43);
        Self {
            z,
            bottom,
            height: o.size.y - 0.010 - bottom,
            width: w * r.random_range(0.89..1.05),
            corner: r.random_range(0.075..0.195),
            waist: r.random_range(0.02..0.15),
            taper: r.random_range(0.015..0.14),
            cup,
            rake: available * r.random_range(0.23..0.38),
            pad: if fixed {
                cup * r.random_range(0.40..0.88)
            } else {
                0.
            },
            gauge: available * r.random_range(0.09..0.15),
        }
    }

    /// Normal and position for a continuous local shell. Front bulging changes
    /// its thickness inside the sewn edge without pushing into the folding arc.
    fn sample_skin(&self, u: f32, v: f32, rear: bool) -> (Vec3, Vec3) {
        let wave = 4. * v * (1. - v);
        let slope = 4. - 8. * v;
        let hw = self.width * 0.5;
        let span = 1. - self.waist * wave - self.taper * v;
        let corner = self.corner * u.powi(4);
        let bow = self.cup - if rear { 0. } else { self.pad * wave };
        let point = Vec3::new(
            hw * u * span,
            self.bottom + self.height * (v + corner * (1. - 2. * v)),
            self.z + self.rake * v + (1. - u * u) * bow + if rear { self.gauge } else { 0. },
        );
        let du = Vec3::new(
            hw * span,
            self.height * 4. * self.corner * u.powi(3) * (1. - 2. * v),
            -2. * u * bow,
        );
        let dv = Vec3::new(
            hw * u * (-self.waist * slope - self.taper),
            self.height * (1. - 2. * corner),
            self.rake
                - if rear {
                    0.
                } else {
                    (1. - u * u) * self.pad * slope
                },
        );
        let n = du.cross(dv).normalize();
        (point, if rear { n } else { -n })
    }

    pub(super) fn mount(&self, side: f32) -> Vec3 {
        self.sample_skin(side * 0.81, 0.43, true).0
    }

    pub(super) fn build(&self, a: &mut Assembly, face: Surface, shell: Surface) {
        const COLS: u32 = 10;
        const ROWS: u32 = 6;
        let offset_at =
            |col: u32, row: u32| (2. * col as f32 / COLS as f32 - 1., row as f32 / ROWS as f32);
        for rear in [false, true] {
            let g = a.part(if rear { shell } else { face }, "chair");
            let start = g.positions.len() as u32;
            for row in 0..=ROWS {
                for col in 0..=COLS {
                    let (u, v) = offset_at(col, row);
                    let (p, n) = self.sample_skin(u, v, rear);
                    g.positions.push(p.to_array());
                    g.normals.push(n.to_array());
                    g.uvs
                        .push([if rear { p.x } else { -p.x }, p.y - self.bottom]);
                }
            }
            for row in 0..ROWS {
                for col in 0..COLS {
                    let i = start + row * (COLS + 1) + col;
                    let j = i + COLS + 1;
                    g.indices.extend(if rear {
                        [i, i + 1, j + 1, i, j + 1, j]
                    } else {
                        [i, j + 1, i + 1, i, j, j + 1]
                    });
                }
            }
        }
        // Metric wrap for the visible shell edge. Derive its winding from the
        // ordered outline even when a corner turns sharply on a compact seat.
        let ring = (0..COLS)
            .map(|i| (i, 0))
            .chain((0..ROWS).map(|j| (COLS, j)))
            .chain((1..=COLS).rev().map(|i| (i, ROWS)))
            .chain((1..=ROWS).rev().map(|j| (0, j)));
        let g = a.part(shell, "chair");
        let mut along = 0.;
        let pair = |(i, j)| {
            let (u, v) = offset_at(i, j);
            [
                self.sample_skin(u, v, false).0,
                self.sample_skin(u, v, true).0,
            ]
        };
        let mut prev = pair((0, 0));
        for ij in ring.skip(1).chain(std::iter::once((0, 0))) {
            let next = pair(ij);
            let distance = along + prev[0].distance(next[0]);
            let n = (next[0] - prev[0]).cross(prev[1] - prev[0]).normalize();
            let start = g.positions.len() as u32;
            for (p, uv) in [
                (prev[0], [along, 0.]),
                (next[0], [distance, 0.]),
                (next[1], [distance, next[1].z - next[0].z]),
                (prev[1], [along, prev[1].z - prev[0].z]),
            ] {
                g.positions.push(p.to_array());
                g.normals.push(n.to_array());
                g.uvs.push(uv);
            }
            g.indices
                .extend([start, start + 1, start + 2, start, start + 2, start + 3]);
            prev = next;
            along = distance;
        }
    }
}
