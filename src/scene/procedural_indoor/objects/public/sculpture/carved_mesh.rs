//! Two continuous faces share the same section lines. A bore splits a section
//! into two flanks. Its rounded wall connects the faces around the whole hole.
use super::{carving::Carving, *};

#[derive(Clone, Copy)]
struct SectionPoint {
    mid: Vec3,
    depth: f32,
}

#[derive(Default)]
struct Pattern {
    points: Vec<SectionPoint>,
    rows: Vec<[usize; 4]>,
    holes: Vec<std::ops::RangeInclusive<usize>>,
}

impl Pattern {
    fn row(&mut self, c: &Carving, v: f32, center: f32, radius: f32) {
        let (left, right) = (center - radius, center + radius);
        let us = [-1., left, right, 1.];
        let mut row = [0; 4];
        for (i, u) in us.into_iter().enumerate() {
            if i == 2 && radius == 0. {
                row[i] = row[1];
            } else {
                row[i] = self.points.len();
                let (mid, depth) = c.section(u, v);
                self.points.push(SectionPoint { mid, depth });
            }
        }
        self.rows.push(row);
    }

    fn sample(c: &Carving) -> Self {
        let mut p = Self::default();
        let mut last_y = 0.;
        let mut last_center = 0.;
        p.row(c, 0., 0., 0.);
        for eye in &c.apertures {
            let bottom_center = eye.center - eye.drift;
            for i in 1..=3 {
                let t = i as f32 / 3.;
                p.row(
                    c,
                    f32::lerp(last_y, eye.bottom, t),
                    f32::lerp(last_center, bottom_center, t),
                    0.,
                );
            }
            let start = p.rows.len() - 1;
            // Cosine spacing resolves the rounded tips without zero-area faces.
            const SEGMENTS: usize = 12;
            for i in 1..=SEGMENTS {
                let (s, z) = (PI * i as f32 / SEGMENTS as f32).sin_cos();
                let v = eye.bottom.lerp(eye.top, 0.5 - 0.5 * z);
                let radius = if i == SEGMENTS { 0. } else { s * eye.radius };
                p.row(c, v, eye.center - z * eye.drift, radius);
            }
            p.holes.push(start..=p.rows.len() - 1);
            last_y = eye.top;
            last_center = eye.center + eye.drift;
        }
        for i in 1..=4 {
            let t = i as f32 * 0.25;
            p.row(
                c,
                f32::lerp(last_y, 1., t),
                f32::lerp(last_center, 0., t),
                0.,
            );
        }
        p
    }

    fn outer(&self) -> Vec<usize> {
        let bottom = self.rows[0];
        let top = self.rows.last().unwrap();
        let mut loop_ids = vec![bottom[0], bottom[1], bottom[3]];
        loop_ids.extend(self.rows.iter().skip(1).map(|r| r[3]));
        loop_ids.extend([top[1], top[0]]);
        loop_ids.extend(self.rows[1..self.rows.len() - 1].iter().rev().map(|r| r[0]));
        loop_ids
    }

    fn bore(&self, range: &std::ops::RangeInclusive<usize>) -> Vec<usize> {
        let mut ids: Vec<_> = self.rows[range.clone()].iter().map(|r| r[1]).collect();
        ids.extend(
            self.rows[*range.start() + 1..*range.end()]
                .iter()
                .rev()
                .map(|r| r[2]),
        );
        ids
    }
}

struct Stock<'a> {
    carving: &'a Carving,
    rotation: Quat,
    center: Vec2,
}

impl Stock<'_> {
    fn face_xy(&self, mid: Vec3) -> Vec2 {
        self.center.lerp(mid.xy(), 1. - self.carving.face_inset)
    }

    fn point(&self, sample: &SectionPoint, xy: Vec2, across: f32) -> Vec3 {
        let c = self.carving;
        let z = across * sample.depth;
        self.rotation
            * Vec3::new(
                xy.x + z * (c.half_width_m / c.half_depth_m) * c.bore_shear,
                xy.y,
                sample.mid.z + z,
            )
    }

    fn face(&self, p: &Pattern, front: bool) -> Geometry {
        let mut g = Geometry::default();
        let sign = if front { 1. } else { -1. };
        for sample in &p.points {
            let xy = self.face_xy(sample.mid);
            g.positions.push(self.point(sample, xy, sign).to_array());
            g.uvs.push([xy.x * sign, xy.y]);
        }
        // Increasing X and then Y is outward on the front. Adjacent flanks
        // share their seam vertices wherever the section has no opening.
        for pair in p.rows.windows(2) {
            for i in [0, 2] {
                let [a, b, c, d] =
                    [pair[0][i], pair[0][i + 1], pair[1][i + 1], pair[1][i]].map(|j| j as u32);
                g.indices.extend(if front {
                    [a, b, c, a, c, d]
                } else {
                    [a, c, b, a, d, c]
                });
            }
        }
        g.normals.resize(p.points.len(), [0.; 3]);
        for tri in g.indices.as_chunks::<3>().0 {
            let pos = |i| Vec3::from_array(g.positions[i as usize]);
            let n = (pos(tri[1]) - pos(tri[0])).cross(pos(tri[2]) - pos(tri[0]));
            for &j in tri {
                let dst = &mut g.normals[j as usize];
                *dst = (Vec3::from_array(*dst) + n).to_array();
            }
        }
        for n in &mut g.normals {
            *n = Vec3::from_array(*n).normalize().to_array();
        }
        g
    }

    /// At the outside the roll grows to the full envelope. At a bore the wall
    /// rolls inward towards the throat. Its UVs follow perimeter and depth.
    fn rim(
        &self,
        p: &Pattern,
        path: &[usize],
        bore: Option<(Vec2, f32)>,
        faces: &[Geometry; 2],
    ) -> Geometry {
        let mut g = Geometry::default();
        let m = path.len();
        let stride = m + 1;
        let layers = [
            (-1., 0.),
            (-0.90, 0.72),
            (-0.65, 1.),
            (0.65, 1.),
            (0.90, 0.72),
            (1., 0.),
        ];
        let mut distance = 0.;
        let mut lengths = Vec::with_capacity(stride);
        g.positions.reserve(stride * layers.len());
        g.normals.reserve(stride * layers.len());
        g.uvs.reserve(stride * layers.len());
        g.indices.reserve(m * (layers.len() - 1) * 6);
        for (i, &id) in path.iter().cycle().take(stride).enumerate() {
            if i > 0 {
                distance += p.points[id].mid.distance(p.points[path[i - 1]].mid);
            }
            lengths.push(distance);
        }
        for (layer, (across, roll)) in layers.into_iter().enumerate() {
            for (i, &id) in path.iter().cycle().take(stride).enumerate() {
                let point = &p.points[id];
                let flat = self.face_xy(point.mid);
                let xy = match bore {
                    Some((center, throat)) => flat.lerp(center, roll * throat),
                    None => flat.lerp(point.mid.xy(), roll),
                };
                let pos = self.point(point, xy, across);
                g.positions.push(pos.to_array());
                let v = if layer == 0 {
                    0.
                } else {
                    let prev = (layer - 1) * stride + i;
                    g.uvs[prev][1] + pos.distance(Vec3::from_array(g.positions[prev]))
                };
                g.uvs.push([lengths[i], v]);
                if i < m && layer + 1 < layers.len() {
                    let a = (layer * stride + i) as u32;
                    let b = a + stride as u32;
                    g.indices.extend([a, a + 1, b + 1, a, b + 1, b]);
                }
            }
        }
        for layer in 0..layers.len() {
            for (i, &id) in path.iter().cycle().take(stride).enumerate() {
                let at = |l: usize, k: usize| Vec3::from_array(g.positions[l * stride + k]);
                let tangent = at(layer, (i + 1) % m) - at(layer, (i + m - 1) % m);
                let next = (layer + 1).min(layers.len() - 1);
                let through = at(next, i) - at(layer.saturating_sub(1), i);
                let rim = tangent.cross(through).normalize();
                let normal = if layer == 0 || layer + 1 == layers.len() {
                    // A deeply sloped face can meet the lip at a sharp angle.
                    // Blend only the part of its normal facing the same way.
                    let face = Vec3::from_array(faces[usize::from(layer != 0)].normals[id]);
                    (rim + face * rim.dot(face).max(0.)).normalize()
                } else {
                    rim
                };
                g.normals.push(normal.to_array());
            }
        }
        g
    }
}

pub(super) fn build(a: &mut Assembly, carving: &Carving, finish: Surface) {
    let stock = Stock {
        carving,
        rotation: Quat::from_rotation_y(carving.mount_yaw_radians),
        center: Vec2::new(0., carving.foot_y_m + carving.height_m * 0.5),
    };
    let p = Pattern::sample(carving);
    let faces = [stock.face(&p, false), stock.face(&p, true)];
    let label = "sculpture";
    a.part(finish, label)
        .append(stock.rim(&p, &p.outer(), None, &faces));
    for (hole, eye) in p.holes.iter().zip(&carving.apertures) {
        let (center, _) = carving.section(eye.center, 0.5 * (eye.bottom + eye.top));
        let bore = Some((stock.face_xy(center), eye.throat));
        a.part(carving.bore_surface, label)
            .append(stock.rim(&p, &p.bore(hole), bore, &faces));
    }
    for face in faces {
        a.part(finish, label).append(face);
    }
}
