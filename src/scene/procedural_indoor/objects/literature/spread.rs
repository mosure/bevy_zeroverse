//! Facing leaves of an open notebook. The gutter is a shared edge of the paper
//! surface. Ink and ruling use the same chart, position and depth as the leaves.
use super::*;
use crate::scene::procedural_indoor::materials::paper;

pub(crate) struct Spread {
    /// Full cover dimensions in the reader's frame, including its overhang.
    pub size: Vec3,
    rotation: Quat,
    cover: f32,
    overhang: f32,
    seam: f32,
    curl: [f32; 2],
}

impl Spread {
    pub fn new(o: &IndoorObject, p: &BindingProgram) -> Self {
        let available = o.size * p.envelope_fraction;
        let mut r = stream(o.seed, 0x464143494e475047);
        // Fit the pair into the already-cleared footprint. A sideways pocket
        // notebook still has two ordinary leaves when the reservation is long in Z.
        let sideways = available.z > available.x;
        let long = available.x.max(available.z);
        let short = available.x.min(available.z);
        let leaf_aspect = r.random_range(0.61..0.91);
        let depth = short.min(long / (2. * leaf_aspect));
        let width = depth * 2. * leaf_aspect;
        let heading = if sideways {
            if r.random_bool(0.5) {
                FRAC_PI_2
            } else {
                -FRAC_PI_2
            }
        } else {
            0.
        };
        Self {
            size: Vec3::new(width, available.y, depth),
            rotation: Quat::from_rotation_y(heading),
            cover: (available.y * p.cover_fraction).min(0.002),
            overhang: depth * r.random_range(0.006..0.018),
            seam: width * r.random_range(-0.024..0.024),
            curl: [r.random_range(-0.07..0.07), r.random_range(-0.07..0.07)],
        }
    }

    /// t runs from the gutter to the fore-edge; v runs from page head to foot.
    fn leaf(&self, p: &BindingProgram, right: bool, t: f32, v: f32) -> (Vec3, Vec3) {
        let sign = if right { 1. } else { -1. };
        let balance = if right {
            1. - p.page_balance
        } else {
            p.page_balance
        };
        let run = sign * (self.size.x * 0.5 - self.overhang) - self.seam;
        let depth = self.size.z - 2. * self.overhang;
        let gutter = 0.08 + 0.20 * p.page_balance.min(1. - p.page_balance);
        let rise = 0.10 + 0.64 * balance - gutter;
        let a = 1. - t;
        // A quartic shoulder rises near the binding and relaxes toward the
        // fore-edge. A small signed twist lets one corner lift more than the other.
        let arch = p.page_crown * (256. / 27.);
        let twist = self.curl[usize::from(right)];
        let bend =
            gutter + rise * (1. - a.powi(4)) + arch * t * a.powi(3) + twist * t * t * (v - 0.5);
        let dt =
            4. * rise * a.powi(3) + arch * a.powi(2) * (1. - 4. * t) + 2. * twist * t * (v - 0.5);
        let point = Vec3::new(
            self.seam + run * t,
            self.cover + self.size.y * bend,
            depth * (v - 0.5),
        );
        let normal = Vec3::new(
            -self.size.y * dt / run,
            1.,
            -self.size.y * twist * t * t / depth,
        );
        (self.rotation * point, self.rotation * normal.normalize())
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &BindingProgram) {
    let s = Spread::new(o, p);
    let label = o.kind.class_name();
    let corner = (s.size.z * p.corner_fraction).min(1.65 * s.overhang);
    a.part(p.cover, label).profile_slab(
        &cover_outline(s.size.x, s.size.z, corner),
        s.cover,
        s.cover * 0.20,
        Transform::from_translation(Vec3::Y * (s.cover * 0.5)).with_rotation(s.rotation),
    );
    let chart = format!("{label}#finish{}#uv_local", paper::spread_slot(o.seed));
    // Extra sections at the crease resolve its curvature. Across page depth the
    // height is linear; both ends of each section suffice.
    const SECTIONS: [f32; 10] = [0., 0.035, 0.08, 0.15, 0.24, 0.37, 0.53, 0.70, 0.86, 1.];
    for right in [false, true] {
        let mut block = Geometry::default();
        let sheet = a.part(Surface::PrintedPaper, &chart);
        let first = sheet.positions.len() as u32;
        let mut previous = [Vec3::ZERO; 2];
        for (i, t) in SECTIONS.into_iter().enumerate() {
            let u = 0.5 + if right { t * 0.5 } else { -t * 0.5 };
            let mut current = [Vec3::ZERO; 2];
            for (end, vertex) in current.iter_mut().enumerate() {
                let (pos, normal) = s.leaf(p, right, t, end as f32);
                *vertex = pos;
                sheet.positions.push(pos.to_array());
                sheet.normals.push(normal.to_array());
                sheet.uvs.push([u, end as f32]);
            }
            if i > 0 {
                let k = first + 2 * (i as u32 - 1);
                let corners = if right {
                    [k, k + 1, k + 3, k + 2]
                } else {
                    [k + 2, k + 3, k + 1, k]
                };
                let [a, b, c, d] = corners;
                sheet.indices.extend([a, b, c, a, c, d]);
                for end in 0..2 {
                    let p0 = previous[end];
                    let p1 = current[end];
                    block.metric_quad(
                        [p0, p1, p1.with_y(s.cover), p0.with_y(s.cover)],
                        s.rotation * if end == 0 { Vec3::NEG_Z } else { Vec3::Z },
                    );
                }
            }
            previous = current;
        }
        let [head, foot] = previous;
        block.metric_quad(
            [head, foot, foot.with_y(s.cover), head.with_y(s.cover)],
            s.rotation * if right { Vec3::X } else { Vec3::NEG_X },
        );
        a.part(Surface::Paper, label).append(block);
    }
}

#[cfg(test)]
mod tests;
