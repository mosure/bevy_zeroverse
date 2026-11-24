//! A perforated sheet with folded returns. Only the surviving skin is emitted,
//! so both sides and views into the compartment agree on the vent openings.
use super::*;
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct VentStock {
    pub upper: bool,
    pub lower: bool,
    pub hooded: bool,
    pub width_fraction: f32,
    pub slot_height_m: f32,
    pub pitch_m: f32,
    pub projection_m: f32,
    pub gauge_m: f32,
    pub edge_inset_fraction: f32,
}
impl VentStock {
    pub(super) fn sample(r: &mut ChaCha8Rng) -> Self {
        let bands = r.random_range(0..5);
        let height = r.random_range(0.007..0.014);
        Self {
            upper: bands != 0,
            lower: bands != 1,
            hooded: r.random_bool(0.63),
            width_fraction: r.random_range(0.39..0.66),
            slot_height_m: height,
            pitch_m: height + r.random_range(0.006..0.016),
            projection_m: r.random_range(0.0035..0.0075),
            gauge_m: r.random_range(0.0009..0.0016),
            edge_inset_fraction: r.random_range(0.09..0.20),
        }
    }

    pub(super) fn centers(self, height: f32, rows: usize) -> impl Iterator<Item = f32> {
        let inset = (height * self.edge_inset_fraction).clamp(0.025, 0.125);
        let max_count = 1
            + ((height * 0.5 - inset - self.slot_height_m) / self.pitch_m)
                .max(0.)
                .floor() as usize;
        let count = rows.min(max_count);
        let first = -height * 0.5 + inset;
        (0..count)
            .filter(move |_| self.lower)
            .map(move |i| first + i as f32 * self.pitch_m)
            .chain(
                (0..count)
                    .rev()
                    .filter(move |_| self.upper)
                    .map(move |i| -first - i as f32 * self.pitch_m),
            )
    }

    pub(super) fn build(self, g: &mut Geometry, size: Vec3, rows: usize) {
        let w = size.x * 0.5;
        let h = size.y * 0.5;
        let z = size.z * 0.5;
        let gauge = self.gauge_m;
        let half_slot = w * self.width_fraction;
        let half_h = self.slot_height_m * 0.5;
        let bevel = half_h * 0.52;
        // Corresponding vertices at every station keep angled views watertight.
        for (plane, xmax, ymax, n) in [
            (z, w, h, Vec3::Z),
            (z - gauge, w - gauge, h - gauge, Vec3::NEG_Z),
        ] {
            // Four indexed vertices per station serve both neighbouring strips.
            // Each side is planar and uses one continuous metric UV chart.
            let mut last = None;
            let mut station = |y: f32, slot_half: f32, filled: bool| {
                let start = g.positions.len() as u32;
                for x in [-xmax, -slot_half, slot_half, xmax] {
                    g.positions.push([x, y, plane]);
                    g.normals.push(n.to_array());
                    g.uvs.push([x * n.z, y]);
                }
                if let Some(previous) = last {
                    for column in 0..3 {
                        if column == 1 && !filled {
                            continue;
                        }
                        let lower = previous + column;
                        let upper = start + column;
                        if n.z > 0. {
                            g.indices.extend([
                                lower,
                                lower + 1,
                                upper + 1,
                                lower,
                                upper + 1,
                                upper,
                            ]);
                        } else {
                            g.indices.extend([
                                lower,
                                upper + 1,
                                lower + 1,
                                lower,
                                upper,
                                upper + 1,
                            ]);
                        }
                    }
                }
                last = Some(start);
            };
            let small = half_slot - bevel;
            station(-ymax, small, true);
            for y in self.centers(size.y, rows) {
                station(y - half_h, small, true);
                station(y - half_h + bevel, half_slot, false);
                station(y + half_h - bevel, half_slot, false);
                station(y + half_h, small, false);
            }
            station(ymax, small, true);
        }
        let corners = |w: f32, h: f32| {
            [
                Vec2::new(-w, -h),
                Vec2::new(w, -h),
                Vec2::new(w, h),
                Vec2::new(-w, h),
            ]
        };
        let outside = corners(w, h);
        let inside = corners(w - gauge, h - gauge);
        for i in 0..4 {
            let j = (i + 1) % 4;
            let a = outside[i];
            let b = outside[j];
            let c = inside[i];
            let d = inside[j];
            let n = Vec3::new(b.y - a.y, a.x - b.x, 0.);
            g.metric_quad([a.extend(-z), b.extend(-z), b.extend(z), a.extend(z)], n);
            g.metric_quad(
                [
                    c.extend(-z),
                    d.extend(-z),
                    d.extend(z - gauge),
                    c.extend(z - gauge),
                ],
                -n,
            );
            g.metric_quad(
                [a.extend(-z), b.extend(-z), d.extend(-z), c.extend(-z)],
                Vec3::NEG_Z,
            );
        }
        for y in self.centers(size.y, rows) {
            let left = -half_slot;
            let right = half_slot;
            let low = y - half_h;
            let high = y + half_h;
            let outline = [
                Vec2::new(left + bevel, low),
                Vec2::new(right - bevel, low),
                Vec2::new(right, low + bevel),
                Vec2::new(right, high - bevel),
                Vec2::new(right - bevel, high),
                Vec2::new(left + bevel, high),
                Vec2::new(left, high - bevel),
                Vec2::new(left, low + bevel),
            ];
            for i in 0..8 {
                // The turned lip joins the front and rear skins across this edge.
                if self.hooded && i == 4 {
                    continue;
                }
                let a = outline[i];
                let b = outline[(i + 1) % 8];
                g.metric_quad(
                    [
                        a.extend(z - gauge),
                        b.extend(z - gauge),
                        b.extend(z),
                        a.extend(z),
                    ],
                    Vec3::new(a.y - b.y, b.x - a.x, 0.),
                );
            }
            if self.hooded {
                // A folded lip grows out of the upper cut, leaving the lower slot open.
                let half = half_slot - bevel;
                let end = high - self.slot_height_m * 0.55;
                let points = [
                    Vec2::new(high, z),
                    Vec2::new(end, z + self.projection_m),
                    Vec2::new(end - gauge, z + self.projection_m - gauge * 0.4),
                    Vec2::new(high, z - gauge),
                ];
                let v = |x, p: Vec2| Vec3::new(x, p.x, p.y);
                for i in 0..3 {
                    let a = points[i];
                    let b = points[i + 1];
                    let n = Vec3::new(0., b.y - a.y, a.x - b.x);
                    g.metric_quad([v(-half, a), v(half, a), v(half, b), v(-half, b)], n);
                }
                for side in [-1., 1.] {
                    g.metric_quad(points.map(|p| v(side * half, p)), Vec3::X * side);
                }
            }
        }
    }
}
