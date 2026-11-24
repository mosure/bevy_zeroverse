use super::*;

// A Cartesian chart samples the bowl across its full width and adds stations
// at the corner arcs. Both skins and the rim meet at the same vertices in space.
const COLS: usize = 9;
const ROWS: usize = 13;
const OUTLINE: usize = 2 * (COLS + ROWS) - 4;
const ARC: [f32; 5] = [
    0.,
    0.382_683_43,
    std::f32::consts::FRAC_1_SQRT_2,
    0.923_879_5,
    1.,
];

impl FormedSeat {
    pub(super) fn shell(&self, a: &mut Assembly, face: Surface, half: Vec2) {
        let bevel = (self.gauge_m * 0.23).min(0.0032);
        let inset = half - Vec2::splat(bevel);
        let radius = self.corner_radius_m.min(inset.min_element() * 0.68);
        let grid: [Vec2; COLS * ROWS] = std::array::from_fn(|i| {
            let row = i / COLS;
            let u = 2. * (i % COLS) as f32 / (COLS - 1) as f32 - 1.;
            let (half_width, z) = if (4..=8).contains(&row) {
                (inset.x, (row as f32 - 6.) * 0.5 * (inset.y - radius))
            } else {
                let j = row.min(ROWS - 1 - row);
                let side = if row < 4 { -1. } else { 1. };
                (
                    inset.x - radius * (1. - ARC[j]),
                    side * (inset.y - radius * (1. - ARC[4 - j])),
                )
            };
            let v = z / inset.y;
            let narrow = (1. + v * self.width_taper.signum()) * 0.5 * self.width_taper.abs();
            // Bend the cut in physical X. A row-relative U would shear its
            // first corner strip back across the narrower row at the nose.
            let x = u * half_width * (1. - narrow);
            let sweep = self.front_sweep_m.min(inset.min_element() * 0.22);
            Vec2::new(x, z + sweep * (x / inset.x * v.min(0.)).powi(2))
        });
        let (s, c) = self.chart_angle_radians.sin_cos();
        let grain = Vec2::new(s, c);
        let across = Vec2::new(c, -s);
        let cap = a.part(face, "chair");
        cap.positions.reserve(2 * grid.len());
        cap.normals.reserve(2 * grid.len());
        cap.uvs.reserve(2 * grid.len());
        cap.indices.reserve(12 * (ROWS - 1) * (COLS - 1));
        for bottom in [false, true] {
            let start = cap.positions.len() as u32;
            for xz in grid {
                let (p, n) = self.skin(xz, half, bottom);
                cap.positions.push(p.to_array());
                cap.normals.push(n.to_array());
                cap.uvs.push([xz.dot(across), xz.dot(grain)]);
            }
            for row in 0..ROWS - 1 {
                for col in 0..COLS - 1 {
                    let a = start + (row * COLS + col) as u32;
                    let b = a + 1;
                    let c = a + COLS as u32;
                    let d = c + 1;
                    cap.indices.extend(if bottom {
                        [a, d, c, a, b, d]
                    } else {
                        [a, c, d, a, d, b]
                    });
                }
            }
        }
        let outline: [Vec2; OUTLINE] = std::array::from_fn(|i| {
            let idx = match i {
                _ if i < COLS => i,
                _ if i < COLS + ROWS - 1 => (i - COLS + 1) * COLS + COLS - 1,
                _ if i < 2 * COLS + ROWS - 2 => COLS * ROWS - 2 - (i - (COLS + ROWS - 1)),
                _ => (OUTLINE - i) * COLS,
            };
            grid[idx]
        });
        let rim: [RimSection; OUTLINE] = std::array::from_fn(|i| {
            let p = outline[i];
            let before = outline[(i + OUTLINE - 1) % OUTLINE];
            let after = outline[(i + 1) % OUTLINE];
            let along = ((p - before).normalize() + (after - p).normalize()).normalize();
            let outward = Vec2::new(along.y, -along.x);
            let outer = p + outward * bevel;
            let horizontal = Vec3::new(outward.x, 0., outward.y);
            let (top, up) = self.skin(p, half, false);
            let (bottom, down) = self.skin(p, half, true);
            RimSection {
                points: [
                    top,
                    self.skin(outer, half, false).0 - Vec3::Y * bevel,
                    self.skin(outer, half, true).0 + Vec3::Y * bevel,
                    bottom,
                ],
                normals: [up, horizontal, horizontal, down],
            }
        });
        // Adjacent rim sections share a vertex column unless the stock or
        // the direction of its UV chart changes. The wrap seam keeps both UVs.
        let strips: &[(Surface, usize, usize)] = if self.stock == SeatStock::MouldedPolymer {
            &[(face, 0, 3)]
        } else {
            &[(face, 0, 1), (Surface::WoodEdge, 1, 2), (face, 2, 3)]
        };
        for &(surface, first, last) in strips {
            let g = a.part(surface, "chair");
            let mut distance = 0.;
            let mut previous = None;
            for i in 0..OUTLINE {
                let j = (i + 1) % OUTLINE;
                let delta = outline[j] - outline[i];
                let length = delta.length();
                let next = distance + length;
                let long_edge =
                    self.stock == SeatStock::CarvedTimber && delta.dot(grain).abs() > 0.5 * length;
                let mut column = |section: &RimSection, along: f32| {
                    let start = g.positions.len() as u32;
                    for k in first..=last {
                        let p = section.points[k];
                        let uv = if self.stock != SeatStock::CarvedTimber {
                            [p.y - section.points[0].y, along]
                        } else if long_edge {
                            [p.y - SEAT_Y, p.xz().dot(grain)]
                        } else {
                            [p.xz().dot(across), p.y - SEAT_Y]
                        };
                        g.positions.push(p.to_array());
                        g.normals.push(section.normals[k].to_array());
                        g.uvs.push(uv);
                    }
                    start
                };
                let left = match previous {
                    Some((chart, index)) if chart == long_edge => index,
                    _ => column(&rim[i], distance),
                };
                let right = column(&rim[j], next);
                for k in 0..(last - first) as u32 {
                    let (l, r) = (left + k, right + k);
                    g.indices.extend([l, r, r + 1, l, r + 1, l + 1]);
                }
                previous = Some((long_edge, right));
                distance = next;
            }
        }
    }
}

struct RimSection {
    points: [Vec3; 4],
    normals: [Vec3; 4],
}
