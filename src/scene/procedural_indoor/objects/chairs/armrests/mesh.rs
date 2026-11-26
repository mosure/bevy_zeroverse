use super::*;

const SIDES: usize = 9;
const ROWS: usize = 9;
type Ring = [Vec3; SIDES];

impl ArmPad {
    /// One closed loft. The shell and wrapping meet along shared edges. The flat underside
    /// is also the physical mounting datum for the arm post, loop or upholstered side.
    pub(in super::super) fn build(self, a: &mut Assembly, p: &ChairProgram, x: f32) {
        let nose = self.nose_length_fraction;
        let heel = self.heel_length_fraction;
        // Approximate quarter-elliptic ends with closer stations near each tip.
        let stations = [
            (0., 0.),
            (0.134 * nose, 0.5),
            (0.5 * nose, 0.866),
            (nose, 1.),
            (0.5, 1.),
            (1. - heel, 1.),
            (1. - 0.5 * heel, 0.866),
            (1. - 0.134 * heel, 0.5),
            (1., 0.),
        ];
        let bottom = self.underside(p);
        // A flared paddle still fits the width reserved beside the seat.
        let widest = 1. - self.front_taper.min(self.rear_taper).min(0.);
        let rings: [Ring; ROWS] = stations.map(|(t, round)| {
            let along: f32 = 1. - 2. * t;
            let sweep = self.sweep_m * along;
            let taper = if along > 0. {
                self.front_taper
            } else {
                self.rear_taper
            };
            let half_width = (0.5 * p.arm_pad_width_m * (1. - taper * along.abs()) / widest
                - sweep.abs())
                * (self.tip_width_fraction + (1. - self.tip_width_fraction) * round);
            let gauge = self.gauge_m * (0.72 + 0.28 * round);
            let shoulder = 1. - self.crown_fraction;
            [
                (-0.76, 0.),
                (0.76, 0.),
                (0.94, 0.16),
                (1., shoulder * 0.72),
                (0.79, shoulder),
                (0., 1.),
                (-0.79, shoulder),
                (-1., shoulder * 0.72),
                (-0.94, 0.16),
            ]
            .map(|(u, v)| {
                Vec3::new(
                    x + x.signum() * sweep + u * half_width,
                    bottom + v * gauge,
                    (t - 0.5) * p.arm_pad_length + self.offset_z_m,
                )
            })
        });

        // Cross-ring arc length unwraps the sides and the contact face in metres.
        // The raw chair X offsets the grain cut on the two arms.
        let lengths = rings.map(|ring| {
            let mut arc = [0.; SIDES + 1];
            for j in 0..SIDES {
                arc[j + 1] = arc[j] + ring[j].distance(ring[(j + 1) % SIDES]);
            }
            arc
        });
        let normals: [Ring; ROWS] = std::array::from_fn(|i| {
            let ring = &rings[i];
            std::array::from_fn(|j| {
                let incoming = (ring[j] - ring[(j + SIDES - 1) % SIDES]).normalize();
                let outgoing = (ring[(j + 1) % SIDES] - ring[j]).normalize();
                let along = rings[(i + 1).min(ROWS - 1)][j] - rings[i.saturating_sub(1)][j];
                (incoming + outgoing).cross(along).normalize()
            })
        });

        for j in 0..SIDES {
            let lower = j <= 1 || j == SIDES - 1;
            let g = a.part(
                if lower {
                    self.lower_surface()
                } else {
                    self.face
                },
                "chair",
            );
            let start = g.positions.len() as u32;
            for i in 0..ROWS {
                for k in j..=j + 1 {
                    let v = rings[i][k % SIDES];
                    g.positions.push(v.to_array());
                    g.normals.push(
                        if j == 0 {
                            Vec3::NEG_Y
                        } else {
                            normals[i][k % SIDES]
                        }
                        .to_array(),
                    );
                    g.uvs.push([v.z, x + lengths[i][k]]);
                }
                if i + 1 < ROWS {
                    let n = start + 2 * i as u32;
                    g.indices.extend([n, n + 1, n + 3, n, n + 3, n + 2]);
                }
            }
        }
        for (i, normal) in [(0, Vec3::NEG_Z), (ROWS - 1, Vec3::Z)] {
            for (lower, triangles) in [
                (true, &[[0, 1, 2], [0, 2, 8]][..]),
                (
                    false,
                    &[[8, 2, 3], [8, 3, 7], [7, 3, 4], [7, 4, 6], [6, 4, 5]][..],
                ),
            ] {
                let surface = if lower || self.stock == ArmPadStock::Timber {
                    self.lower_surface()
                } else {
                    self.face
                };
                let g = a.part(surface, "chair");
                let start = g.positions.len() as u32;
                // Keep the timber end grain and the upholstery return on a metric end chart.
                let (first, count) = if lower { (8, 4) } else { (2, 7) };
                for k in first..first + count {
                    let v = rings[i][k % SIDES];
                    g.positions.push(v.to_array());
                    g.normals.push(normal.to_array());
                    g.uvs.push([v.x, v.y - bottom]);
                }
                for &tri in triangles {
                    let [a, b, c] = tri.map(|k| start + ((k + SIDES - first) % SIDES) as u32);
                    g.indices.extend(if i == 0 { [a, c, b] } else { [a, b, c] });
                }
            }
        }
    }
}
