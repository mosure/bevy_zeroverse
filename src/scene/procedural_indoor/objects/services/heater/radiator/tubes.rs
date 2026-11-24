use super::*;

/// Horizontal round or flattened water tubes welded into two vertical headers.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct TubeBank {
    pub body: Body,
    pub rows: usize,
    pub pitch_m: f32,
    pub tube_radii_m: Vec2,
    pub header_width_m: f32,
}

impl TubeBank {
    pub(in super::super) fn fit(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"HeatTube"));
        let body = Body::fit(o, &mut r);
        let rows = (body.height() / r.random_range(0.026..0.055))
            .round()
            .clamp(2., 16.) as usize;
        let pitch = body.height() / rows as f32;
        let y = pitch * r.random_range(0.32..0.43);
        let z = if r.random_bool(0.44) {
            y
        } else {
            y * r.random_range(0.41..0.78)
        };
        Self {
            body,
            rows,
            pitch_m: pitch,
            tube_radii_m: Vec2::new(y, z.min(o.size.z * 0.24)),
            header_width_m: r.random_range(0.022..0.040),
        }
    }

    pub(in super::super) fn build(self, a: &mut Assembly, o: &IndoorObject, label: &str) {
        const SIDES: usize = 16;
        let b = self.body;
        let radii = self.tube_radii_m;
        let depth = 2. * radii.y + 0.008;
        let z = o.size.z * 0.5 - 0.005 - 0.5 * depth;
        let end = 0.5 * (b.width_m - self.header_width_m);
        let g = a.part(Surface::Metal, label);
        for sign in [-1., 1.] {
            g.cuboid(
                Vec3::new(self.header_width_m, b.height(), depth),
                0.003,
                Transform::from_xyz(sign * end, (b.bottom_m + b.top_m) * 0.5, z),
            );
        }
        // All tubes of the same stock share one metric section. The ends are
        // welded inside the headers; only the barrel is exposed.
        let mut arc = 0.;
        let mut previous = Vec2::new(radii.x, 0.);
        let section: [_; SIDES + 1] = std::array::from_fn(|i| {
            let dir = Vec2::from_angle(TAU * (i % SIDES) as f32 / SIDES as f32);
            let offset = radii * dir;
            arc += previous.distance(offset);
            previous = offset;
            let normal = (dir / radii).normalize();
            (offset, [0., normal.x, normal.y], arc)
        });
        g.positions.reserve(2 * (SIDES + 1) * self.rows);
        g.normals.reserve(2 * (SIDES + 1) * self.rows);
        g.uvs.reserve(2 * (SIDES + 1) * self.rows);
        g.indices.reserve(6 * SIDES * self.rows);
        for i in 0..self.rows {
            let y = b.bottom_m + self.pitch_m * (i as f32 + 0.5);
            let start = g.positions.len() as u32;
            for (j, &(offset, normal, distance)) in section.iter().enumerate() {
                for x in [-end, end] {
                    g.positions.push([x, y + offset.x, z + offset.y]);
                    g.normals.push(normal);
                    g.uvs.push([x, distance]);
                }
                if j < SIDES {
                    let a = start + 2 * j as u32;
                    g.indices.extend([a, a + 2, a + 1, a + 1, a + 2, a + 3]);
                }
            }
        }
        b.feet(a, o, z, z, b.bottom_m + self.pitch_m * 0.5);
        b.plumbing(a, o, z);
    }
}
