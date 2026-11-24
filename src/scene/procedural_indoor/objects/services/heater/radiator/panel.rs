use super::*;

/// Stamped water channels merge into the horizontal header at either end.
/// All length fields describe the installed product in metres.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct PressedPanel {
    pub body: Body,
    pub channels: usize,
    pub pitch_m: f32,
    pub relief_m: f32,
    pub shoulder: f32,
    pub margin_m: f32,
    pub header_m: f32,
    pub web_depth_m: f32,
    pub bank_spacing_m: Option<f32>,
}

impl PressedPanel {
    pub(in super::super) fn fit(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"HeatPres"));
        let body = Body::fit(o, &mut r);
        let margin = r.random_range(0.009_f32..0.022).min(body.width_m * 0.06);
        let span = body.width_m - 2. * margin;
        let channels = (span / r.random_range(0.029..0.054)).round().clamp(5., 92.) as usize;
        let pitch = span / channels as f32;
        let relief = r.random_range(0.0034_f32..0.0078).min(pitch * 0.20);
        Self {
            body,
            channels,
            pitch_m: pitch,
            relief_m: relief,
            shoulder: r.random_range(0.19..0.34),
            margin_m: margin,
            header_m: r.random_range(0.014_f32..0.030).min(body.height() * 0.18),
            web_depth_m: r.random_range(0.008..0.016),
            bank_spacing_m: (o.size.z >= 0.099 && r.random_bool(0.55))
                .then(|| o.size.z * r.random_range(0.40..0.62)),
        }
    }

    pub(in super::super) fn build(self, a: &mut Assembly, o: &IndoorObject, label: &str) {
        let front = o.size.z * 0.5 - 0.004 - self.relief_m;
        let core = front - 0.5 * self.web_depth_m;
        let metal = a.part(Surface::Metal, label);
        self.sheet(metal, front, 1.);
        let mut rear_core = core;
        if let Some(spacing) = self.bank_spacing_m {
            // Opposing stamped faces leave an actual convection passage.
            rear_core = core - spacing;
            self.sheet(metal, rear_core - self.web_depth_m * 0.5, -1.);
            for side in [-1., 1.] {
                let x = side * self.body.width_m * 0.43;
                let y = if side < 0. {
                    self.body.bottom_m + self.header_m
                } else {
                    self.body.top_m - self.header_m
                };
                metal.rod(
                    Vec3::new(x, y, rear_core),
                    Vec3::new(x, y, core),
                    self.header_m * 0.36,
                );
            }
        }
        self.body
            .feet(a, o, rear_core, core, self.body.bottom_m + 0.004);
        self.body.plumbing(a, o, core);
    }

    fn sheet(self, g: &mut Geometry, z: f32, face: f32) {
        let b = self.body;
        let half = b.width_m * 0.5;
        let point = |x, y, depth| Vec3::new(x * face, y, z + face * depth);
        // Four rows track the start and end of each press. Both smooth headers
        // share the exact boundary of the corrugated field.
        let runout = self.header_m.min(b.height() * 0.09) * 0.67;
        let y = [
            b.bottom_m + self.header_m,
            b.bottom_m + self.header_m + runout,
            b.top_m - self.header_m - runout,
            b.top_m - self.header_m,
        ];
        let mut cols = Vec::with_capacity(self.channels * 5 + 3);
        cols.push((-half, 0.));
        for i in 0..self.channels {
            let start = -half + self.margin_m + i as f32 * self.pitch_m;
            for (u, amp) in [
                (0., 0.),
                (self.shoulder * 0.4, 0.18),
                (self.shoulder, 1.),
                (1. - self.shoulder, 1.),
                (1. - self.shoulder * 0.4, 0.18),
            ] {
                cols.push((start + self.pitch_m * u, amp));
            }
        }
        cols.extend([(half - self.margin_m, 0.), (half, 0.)]);
        let header = 0.53 * self.relief_m;
        let nx = cols.len();
        let start = g.positions.len() as u32;
        g.positions.reserve(4 * nx);
        g.normals.reserve(4 * nx);
        g.uvs.reserve(4 * nx);
        g.indices.reserve(18 * (nx - 1));
        for (row, &height) in y.iter().enumerate() {
            let field = row == 1 || row == 2;
            for (col, &(x, amp)) in cols.iter().enumerate() {
                let delta = self.relief_m * amp - header;
                let (dx, dy) = if field {
                    let (low, high) = (cols[col.saturating_sub(1)], cols[(col + 1).min(nx - 1)]);
                    (
                        self.relief_m * (high.1 - low.1) / (high.0 - low.0),
                        delta / (2. * runout),
                    )
                } else {
                    (0., delta / (2. * runout))
                };
                let dy = if row >= 2 { -dy } else { dy };
                let depth = if field { amp * self.relief_m } else { header };
                g.positions.push(point(x, height, depth).to_array());
                g.normals
                    .push(Vec3::new(-dx * face, -dy, face).normalize().to_array());
                g.uvs.push([x, height]);
                if row < 3 && col + 1 < nx {
                    let a = start + (row * nx + col) as u32;
                    let c = a + nx as u32;
                    g.indices.extend([a, a + 1, c + 1, a, c + 1, c]);
                }
            }
        }
        let weld = self.header_m * 0.23;
        let rear = -self.web_depth_m;
        let h = [
            (b.bottom_m, 0.),
            (b.bottom_m + weld, header),
            (y[0], header),
            (y[1], 0.),
            (y[2], 0.),
            (y[3], header),
            (b.top_m - weld, header),
            (b.top_m, 0.),
        ];
        let across = |a: (f32, f32), b: (f32, f32)| {
            [
                point(-half, a.0, a.1),
                point(half, a.0, a.1),
                point(half, b.0, b.1),
                point(-half, b.0, b.1),
            ]
        };
        for (i, bounds) in h.windows(2).enumerate() {
            let (lo, hi) = (bounds[0], bounds[1]);
            if !(2..5).contains(&i) {
                g.metric_quad(across(lo, hi), Vec3::Z * face);
            }
            for side in [-1., 1.] {
                let x = side * half;
                g.metric_quad(
                    [
                        point(x, lo.0, rear),
                        point(x, lo.0, lo.1),
                        point(x, hi.0, hi.1),
                        point(x, hi.0, rear),
                    ],
                    Vec3::X * side * face,
                );
            }
        }
        g.metric_quad(
            across((b.bottom_m, rear), (b.top_m, rear)),
            Vec3::NEG_Z * face,
        );
        g.metric_quad(across((b.bottom_m, rear), (b.bottom_m, 0.)), Vec3::NEG_Y);
        g.metric_quad(across((b.top_m, 0.), (b.top_m, rear)), Vec3::Y);
    }
}
