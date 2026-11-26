//! Die-cut paperboard on the actual tapered cup. Each strip owns its surface,
//! including the exposed raw edges and the thickness of the glued overlap.
use super::*;
use crate::scene::procedural_indoor::materials::{cup_board::Stock, drinkware::KRAFT_SLOT};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) enum Trim {
    Straight,
    ThumbCut,
    OffsetCuts,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct SleeveCut {
    pub stock: Stock,
    pub trim: Trim,
    pub seam_facet: u8,
    upper_center_turns: f32,
    lower_center_turns: f32,
    half_width_turns: f32,
    upper_depth_fraction: f32,
    lower_depth_fraction: f32,
    pub lap_width_m: f32,
    lip_width_m: f32,
    lap_fraction: f32,
}

impl SleeveCut {
    pub(super) fn sample(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"CupBlank"));
        let trim = match r.random_range(0..10) {
            0..=3 => Trim::Straight,
            4..=6 => Trim::ThumbCut,
            _ => Trim::OffsetCuts,
        };
        Self {
            stock: Stock::for_object(o),
            trim,
            seam_facet: r.random_range(0..32),
            upper_center_turns: r.random_range(0.15..0.85),
            lower_center_turns: r.random_range(0.05..0.95),
            half_width_turns: r.random_range(0.12..0.31),
            upper_depth_fraction: if trim == Trim::Straight {
                0.
            } else {
                r.random_range(0.10..0.30)
            },
            lower_depth_fraction: if trim == Trim::OffsetCuts {
                r.random_range(0.07..0.23)
            } else {
                0.
            },
            lap_width_m: r.random_range(0.004..0.013),
            lip_width_m: r.random_range(0.0008..0.0016),
            lap_fraction: r.random_range(0.24..0.43),
        }
    }

    fn notch(&self, u: f32, center: f32) -> f32 {
        let d = ((u - center + 0.5).rem_euclid(1.) - 0.5).abs() / self.half_width_turns;
        if d >= 1. {
            0.
        } else {
            0.5 + 0.5 * (PI * d).cos()
        }
    }

    pub(super) fn draw(
        &self,
        a: &mut Assembly,
        section: &mesh::Section,
        f: &CupForm,
        sleeve: &Sleeve,
        band: Vec2,
    ) {
        let label = ObjectKind::CoffeeCup.class_name();
        let height = band.y - band.x;
        let (r, slope) = section.radius_and_slope((band.x + band.y) * 0.5);
        let circumference = TAU * r;
        let lap_start = 1. - self.lap_width_m / circumference;
        let lip = 1. - self.lip_width_m / circumference;
        let sides = f.sides as usize;
        let mut turns: Vec<_> = (0..=sides).map(|i| i as f32 / sides as f32).collect();
        for u in [lap_start, lip] {
            if turns.iter().all(|t| (t - u).abs() > 0.00065) {
                turns.push(u);
            }
        }
        turns.sort_unstable_by(f32::total_cmp);
        let tf = f.transform();
        let columns: Vec<_> = turns
            .into_iter()
            .map(|u| {
                let phase = u * sides as f32;
                let side = (phase as usize + self.seam_facet as usize) % sides;
                // Additional lap columns lie on the cup's chord, not outside it.
                let meridian = outline(f.sides).facet_point(side, phase.fract());
                let low = band.x
                    + height * self.lower_depth_fraction * self.notch(u, self.lower_center_turns);
                let high = band.y
                    - height * self.upper_depth_fraction * self.notch(u, self.upper_center_turns);
                let lap = ((u - lap_start) / (lip - lap_start))
                    .clamp(0., 1.)
                    .min(((1. - u) / (1. - lip)).clamp(0., 1.));
                let thick = sleeve.thickness_m * (1. - self.lap_fraction * (1. - lap));
                let y = [f.heel_m, low, low, high, high, f.height_m - f.lip_m];
                let positions = std::array::from_fn(|i| {
                    let radius =
                        section.radius_at(y[i]) + if i == 2 || i == 3 { thick } else { 0. };
                    let q = meridian * radius;
                    tf.transform_point(Vec3::new(q.x, y[i], q.y))
                });
                Column {
                    u,
                    y,
                    positions,
                    thick,
                    radial: meridian.normalize(),
                }
            })
            .collect();
        let face_label = self.stock.label();
        let edge_label = format!("{label}#finish{KRAFT_SLOT}");
        for i in 0..5 {
            let facing = i == 2;
            let cut_edge = i == 1 || i == 3;
            let part = a.part(
                Surface::Paper,
                if facing {
                    &face_label
                } else if cut_edge {
                    &edge_label
                } else {
                    label
                },
            );
            for pair in columns.windows(2) {
                let (l, r) = (&pair[0], &pair[1]);
                let quad = [
                    l.positions[i],
                    l.positions[i + 1],
                    r.positions[i + 1],
                    r.positions[i],
                ];
                let normals = if cut_edge {
                    [(quad[2] - quad[0]).cross(quad[3] - quad[1]).normalize(); 4]
                } else {
                    let lift = if facing {
                        (r.thick - l.thick) / ((r.u - l.u) * circumference)
                    } else {
                        0.
                    };
                    let normal = |c: &Column| {
                        let radial = c.radial - c.radial.perp() * lift;
                        tf.rotation * Vec3::new(radial.x, -slope, radial.y).normalize()
                    };
                    [normal(l), normal(l), normal(r), normal(r)]
                };
                let uv = |c: &Column, j: usize| {
                    if facing {
                        // Print on the uncut blank. A notch trims the ink instead of stretching it.
                        [1. - c.u, (band.y - c.y[j]) / height]
                    } else if cut_edge {
                        [
                            c.u * circumference,
                            if j == 2 || j == 3 { c.thick } else { 0. },
                        ]
                    } else {
                        [c.u * circumference, c.y[j]]
                    }
                };
                part.vertex_quad(
                    quad,
                    normals,
                    [uv(l, i), uv(l, i + 1), uv(r, i + 1), uv(r, i)],
                );
            }
        }
    }
}

struct Column {
    u: f32,
    y: [f32; 6],
    positions: [Vec3; 6],
    thick: f32,
    radial: Vec2,
}
