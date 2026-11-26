//! Metric stanchions and the hardware specific to each kind of span.
use super::*;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Infill {
    Webbing {
        width_m: f32,
        gauge_m: f32,
        sag_m: f32,
        reel_radius_m: f32,
        reel_left: bool,
    },
    Rope {
        radius_m: f32,
        end_reach_m: f32,
        ferrule_m: f32,
        catenary_q: f32,
    },
    Pickets {
        lower_y_m: f32,
        rail_radius_m: f32,
        picket_radius_m: f32,
        pickets: usize,
        square: bool,
    },
    Panel {
        bottom_m: f32,
        thickness_m: f32,
        side_gap_m: f32,
        clamp_inset_m: f32,
        jaw_reach_m: f32,
        finish: Surface,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct Installation {
    pub post_half_span_m: f32,
    pub post_radius_m: f32,
    pub base_radius_m: f32,
    pub base_height_m: f32,
    pub connection_y_m: f32,
    pub hardware: Surface,
    pub infill: Infill,
}

impl Installation {
    pub fn sample(o: &IndoorObject, p: &BarrierProgram) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"Stanchon"));
        let mut f = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"RailStok"),
        );
        let s = o.size;
        let base = s.z.min(s.x * 0.45) * p.base_radius;
        let x = s.x * 0.5 - base - 0.003;
        let post = (s.z * p.post_radius).min(base * 0.28);
        let y = s.y * p.rail_height;
        let span = 2. * (x - post);
        let hardware = if f.random_bool(0.43) {
            Surface::Chrome
        } else {
            Surface::Metal
        };
        let infill = match p.construction {
            0 => Infill::Webbing {
                width_m: p.belt_width.min((s.y - y) * 1.25),
                gauge_m: r.random_range(0.0012..0.0025),
                sag_m: span * r.random_range(0.001..0.014),
                reel_radius_m: (post * r.random_range(1.35..1.85)).min(base * 0.62),
                reel_left: r.random_bool(0.5),
            },
            1 => {
                let radius = r.random_range(0.011_f32..0.024).min(post * 0.99);
                let reach = (post + radius * r.random_range(1.30..2.25)).min(x * 0.35);
                let free_span = 2. * (x - reach);
                Infill::Rope {
                    radius_m: radius,
                    end_reach_m: reach,
                    ferrule_m: (radius * r.random_range(2.1..3.8)).min(free_span * 0.13),
                    // Half span / catenary scale. Limit the drop of a long rope over a low floor.
                    catenary_q: (4. * p.sag).min(1.7 * y / free_span),
                }
            }
            2 => Infill::Pickets {
                lower_y_m: y * r.random_range(0.17..0.44),
                rail_radius_m: post * r.random_range(0.53..0.82),
                picket_radius_m: post * r.random_range(0.29..0.54),
                pickets: ((span / r.random_range(0.12..0.26)).round() as usize).clamp(3, 18) - 1,
                square: r.random_bool(0.42),
            },
            _ => {
                let bottom = y * r.random_range(0.17..0.49);
                Infill::Panel {
                    bottom_m: bottom,
                    thickness_m: r.random_range(0.006..0.017),
                    side_gap_m: r.random_range(0.018_f32..0.064).min(span * 0.11),
                    clamp_inset_m: (y - bottom) * r.random_range(0.10..0.23),
                    jaw_reach_m: r.random_range(0.020_f32..0.047).min(span * 0.10),
                    finish: [
                        Surface::Wood,
                        Surface::Plastic,
                        Surface::Metal,
                        Surface::FabricAlt,
                    ][f.random_range(0..4)],
                }
            }
        };
        Self {
            post_half_span_m: x,
            post_radius_m: post,
            base_radius_m: base,
            base_height_m: (s.y * r.random_range(0.031..0.065)).min(base * 0.44),
            connection_y_m: y,
            hardware,
            infill,
        }
    }

    pub(crate) fn sag_m(&self) -> Option<f32> {
        match self.infill {
            Infill::Webbing { sag_m, .. } => Some(sag_m),
            Infill::Rope {
                end_reach_m,
                catenary_q,
                ..
            } => {
                let half = self.post_half_span_m - end_reach_m;
                Some(half / catenary_q * (2. * (0.5 * catenary_q).sinh().powi(2)))
            }
            _ => None,
        }
    }

    pub(super) fn build(&self, a: &mut Assembly, o: &IndoorObject, p: &BarrierProgram) {
        for side in [-1., 1.] {
            self.post(a, o, p, side);
        }
        spans::build(a, o.kind.class_name(), p, self);
    }

    fn post(&self, a: &mut Assembly, o: &IndoorObject, p: &BarrierProgram, side: f32) {
        let l = o.kind.class_name();
        let (r, foot, h) = (self.post_radius_m, self.base_radius_m, self.base_height_m);
        let x = side * self.post_half_span_m;
        let tf = Transform::from_xyz(x, 0., 0.);
        let sole = h * 0.14;
        if p.base == 0 {
            for (surface, width, bottom, top) in [
                (Surface::Rubber, foot * 1.71, 0., sole),
                (p.frame, foot * 1.86, sole * 0.86, h),
            ] {
                a.box_part(
                    surface,
                    l,
                    Vec3::new(x, (bottom + top) * 0.5, 0.),
                    Vec3::new(width, top - bottom, width),
                    (top - bottom) * 0.17,
                );
            }
        } else {
            let shoulder = if p.base == 1 { 0.53 } else { 0.88 };
            a.part(Surface::Rubber, l).lathe(
                &[(0., 0.), (foot * 0.92, 0.), (foot * 0.96, sole), (0., sole)],
                20,
                tf,
            );
            a.part(p.frame, l).lathe(
                &[
                    (0., sole * 0.84),
                    (foot, sole * 0.84),
                    (foot, h * 0.40),
                    (foot * shoulder, h),
                    (r * 1.25, h * 1.16),
                    (0., h * 1.16),
                ],
                24,
                tf,
            );
        }
        let start = if p.base == 0 { h } else { h * 1.16 };
        let reel =
            matches!(self.infill, Infill::Webbing { reel_left, .. } if reel_left == (side < 0.));
        let top = if reel { self.connection_y_m } else { o.size.y };
        let cap = if matches!(self.infill, Infill::Rope { .. }) {
            1.38
        } else {
            1.08
        };
        a.part(p.frame, l).lathe(
            &[
                (0., start - sole * 0.5),
                (r, start - sole * 0.5),
                (r, top - r * 1.6),
                (r * cap, top - r * 0.85),
                (r * cap, top - r * 0.55),
                (r * 0.78, top),
                (0., top),
            ],
            20,
            tf,
        );
    }
}
