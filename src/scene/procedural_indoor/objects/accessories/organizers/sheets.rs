//! Fitted, registered paper stock. A hundred sheets costs the same as a short sheaf.
use super::*;
use crate::scene::procedural_indoor::{
    materials::variants, objects::literature::media::*, placement::rotated_half,
};

#[derive(Debug, Clone, serde::Serialize)]
pub struct PaperStack {
    pub media: &'static str,
    pub sheet_m: Vec2,
    pub sheets: u16,
    pub caliper_m: f32,
    pub center: Vec3,
    pub yaw_rad: f32,
    /// Offset of the bottom of the stack relative to the exposed leaf.
    pub registration_m: Vec2,
    pub printed_face_up: bool,
    pub curl_m: f32,
    pub curl_start: f32,
    pub curl_across: f32,
    pub print_slot: usize,
}

fn fit_stock(stock: Stock, usable: Vec2, twist: f32, landscape: bool) -> Option<(Stock, f32)> {
    let starts = if landscape {
        [FRAC_PI_2, 0.]
    } else {
        [0., FRAC_PI_2]
    };
    for start in starts {
        // Tight letter trays straighten large documents before falling back to smaller stock.
        for scale in [1., 0.5, 0.25, 0.] {
            let yaw = start + twist * scale;
            if (2. * rotated_half(stock.1, yaw)).cmple(usable).all() {
                return Some((stock, yaw));
            }
        }
    }
    None
}

impl PaperStack {
    pub(super) fn fit(o: &IndoorObject, b: &Bay) -> Option<Self> {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"BayPaper") + b.index as u64);
        let tray = o.kind == ObjectKind::DocumentTray;
        let usable = b.inner - Vec2::splat(0.0015);
        let twist = r.random_range(-0.21..0.21);
        let landscape = r.random_bool(if tray { 0.08 } else { 0.45 });
        let preferred = if tray {
            match r.random_range(0..100) {
                0..=36 => LETTER,
                37..=73 => A4,
                74..=87 => A5,
                88..=95 => PHOTO,
                _ => INDEX,
            }
        } else {
            match r.random_range(0..5) {
                0 => A6,
                1 => INDEX,
                2 => PHOTO,
                _ => NOTE,
            }
        };
        let slip = (
            "memo slip",
            Vec2::new(r.random_range(0.021..0.039), r.random_range(0.048..0.091)),
        );
        let (stock, mut yaw_rad) = [
            preferred, A4, LETTER, A5, A6, PHOTO, INDEX, NOTE, SMALL_NOTE, slip,
        ]
        .into_iter()
        .find_map(|stock| fit_stock(stock, usable, twist, landscape))?;
        if r.random_bool(0.37) {
            yaw_rad += PI;
        }
        let (media, sheet_m) = stock;
        let caliper_m = if matches!(media, "4x6" | "3x5") {
            r.random_range(0.00016..0.00030)
        } else {
            r.random_range(0.000086..0.000135)
        };
        // Leave room between tiers and for fingertips above the paper.
        let capacity = ((b.height - b.t) * 0.80).min(b.wall_h + 0.004);
        let max_sheets = (capacity / caliper_m).floor() as u16;
        if max_sheets == 0 {
            return None;
        }
        let sheets = (match r.random_range(0..10) {
            0..=1 => 1,
            2..=5 => r.random_range(2..=12),
            6..=8 => r.random_range(13..=46),
            _ => r.random_range(47..=136),
        })
        .min(max_sheets);
        let clearance = (usable - 2. * rotated_half(sheet_m, yaw_rad)).max(Vec2::ZERO);
        let drift = (f32::from(sheets - 1) * 0.000075).min(0.006);
        let registration_m = clearance.min(Vec2::splat(drift))
            * Vec2::new(r.random_range(-1.0..1.0), r.random_range(-1.0..1.0));
        let freedom = (clearance - registration_m.abs()) * 0.5;
        let offset = freedom * Vec2::new(r.random_range(-1.0..1.0), r.random_range(-1.0..1.0))
            - registration_m * 0.5;
        let curl_m = if r.random_bool(0.66) {
            r.random_range(0.014..0.060) * sheet_m.min_element()
        } else {
            0.0_f32
        };
        let curl_m = curl_m.min((b.height - b.t - f32::from(sheets) * caliper_m) * 0.65);
        Some(Self {
            media,
            sheet_m,
            sheets,
            caliper_m,
            yaw_rad,
            registration_m,
            center: Vec3::new(b.x + offset.x, b.base + b.t, offset.y),
            printed_face_up: r.random_bool(if matches!(media, "square memo" | "small memo") {
                0.25
            } else {
                0.82
            }),
            curl_m,
            curl_start: r.random_range(0.54..0.82),
            curl_across: r.random_range(-0.77..0.77),
            print_slot: variants::slot(o.seed ^ (b.index as u64).wrapping_mul(0x9e3779b97f4a7c15)),
        })
    }

    pub(super) fn build(&self, a: &mut Assembly) {
        let yaw = Quat::from_rotation_y(self.yaw_rad);
        let bulk = f32::from(self.sheets - 1) * self.caliper_m;
        let point = |u: f32, v: f32, y: f32, offset: Vec2| {
            self.center
                + yaw * Vec3::new((u - 0.5) * self.sheet_m.x, y, (v - 0.5) * self.sheet_m.y)
                + Vec3::new(offset.x, 0., offset.y)
        };
        let flat = |u, v, y| point(u, v, y, Vec2::ZERO);
        let fraction = |v: f32| ((v - self.curl_start) / (1. - self.curl_start)).max(0.);
        let warp = |u: f32, v: f32| {
            self.curl_m * fraction(v).powi(2) * (1. + (2. * u - 1.) * self.curl_across)
                / (1. + self.curl_across.abs())
        };
        let top = |u, v| flat(u, v, bulk + self.caliper_m + warp(u, v));
        let underside = |u, v| flat(u, v, bulk + warp(u, v));
        let normal = |u: f32, v: f32| {
            let f = self.curl_m / (1. + self.curl_across.abs());
            let dx = f * fraction(v).powi(2) * 2. * self.curl_across / self.sheet_m.x;
            let dz = f * 2. * fraction(v) * (1. + (2. * u - 1.) * self.curl_across)
                / ((1. - self.curl_start) * self.sheet_m.y);
            yaw * Vec3::new(-dx, 1., -dz).normalize()
        };
        let stations = [0., self.curl_start, 0.5 + 0.5 * self.curl_start, 1.];
        let cuts = if self.curl_m > 0. {
            &stations[..]
        } else {
            &[0., 1.][..]
        };
        let stock = a.part(Surface::Paper, "paper");
        // Only exposed surfaces are drawn. The bed supports the open underside;
        // an underlying page is visible only where the last leaf lifts away.
        for &[va, vb] in cuts.array_windows::<2>() {
            if va >= self.curl_start {
                let corners = [(0., va), (1., va), (1., vb), (0., vb)];
                stock.vertex_quad(
                    corners.map(|(u, v)| underside(u, v)),
                    corners.map(|(u, v)| -normal(u, v)),
                    corners.map(|(u, v)| [u * self.sheet_m.x, v * self.sheet_m.y]),
                );
                if self.sheets > 1 {
                    stock.metric_quad(corners.map(|(u, v)| flat(u, v, bulk)), Vec3::Y);
                }
            }
            for u in [0., 1.] {
                let outward = yaw * Vec3::X * (2. * u - 1.);
                if self.sheets > 1 {
                    stock.metric_quad(
                        [
                            point(u, va, 0., self.registration_m),
                            point(u, vb, 0., self.registration_m),
                            flat(u, vb, bulk),
                            flat(u, va, bulk),
                        ],
                        outward,
                    );
                }
                stock.metric_quad(
                    [underside(u, va), underside(u, vb), top(u, vb), top(u, va)],
                    outward,
                );
            }
        }
        for v in [0., 1.] {
            let outward = yaw * Vec3::Z * (2. * v - 1.);
            if self.sheets > 1 {
                stock.metric_quad(
                    [
                        point(0., v, 0., self.registration_m),
                        point(1., v, 0., self.registration_m),
                        flat(1., v, bulk),
                        flat(0., v, bulk),
                    ],
                    outward,
                );
            }
            stock.metric_quad(
                [underside(0., v), underside(1., v), top(1., v), top(0., v)],
                outward,
            );
        }
        let face = if self.printed_face_up {
            a.part(
                Surface::PrintedPaper,
                &format!("paper#finish{}#uv_local", self.print_slot),
            )
        } else {
            stock
        };
        for &[va, vb] in cuts.array_windows::<2>() {
            let corners = [(0., va), (0., vb), (1., vb), (1., va)];
            face.vertex_quad(
                corners.map(|(u, v)| top(u, v)),
                corners.map(|(u, v)| normal(u, v)),
                corners.map(|(u, v)| {
                    if self.printed_face_up {
                        [u, v]
                    } else {
                        [u * self.sheet_m.x, v * self.sheet_m.y]
                    }
                }),
            );
        }
    }
}
