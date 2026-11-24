//! The exhaust and expansion covers occupy measured clear bands of rear sheet.
//! Stamped perforations use metric, mip-filtered maps with an opaque backing.
use super::*;
use super::{face::Plane, sockets::Socket};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Exhaust {
    pub opening: Rect,
    pub round: bool,
    pub depth_m: f32,
    pub expansion: Option<Rect>,
    pub blanking_plates: usize,
}

impl Exhaust {
    pub(super) fn fit(o: &IndoorObject, face: Rect, sockets: &[Socket; 5]) -> Option<Self> {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"RearVent"));
        let field = face.inflate(-0.0024);
        let mut cuts = sockets.map(|s| s.aperture.inflate(0.0022));
        cuts.sort_by(|a, b| a.min.y.total_cmp(&b.min.y));
        let mut gaps = [Vec2::ZERO; 6];
        let mut n = 0;
        let mut y = field.min.y;
        for h in cuts {
            if h.min.y > y {
                gaps[n] = Vec2::new(y, h.min.y);
                n += 1;
            }
            y = y.max(h.max.y);
        }
        if field.max.y > y {
            gaps[n] = Vec2::new(y, field.max.y);
            n += 1;
        }
        let rows = &mut gaps[..n];
        // Favor enough sheet for the assembly, with either upper or lower exhaust.
        rows.sort_by(|a, b| (b.y - b.x).total_cmp(&(a.y - a.x)));
        let first = *rows.first()?;
        let h = first.y - first.x;
        let room = field.width().min(h);
        let round = o.size.y > o.size.x * 0.67 && room >= 0.042 && r.random_bool(0.82);
        let size = if round {
            let d = ([0.12, 0.092, 0.080, 0.070, 0.060, 0.050, 0.040]
                .into_iter()
                .find(|d| *d <= room)
                .unwrap_or(0.040))
                * r.random_range(0.91..0.99);
            Vec2::splat(d)
        } else {
            if h < 0.0045 {
                return None;
            }
            Vec2::new(
                field.width() * r.random_range(0.45..0.87),
                (h * r.random_range(0.48..0.84)).min(0.063),
            )
        };
        let position = |size: Vec2, band: Vec2, t: Vec2| {
            Vec2::new(field.min.x, band.x)
                + size * 0.5
                + (Vec2::new(field.width(), band.y - band.x) - size) * t
        };
        let opening = Rect::from_center_size(
            position(
                size,
                first,
                Vec2::new(r.random_range(0.09..0.91), r.random_range(0.15..0.85)),
            ),
            size,
        );
        let mut expansion = None;
        let mut blanking_plates = 0;
        if o.size.x > 0.13 && o.size.y > 0.24 && r.random_bool(0.77) {
            if let Some(&band) = rows.get(1) {
                let pitch = r.random_range(0.0155..0.0205);
                blanking_plates = (((band.y - band.x) / pitch) as usize).min(r.random_range(2..=4));
                if blanking_plates > 0 {
                    let size = Vec2::new(
                        (field.width() * r.random_range(0.68..0.92)).min(0.124),
                        blanking_plates as f32 * pitch,
                    );
                    expansion = Some(Rect::from_center_size(
                        position(size, band, Vec2::new(r.random_range(0.14..0.86), 0.5)),
                        size,
                    ));
                }
            }
        }
        Some(Self {
            opening,
            round,
            depth_m: r.random_range(0.0010..0.0028),
            expansion,
            blanking_plates,
        })
    }

    pub(super) fn build(
        &self,
        a: &mut Assembly,
        plane: Plane,
        stock: &Housing,
        label: &str,
        grille: &str,
    ) {
        let r = self.opening;
        let d = self.depth_m;
        if self.round {
            const N: usize = 24;
            static RING: std::sync::OnceLock<[Vec2; N]> = std::sync::OnceLock::new();
            let circle = RING.get_or_init(|| {
                std::array::from_fn(|i| Vec2::from_angle((i as f32 / N as f32 - 0.375) * TAU))
            });
            let pos = |u: Vec2| r.center() + r.size() * 0.5 * u;
            for i in 0..N {
                let j = (i + 1) % N;
                let (u, v) = (circle[i], circle[j]);
                let (p, q) = (
                    pos(u / u.abs().max_element()),
                    pos(v / v.abs().max_element()),
                );
                let (lip, end) = (pos(u * 0.985), pos(v * 0.985));
                let (inner, next) = (pos(u * 0.93), pos(v * 0.93));
                let sheet = a.part(stock.surface, label);
                // Keep the sheet around the circular punch flat. Sloping the
                // entire square to the grille creates a faceted square highlight.
                sheet.metric_quad(
                    [
                        plane.point(p, 0.),
                        plane.point(q, 0.),
                        plane.point(end, 0.),
                        plane.point(lip, 0.),
                    ],
                    plane.normal(),
                );
                sheet.metric_quad(
                    [
                        plane.point(lip, 0.),
                        plane.point(end, 0.),
                        plane.point(next, d),
                        plane.point(inner, d),
                    ],
                    plane.normal(),
                );
                super::sockets::cap_triangle(
                    a.part(Surface::Metal, grille),
                    plane,
                    [r.center(), inner, next],
                    d,
                );
            }
        } else {
            plane.walls(a.part(stock.surface, label), r, d);
            plane.rect(a.part(Surface::Metal, grille), r, d);
        }
        if let Some(bank) = self.expansion {
            let pitch = bank.height() / self.blanking_plates as f32;
            for i in 0..self.blanking_plates {
                let lo = bank.min.y + pitch * i as f32;
                let inner = Rect::new(bank.min.x, lo + 0.001, bank.max.x, lo + pitch - 0.001);
                plane.rect(
                    a.part(stock.surface, label),
                    Rect::new(bank.min.x, lo, bank.max.x, inner.min.y),
                    0.,
                );
                plane.rect(
                    a.part(stock.surface, label),
                    Rect::new(bank.min.x, inner.max.y, bank.max.x, lo + pitch),
                    0.,
                );
                plane.walls(a.part(stock.surface, label), inner, 0.0012);
                plane.rect(a.part(Surface::Metal, "pc_tower"), inner, 0.0012);
            }
        }
    }
}
