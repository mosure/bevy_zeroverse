//! A freestanding collection with shelving on both sides of one structural spine.
use super::*;
mod display;
mod stock;
pub use display::{BookRest, FeaturedBook};
#[cfg(test)]
mod tests;

pub fn dimensions(seed: u64) -> Vec3 {
    let mut r = stream(seed, u64::from_le_bytes(*b"LibSizes"));
    Vec3::new(
        r.random_range(1.05..2.45),
        r.random_range(0.99..2.12),
        r.random_range(0.50..0.76),
    )
}

/// Clear floor depth in metres, measured outward from either shelf front.
pub fn browsing_depth_m(o: &IndoorObject) -> f32 {
    0.78 + (o.seed >> 48) as f32 / u16::MAX as f32 * 0.34
}

#[derive(Debug, serde::Serialize)]
pub struct Stack {
    pub bays: usize,
    pub levels: usize,
    pub open_ends: bool,
    pub frame: Surface,
    pub deck: Surface,
    pub board_m: f32,
    pub base_m: f32,
    pub fill: f32,
}

impl Stack {
    pub fn sample(o: &IndoorObject) -> Self {
        let mut r = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"LibFrame"),
        );
        let open_ends = r.random_bool(0.48);
        let mut stock = stream(o.seed, u64::from_le_bytes(*b"LibStock"));
        Self {
            bays: (o.size.x / r.random_range(0.66..0.96))
                .round()
                .clamp(1., 3.) as usize,
            levels: (o.size.y / r.random_range(0.31..0.42))
                .floor()
                .clamp(2., 6.) as usize,
            open_ends,
            frame: if open_ends {
                Surface::Metal
            } else {
                Surface::Wood
            },
            deck: if r.random_bool(if open_ends { 0.63 } else { 0.16 }) {
                Surface::Metal
            } else {
                Surface::Wood
            },
            board_m: r.random_range(0.018..0.032),
            base_m: r.random_range(0.06..0.14),
            fill: stock.random_range(0.48..0.91),
        }
    }
    fn cell(&self, size: Vec3, bay: usize, level: usize, side: i8) -> stock::Cell {
        let t = self.board_m;
        let bay_width = (size.x - t) / self.bays as f32;
        let rise = (size.y - t - self.base_m) / self.levels as f32;
        let y = self.base_m + t * 0.5 + level as f32 * rise;
        stock::Cell {
            left: -size.x * 0.5 + t + bay as f32 * bay_width,
            width: bay_width - t,
            floor: y + t * 0.5,
            height: rise - t,
            depth: (size.z - t) * 0.5,
            outer: size.z * 0.5,
            side,
        }
    }

    /// Face-out titles and their stands in stack-local metres, reconstructed
    /// from an archived object and its room's generator version.
    pub fn featured_books(&self, o: &IndoorObject, version: u32) -> Vec<FeaturedBook> {
        let Some(stock) = display::Selection::for_stack(o, version) else {
            return Vec::new();
        };
        let mut books = Vec::new();
        for bay in 0..self.bays {
            for side in [-1, 1] {
                for level in 0..self.levels {
                    let mut cell = self.cell(o.size, bay, level, side);
                    if let Some(book) = stock.fit(o.seed, bay, level, &mut cell) {
                        books.push(book);
                    }
                }
            }
        }
        books
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    let p = Stack::sample(o);
    let displays = display::Selection::for_stack(o, version);
    let s = o.size;
    let t = p.board_m;
    let l = o.kind.class_name();
    let mut cuts = TimberCuts::new(o.finish_seed.unwrap_or(o.seed), 0);
    let bay_width = (s.x - t) / p.bays as f32;
    let rise = (s.y - t - p.base_m) / p.levels as f32;
    let depth = (s.z - t) * 0.5;
    a.board(
        &mut cuts,
        (p.frame, l),
        Vec3::Y * (p.base_m * 0.5),
        Vec3::new(s.x - 2. * t, p.base_m, s.z * 0.76),
        0.002,
        GrainAxis::X,
    );
    for i in 0..=p.bays {
        let x = -s.x * 0.5 + t * 0.5 + i as f32 * bay_width;
        if p.open_ends {
            for sign in [-1., 1.] {
                a.box_part(
                    p.frame,
                    l,
                    Vec3::new(x, s.y * 0.5, sign * (s.z * 0.5 - t * 0.5)),
                    Vec3::new(t, s.y, t),
                    0.002,
                );
            }
        } else {
            a.board(
                &mut cuts,
                (p.frame, l),
                Vec3::new(x, s.y * 0.5, 0.),
                Vec3::new(t, s.y, s.z),
                0.002,
                GrainAxis::Y,
            );
        }
    }
    for i in 0..p.bays {
        let left = -s.x * 0.5 + t + i as f32 * bay_width;
        let width = bay_width - t;
        let x = left + width * 0.5;
        a.board(
            &mut cuts,
            (p.frame, l),
            Vec3::new(x, (s.y + p.base_m) * 0.5, 0.),
            Vec3::new(width, s.y - p.base_m, t),
            0.0015,
            GrainAxis::X,
        );
        for side in [-1_i8, 1] {
            let z = side as f32 * (t + depth) * 0.5;
            for j in 0..=p.levels {
                let y = p.base_m + t * 0.5 + j as f32 * rise;
                a.board(
                    &mut cuts,
                    (p.deck, l),
                    Vec3::new(x, y, z),
                    Vec3::new(width, t, depth),
                    0.0015,
                    GrainAxis::X,
                );
                if j < p.levels {
                    let mut cell = p.cell(s, i, j, side);
                    if let Some(book) =
                        displays.and_then(|stock| stock.fit(o.seed, i, j, &mut cell))
                    {
                        book.build(a, version);
                    }
                    cell.books(o.seed, cell.index(i, j), p.fill, version, |book| {
                        book.build(a, version)
                    });
                }
            }
        }
    }
}
