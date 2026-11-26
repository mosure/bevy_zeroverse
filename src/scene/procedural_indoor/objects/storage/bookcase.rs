//! Fitted shelving programs: physical bays and independent adjustable levels.
use super::*;
mod binding;
pub(crate) mod contents;
mod rack;
mod runs;
pub use rack::{RackBracing, RackFrame};
pub(crate) const RACK_VERSION: u32 = 257;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ShelfBay {
    pub left: f32,
    pub right: f32,
    /// Shelf centre heights, including the bottom and top boards.
    pub levels: Vec<f32>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BookcaseProgram {
    pub open_frame: bool,
    pub closed_back: bool,
    pub panel_m: f32,
    pub fill: f32,
    pub bays: Vec<ShelfBay>,
    pub frame: Surface,
    pub shelves: Surface,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rack: Option<RackFrame>,
}

pub fn parameters(o: &IndoorObject) -> BookcaseProgram {
    parameters_at(
        o,
        crate::scene::procedural_indoor::layout::GENERATOR_VERSION,
    )
}

pub(crate) fn parameters_at(o: &IndoorObject, version: u32) -> BookcaseProgram {
    let mut r = stream(o.seed, 0x424f_4f4b_4341_5345);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x424f_4f4b_4649_4e49);
    let t = r.random_range(0.016..0.029);
    let n = (o.size.x / r.random_range(0.36..0.76))
        .round()
        .clamp(1., 4.) as usize;
    let bottom = r.random_range(0.04..0.13);
    let top = o.size.y - t * 0.5;
    let count = ((top - bottom) / r.random_range(0.30..0.48))
        .round()
        .clamp(2., 8.) as usize;
    let weights: Vec<f32> = (0..n).map(|_| r.random_range(0.75..1.35)).collect();
    let sum: f32 = weights.iter().sum();
    let mut left = -o.size.x * 0.5 + t;
    let aligned = r.random_bool(0.32);
    let mut bays: Vec<ShelfBay> = Vec::with_capacity(n);
    for weight in weights {
        let right = left + (o.size.x - t * (n + 1) as f32) * weight / sum;
        let levels = if aligned && !bays.is_empty() {
            bays[0].levels.clone()
        } else {
            let spaces: Vec<f32> = (0..count).map(|_| r.random_range(0.6..1.4)).collect();
            let sum: f32 = spaces.iter().sum();
            let minimum = ((top - bottom) / count as f32 * 0.68).min(0.24);
            let free = top - bottom - minimum * count as f32;
            let mut levels = vec![bottom];
            for weight in spaces {
                levels.push(levels.last().unwrap() + minimum + free * weight / sum);
            }
            *levels.last_mut().unwrap() = top;
            levels
        };
        bays.push(ShelfBay {
            left,
            right,
            levels,
        });
        left = right + t;
    }
    let open_frame = r.random_bool(0.38);
    let rack = (open_frame && version >= RACK_VERSION).then(|| RackFrame::sample(o, &mut bays, t));
    BookcaseProgram {
        open_frame,
        closed_back: !open_frame && r.random_bool(0.75),
        panel_m: t,
        fill: r.random_range(0.22..0.90),
        bays,
        frame: if open_frame {
            [Surface::Metal, Surface::Wood][f.random_range(0..2)]
        } else {
            [Surface::Wood, Surface::Plastic][f.random_range(0..2)]
        },
        shelves: [Surface::Wood, Surface::Plastic, Surface::Metal][f.random_range(0..3)],
        rack,
    }
}

impl BookcaseProgram {
    /// Front and rear edges of the load-bearing deck, in object coordinates.
    pub(crate) fn deck_limits(&self, depth: f32) -> (f32, f32) {
        let front = depth * 0.5;
        if let Some(rack) = &self.rack {
            return (
                -front + rack.brace_depth_m + rack.post_depth_m + rack.reveal_m,
                front - rack.post_depth_m - rack.reveal_m,
            );
        }
        (
            -front + if self.closed_back { self.panel_m } else { 0. },
            front,
        )
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    let p = parameters_at(o, version);
    let s = o.size;
    let t = p.panel_m;
    let l = o.kind.class_name();
    let mut cuts = TimberCuts::new(o.seed, 0);
    let (rear, front) = p.deck_limits(s.z);
    let z = (rear + front) * 0.5;
    if p.closed_back {
        a.board(
            &mut cuts,
            (p.frame, l),
            Vec3::new(0., s.y * 0.5, -s.z * 0.5 + t * 0.5),
            Vec3::new(s.x, s.y, t),
            0.002,
            GrainAxis::Y,
        );
    }
    let post_width = p.rack.as_ref().map_or(t, |rack| rack.post_width_m);
    for x in std::iter::once(-s.x * 0.5 + post_width * 0.5)
        .chain(p.bays.iter().map(|b| b.right + post_width * 0.5))
    {
        if let Some(rack) = &p.rack {
            rack.posts(a, o, &p, &mut cuts, x);
        } else if p.open_frame {
            for z in [rear + t * 0.5, front - t * 0.5] {
                a.board(
                    &mut cuts,
                    (p.frame, l),
                    Vec3::new(x, s.y * 0.5, z),
                    Vec3::new(t, s.y, t),
                    0.002,
                    GrainAxis::Y,
                );
            }
        } else {
            a.board(
                &mut cuts,
                (p.frame, l),
                Vec3::new(x, s.y * 0.5, z),
                Vec3::new(t, s.y, front - rear),
                0.002,
                GrainAxis::Y,
            );
        }
    }
    if let Some(rack) = &p.rack {
        rack.build_supports(a, o, &p);
    }
    for (index, b) in p.bays.iter().enumerate() {
        let bearing = p.rack.as_ref().map_or(0., |rack| rack.bearing_m);
        for &y in &b.levels {
            a.board(
                &mut cuts,
                (p.shelves, l),
                Vec3::new((b.left + b.right) * 0.5, y, z),
                Vec3::new(b.right - b.left + 2. * bearing, t, front - rear),
                0.002,
                GrainAxis::X,
            );
        }
        for level in 0..b.levels.len() - 1 {
            for child in contents::placements(o, &p, index, level) {
                contents::build(a, o, &child, version);
            }
        }
    }
}

#[cfg(test)]
pub(super) mod tests;
