//! A shared shallow room behind a window-facing storey. Neighboring bays look
//! into the same metric space, so the perspective and occlusions agree.
use super::*;
use bay::{Aperture, Face};

pub(super) struct Room {
    bounds: Aperture,
    front: f32,
    rear: f32,
    seed: u64,
}

impl Room {
    pub(super) fn sample(seed: u64, width: f32, depth: f32, pitch: f32, reveal: f32) -> Self {
        let mut rng = stream(seed, 0x524f4f4d444550);
        // Keep the room behind the returns of perpendicular elevations and
        // ahead of the opposite facade. The ceiling sits below the floor slab.
        let half = width * 0.5 - reveal - 0.004;
        Self {
            bounds: Aperture {
                lo: Vec2::new(-half, 0.016),
                hi: Vec2::new(half, pitch - 0.09),
            },
            front: reveal,
            rear: (reveal + rng.random_range(1.15..3.2)).min(depth * 0.74),
            seed,
        }
    }

    pub(super) fn build(&self, a: &mut Assembly, face: &Face) {
        let wall = finish(Surface::Paint, "wall", self.seed);
        face.panel(
            a.part(Surface::Paint, &wall),
            self.bounds.lo,
            self.bounds.hi,
            self.rear,
        );
        for (i, (surface, name)) in [
            (Surface::WoodEdge, "floor"),
            (Surface::Paint, "wall"),
            (Surface::Paint, "ceiling"),
            (Surface::Paint, "wall"),
        ]
        .into_iter()
        .enumerate()
        {
            face.return_panel(
                a.part(surface, &finish(surface, name, self.seed)),
                &self.bounds,
                i,
                self.front,
                self.rear,
            );
        }
        let mut r = stream(self.seed, 0x4655524e495348);
        let span = self.bounds.hi.x - self.bounds.lo.x;
        let space = self.rear - self.front;
        let width = (span * r.random_range(0.51..0.89)).min(r.random_range(0.95..2.08));
        if width < 0.54 || space < 0.62 || r.random_bool(0.13) {
            return;
        }
        let x = self.bounds.lo.x + width * 0.5 + (span - width) * r.random::<f32>();
        let floor = self.bounds.lo.y;
        let mut block = |surface, name, stock, at, size| {
            face.block(a.part(surface, &finish(surface, name, stock)), at, size);
        };
        if r.random_bool(0.48) {
            let depth = (space * 0.53).min(r.random_range(0.34..0.50));
            let z = self.rear - depth * 0.5 - r.random_range(0.025..0.085);
            let height = r.random_range(1.32..2.04);
            let stock = self.seed;
            let board = r.random_range(0.018..0.033);
            for x in [x - (width - board) * 0.5, x + (width - board) * 0.5] {
                block(
                    Surface::WoodEdge,
                    "bookshelf",
                    stock,
                    Vec3::new(x, floor + height * 0.5, z),
                    Vec3::new(board, height, depth),
                );
            }
            let levels = r.random_range(3..=5);
            let pitch = (height - board - 0.07) / (levels - 1) as f32;
            for i in 0..levels {
                let y = floor + 0.07 + pitch * i as f32;
                block(
                    Surface::WoodEdge,
                    "bookshelf",
                    stock,
                    Vec3::new(x, y + board * 0.5, z),
                    Vec3::new(width - 2. * board, board, depth),
                );
                if i + 1 == levels || r.random_bool(0.28) {
                    continue;
                }
                let mut u =
                    x - width * 0.5 + board + (width - 2. * board) * r.random_range(0.06..0.43);
                for _ in 0..r.random_range(2..=7) {
                    let w = r.random_range(0.024..0.068);
                    if u + w > x + width * 0.5 - board {
                        break;
                    }
                    let h = (pitch - board - 0.018).min(r.random_range(0.17..0.35));
                    let d = depth * r.random_range(0.44..0.76);
                    block(
                        Surface::Paint,
                        "books",
                        r.random(),
                        Vec3::new(u + w * 0.5, y + board + h * 0.5, z - depth * 0.40 + d * 0.5),
                        Vec3::new(w, h, d),
                    );
                    u += w + r.random_range(0.002..0.015);
                }
            }
        } else {
            let depth = (space * 0.58).min(r.random_range(0.49..0.76));
            let z = self.rear - depth * 0.5 - r.random_range(0.035..0.105);
            let h = r.random_range(0.71..0.81);
            let top = r.random_range(0.023..0.045);
            let stock = self.seed;
            for x in [x - width * 0.39, x + width * 0.39] {
                block(
                    Surface::WoodEdge,
                    "desk",
                    stock,
                    Vec3::new(x, floor + (h - top) * 0.5, z),
                    Vec3::new(0.043, h - top, depth * 0.84),
                );
            }
            block(
                Surface::WoodEdge,
                "desk",
                stock,
                Vec3::new(x, floor + h - top * 0.5, z),
                Vec3::new(width, top, depth),
            );
            if r.random_bool(0.85) {
                let sw = (width * 0.61).min(r.random_range(0.40..0.66));
                let sh = sw / r.random_range(1.45..1.91);
                let sx = x + (width - sw) * r.random_range(-0.30..0.30);
                let sz = z + depth * r.random_range(0.02..0.22);
                let lift = r.random_range(0.09..0.18);
                let y = floor + h;
                block(
                    Surface::Metal,
                    "television",
                    stock,
                    Vec3::new(sx, y + 0.009, sz),
                    Vec3::new(sw * 0.49, 0.018, depth * 0.33),
                );
                block(
                    Surface::Metal,
                    "television",
                    stock,
                    Vec3::new(sx, y + (lift + 0.018) * 0.5, sz),
                    Vec3::new(0.048, lift - 0.018, 0.043),
                );
                block(
                    Surface::Plastic,
                    "television",
                    stock,
                    Vec3::new(sx, y + lift + sh * 0.5, sz),
                    Vec3::new(sw, sh, 0.035),
                );
                block(
                    Surface::Ink,
                    "television",
                    stock,
                    Vec3::new(sx, y + lift + sh * 0.51, sz - 0.018),
                    Vec3::new(sw - 0.018, sh - 0.020, 0.0025),
                );
            }
        }
    }
}
