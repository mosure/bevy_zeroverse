//! Plan cuts and the usable surface of a wall shelf, in local metres.
//! The placement predicate interpolates the same stations used by the mesh.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ShelfStock {
    RoutedTimber,
    BandedPanel,
    FoldedSteel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum EndCut {
    Square,
    Clipped,
    Round,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct ShelfEnd {
    pub cut: EndCut,
    pub width_m: f32,
    pub setback_m: f32,
}

impl ShelfEnd {
    // From the outside edge to the tangent on the straight front. The quarter
    // ellipse uses exact axis endpoints and five equal angular intervals.
    fn stations(self) -> &'static [(f32, f32)] {
        match self.cut {
            EndCut::Square => &[(0., 0.)],
            EndCut::Clipped => &[(0., 1.), (1., 0.)],
            EndCut::Round => &[
                (0., 1.),
                (0.0489435, 0.690983),
                (0.190983, 0.4122148),
                (0.4122148, 0.190983),
                (0.690983, 0.0489435),
                (1., 0.),
            ],
        }
    }

    fn setback(self, from_end: f32) -> f32 {
        if from_end >= self.width_m || self.cut == EndCut::Square {
            return 0.;
        }
        let t = from_end.max(0.) / self.width_m;
        for pair in self.stations().windows(2) {
            let [(a, za), (b, zb)] = [pair[0], pair[1]];
            if t <= b {
                return self.setback_m * za.lerp(zb, (t - a) / (b - a));
            }
        }
        0.
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ShelfDeck {
    pub stock: ShelfStock,
    pub ends: [ShelfEnd; 2],
    /// Total fascia depth. A folded sheet bears on its much thinner top skin.
    pub depth_m: f32,
    pub nose_m: f32,
    pub lower_run_m: f32,
    pub gauge_m: f32,
    pub return_m: f32,
    pub top_fall_m: f32,
    pub bottom_rise_m: f32,
}

impl ShelfDeck {
    pub fn sample(o: &IndoorObject) -> Self {
        let mut rng = stream(o.seed, u64::from_le_bytes(*b"ShelfCut"));
        let stock = match rng.random_range(0..10) {
            0..=3 => ShelfStock::RoutedTimber,
            4..=7 => ShelfStock::BandedPanel,
            _ => ShelfStock::FoldedSteel,
        };
        let end = |r: &mut rand_chacha::ChaCha8Rng| {
            let cut = match r.random_range(0..10) {
                0..=3 => EndCut::Square,
                4..=6 => EndCut::Clipped,
                _ => EndCut::Round,
            };
            if cut == EndCut::Square {
                return ShelfEnd {
                    cut,
                    width_m: 0.,
                    setback_m: 0.,
                };
            }
            ShelfEnd {
                cut,
                width_m: (o.size.z * r.random_range(0.07..0.74)).min(o.size.x * 0.20),
                setback_m: o.size.z * r.random_range(0.08..0.43),
            }
        };
        let left = end(&mut rng);
        // Matched ends are common on manufactured shelves. A wall return or
        // cut-to-fit installation can use two different ends.
        let right = if rng.random_bool(0.67) {
            left
        } else {
            end(&mut rng)
        };
        let depth_m = (match stock {
            ShelfStock::RoutedTimber => rng.random_range(0.017_f32..0.058),
            ShelfStock::BandedPanel => rng.random_range(0.013..0.039),
            ShelfStock::FoldedSteel => rng.random_range(0.018..0.051),
        })
        .min(o.size.y * 0.32)
        .min(o.size.z * 0.30);
        let (nose_m, lower_run_m, gauge_m, return_m, top_fall_m, bottom_rise_m) = match stock {
            ShelfStock::RoutedTimber => {
                let top = depth_m * rng.random_range(0.14..0.44);
                let bottom = depth_m * rng.random_range(0.20..0.47);
                (
                    top * rng.random_range(0.78..1.38),
                    bottom * rng.random_range(0.82..1.44),
                    depth_m,
                    0.,
                    top,
                    bottom,
                )
            }
            ShelfStock::BandedPanel => {
                let eased = depth_m * rng.random_range(0.023..0.12);
                (
                    eased,
                    depth_m * rng.random_range(0.08..0.64),
                    depth_m,
                    0.,
                    eased,
                    depth_m * rng.random_range(0.10..0.40),
                )
            }
            ShelfStock::FoldedSteel => {
                let gauge = rng.random_range(0.0009_f32..0.0019).min(depth_m * 0.085);
                let bend = gauge * rng.random_range(1.7..3.0);
                (
                    bend,
                    bend,
                    gauge,
                    rng.random_range(0.005_f32..0.016).min(o.size.z * 0.06),
                    bend,
                    bend,
                )
            }
        };
        Self {
            stock,
            ends: [left, right],
            depth_m,
            nose_m,
            lower_run_m,
            gauge_m,
            return_m,
            top_fall_m,
            bottom_rise_m,
        }
    }

    /// (spanwise X, front Z), from left to right, in shelf space.
    pub(super) fn stations(&self, half: Vec2) -> impl Iterator<Item = (f32, f32)> + '_ {
        let [a, b] = self.ends;
        a.stations()
            .iter()
            .map(move |&(t, z)| (-half.x + t * a.width_m, half.y - z * a.setback_m))
            .chain(
                b.stations()
                    .iter()
                    .rev()
                    .map(move |&(t, z)| (half.x - t * b.width_m, half.y - z * b.setback_m)),
            )
    }

    pub(super) fn front_at(&self, half: Vec2, x: f32) -> f32 {
        half.y - self.ends[0].setback(half.x + x) - self.ends[1].setback(half.x - x)
    }

    /// Checking the four corners suffices because the usable top is convex.
    fn contains(&self, half: Vec2, p: Vec2, margin: f32) -> bool {
        p.x.abs() + margin <= half.x
            && p.y - margin >= -half.y
            && p.y + margin
                <= self
                    .front_at(half, p.x - margin)
                    .min(self.front_at(half, p.x + margin))
                    - self.nose_m
    }

    pub(crate) fn supports(
        &self,
        shelf: &IndoorObject,
        center: Vec2,
        size: Vec2,
        yaw: f32,
        margin: f32,
    ) -> bool {
        let (s, c) = yaw.sin_cos();
        let u = Vec2::new(c, -s) * (0.5 * size.x);
        let v = Vec2::new(s, c) * (0.5 * size.y);
        [
            center + u + v,
            center + u - v,
            center - u + v,
            center - u - v,
        ]
        .into_iter()
        .all(|p| self.contains(shelf.size.xz() * 0.5, p, margin))
    }
}
