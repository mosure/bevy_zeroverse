//! Wall-local art arrangements: coherent series, unequal frames and salon hangs.
//! Non-overlapping strips bound placement cost; staggered sizes/heights and
//! optional stacks avoid a fixed grid without rejection-heavy packing.
use super::*;

impl WallComposition {
    pub(super) fn art_panels(
        &self,
        width: f32,
        gap: f32,
        rng: &mut ChaCha8Rng,
    ) -> Vec<(ObjectKind, Vec2, Vec3)> {
        let aligned = rng.random_bool(self.aligned_fraction as f64);
        let columns = rng
            .random_range(1..=8)
            .min(((width + gap) / (0.28 + gap)).floor() as usize)
            .max(1);
        let stacked = !aligned && rng.random_bool(self.stacked_fraction as f64);
        let weights: Vec<f32> = (0..columns)
            .map(|_| {
                if aligned {
                    1.
                } else {
                    rng.random_range(-self.size_variation..self.size_variation)
                        .exp()
                }
            })
            .collect();
        let sum = weights.iter().sum::<f32>();
        let available = width - gap * (columns - 1) as f32 - 0.20 * columns as f32;
        let aspect = self.picture_aspect * rng.random_range(-0.40_f32..0.40).exp();
        let slope = rng.random_range(-0.18..0.18) * self.asymmetry;
        let mut panels = Vec::with_capacity(columns * if stacked { 2 } else { 1 });
        let mut left = -width * 0.5;
        for (column, weight) in weights.into_iter().enumerate() {
            let strip = 0.20 + available * weight / sum;
            let rows = if stacked && (column == 0 || rng.random_bool(0.70)) {
                rng.random_range(2..=3)
            } else {
                1
            };
            let mut sizes = Vec::with_capacity(rows);
            for _ in 0..rows {
                let w = (strip
                    * if aligned {
                        1.
                    } else {
                        rng.random_range(0.65..1.0)
                    })
                .clamp(0.18, 3.0);
                let a = if aligned {
                    aspect
                } else {
                    (aspect * rng.random_range(-1.10_f32..1.10).exp()).clamp(0.30, 3.6)
                };
                sizes.push(Vec3::new(
                    w,
                    (w / a).clamp(0.22, if rows > 1 { 0.82 } else { 1.85 }),
                    rng.random_range(0.024..0.060),
                ));
            }
            let stack_gap = gap * rng.random_range(0.80..1.20);
            let height = sizes.iter().map(|s| s.y).sum::<f32>() + stack_gap * (rows - 1) as f32;
            let center = if aligned {
                0.
            } else {
                rng.random_range(-self.height_variation_m..self.height_variation_m)
                    + slope * (left + strip * 0.5) / width
            };
            let mut bottom = center - height * 0.5;
            for size in sizes {
                let offset = if aligned {
                    0.
                } else {
                    rng.random_range(-0.5..0.5) * (strip - size.x)
                };
                panels.push((
                    ObjectKind::WallArt,
                    Vec2::new(left + strip * 0.5 + offset, bottom),
                    size,
                ));
                bottom += size.y + stack_gap;
            }
            left += strip + gap;
        }
        panels
    }
}

/// Group accepted art by its actual supporting wall plane, rather than yaw
/// alone (parallel walls must not be counted as one display). Used only by
/// distribution exports/tests; it adds no cost to generation.
pub(crate) fn mounted_art_walls(scene: &IndoorManifest) -> Vec<Vec<&IndoorObject>> {
    let mut walls: Vec<(Vec3, f32, Vec<&IndoorObject>)> = Vec::new();
    for o in scene
        .objects
        .iter()
        .filter(|o| o.kind == ObjectKind::WallArt && !o.neighbor && o.support.is_none())
    {
        let normal = Quat::from_rotation_y(o.yaw) * Vec3::Z;
        let plane = normal.dot(o.position)
            - crate::scene::procedural_indoor::objects::wall::mount_offset(o).unwrap();
        if let Some((_, _, members)) = walls
            .iter_mut()
            .find(|(n, d, _)| n.dot(normal) > 0.99999 && (*d - plane).abs() < 0.002)
        {
            members.push(o);
        } else {
            walls.push((normal, plane, vec![o]));
        }
    }
    walls.into_iter().map(|(_, _, members)| members).collect()
}

#[derive(Debug, Serialize)]
pub(crate) struct ArtWallStats {
    pub count: usize,
    pub pairs: usize,
    pub center_height_span_m: f32,
    pub area_ratio: f32,
    /// Centers within 3 cm; sizes within 5% on both axes.
    pub aligned_pairs: usize,
    pub matching_size_pairs: usize,
    /// Horizontal extents overlap, but the frames are vertically separated.
    pub stacked_pairs: usize,
}

impl ArtWallStats {
    pub(crate) fn measure(wall: &[&IndoorObject]) -> Self {
        let center = |o: &IndoorObject| o.position.y + o.size.y * 0.5;
        let range = |f: fn(&IndoorObject) -> f32| {
            wall.iter()
                .map(|o| f(o))
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
                    (lo.min(v), hi.max(v))
                })
        };
        let (low, high) = range(center);
        let (small, large) = range(|o| o.size.x * o.size.y);
        let mut stats = Self {
            count: wall.len(),
            pairs: 0,
            center_height_span_m: if wall.is_empty() { 0. } else { high - low },
            area_ratio: if wall.is_empty() { 1. } else { large / small },
            aligned_pairs: 0,
            matching_size_pairs: 0,
            stacked_pairs: 0,
        };
        for (i, a) in wall.iter().enumerate() {
            let tangent = Quat::from_rotation_y(a.yaw) * Vec3::X;
            for b in &wall[..i] {
                stats.pairs += 1;
                stats.aligned_pairs += usize::from((center(a) - center(b)).abs() <= 0.03);
                stats.matching_size_pairs += usize::from(
                    ((a.size.x / b.size.x).ln()).abs() <= 1.05_f32.ln()
                        && ((a.size.y / b.size.y).ln()).abs() <= 1.05_f32.ln(),
                );
                stats.stacked_pairs += usize::from(
                    tangent.dot(a.position - b.position).abs() < (a.size.x + b.size.x) * 0.5
                        && (center(a) - center(b)).abs() > (a.size.y + b.size.y) * 0.5,
                );
            }
        }
        stats
    }
}
