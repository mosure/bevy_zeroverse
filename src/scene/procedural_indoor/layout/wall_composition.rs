//! Coherent installations on real opaque walls. A group is accepted atomically,
//! with the same reveal, opening, ceiling and occupied-space tests as wall props.
use super::super::architecture::facade::FacadeSide;
use super::*;
mod art;
mod presentation;
pub(crate) use art::{mounted_art_walls, ArtWallStats};
pub(super) use presentation::standalone_size;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct WallComposition {
    /// Gallery, presentation and mixed media weights, normalized per room.
    pub weights: [f32; 3],
    pub coverage: f32,
    pub picture_aspect: f32,
    pub spacing_m: f32,
    pub eye_height_m: f32,
    pub asymmetry: f32,
    pub size_variation: f32,
    pub height_variation_m: f32,
    pub aligned_fraction: f32,
    pub stacked_fraction: f32,
}

impl WallComposition {
    pub(crate) fn sample(scene: &IndoorManifest) -> Self {
        let mut rng = stream(scene.seed, 5810);
        let mut weights = [0, 1, 2].map(|_| rng.random_range(0.12..1.0));
        if scene.layout.prefers_soft_seating()
            || matches!(
                scene.layout,
                IndoorLayout::Studio | IndoorLayout::Gallery | IndoorLayout::Showroom
            )
        {
            weights[0] *= 2.5;
        } else if matches!(
            scene.layout,
            IndoorLayout::Conference
                | IndoorLayout::Training
                | IndoorLayout::Classroom
                | IndoorLayout::LectureHall
        ) {
            weights[1] *= 2.5;
        }
        let sum = weights.iter().sum::<f32>();
        Self {
            weights: weights.map(|w| w / sum),
            coverage: rng.random_range(0.16..0.88),
            picture_aspect: rng.random_range(0.35_f32.ln()..3.2_f32.ln()).exp(),
            spacing_m: rng.random_range(0.055..0.32),
            eye_height_m: rng.random_range(1.35..1.85),
            asymmetry: rng.random_range(0.15..1.0),
            size_variation: rng.random_range(0.40..1.20),
            height_variation_m: rng.random_range(0.16..0.50),
            aligned_fraction: rng.random_range(0.10..0.30),
            stacked_fraction: rng.random_range(0.20..0.60),
        }
    }

    fn choose(&self, rng: &mut ChaCha8Rng) -> usize {
        let mut p = rng.random::<f32>();
        for (i, weight) in self.weights.iter().enumerate() {
            p -= weight;
            if p <= 0. {
                return i;
            }
        }
        2
    }

    /// Installation offsets in wall coordinates. Z is supplied by the actual
    /// mounting contract; Y is the lower edge of each independently framed panel.
    fn panels(
        &self,
        family: usize,
        span: f32,
        rng: &mut ChaCha8Rng,
    ) -> Vec<(ObjectKind, Vec2, Vec3)> {
        let width = (span * self.coverage * rng.random_range(0.65..1.25))
            .clamp(0.80, 5.8)
            .min(span - 0.24);
        if width < 0.65 {
            return Vec::new();
        }
        let gap = self.spacing_m * rng.random_range(0.8..1.2);
        let mut panels = Vec::new();
        if family == 0 {
            return self.art_panels(width, gap, rng);
        } else {
            let kind = if rng.random_bool(0.58) {
                ObjectKind::Whiteboard
            } else {
                ObjectKind::Display
            };
            let aspect = presentation::aspect(kind, rng);
            let screen_width = if family == 2 {
                width * rng.random_range(0.42..0.82)
            } else {
                width * rng.random_range(0.65..1.0)
            };
            let screen_width = screen_width.min(aspect * rng.random_range(1.20..1.90));
            let height = screen_width / aspect;
            let x = if family == 2 {
                -(width - screen_width) * 0.5
            } else {
                0.
            };
            panels.push((
                kind,
                Vec2::new(x, -height * 0.5),
                Vec3::new(screen_width, height, rng.random_range(0.045..0.115)),
            ));
            if family == 2 {
                let side_width = width - screen_width - gap;
                if side_width >= 0.36 {
                    // Art beside a display has its own vertical extent; tying
                    // both small works to the screen height made them clones.
                    let stack_height = (height * rng.random_range(0.85..1.35)).clamp(0.65, 1.70);
                    let side_gap = gap.min(stack_height * 0.20);
                    let count = rng.random_range(1..=3);
                    let weights: Vec<f32> =
                        (0..count).map(|_| rng.random_range(0.4..1.6)).collect();
                    let sum = weights.iter().sum::<f32>();
                    let heights = weights
                        .into_iter()
                        .map(|v| v / sum * (stack_height - side_gap * (count - 1) as f32));
                    let mut y = -stack_height * 0.5 + rng.random_range(-0.20..0.20);
                    for side_height in heights {
                        let pane_width =
                            (side_width * rng.random_range(0.72..1.0)).min(side_height * 2.7);
                        panels.push((
                            ObjectKind::WallArt,
                            Vec2::new(
                                width * 0.5 - side_width * 0.5
                                    + rng.random_range(-0.5..0.5) * (side_width - pane_width),
                                y,
                            ),
                            Vec3::new(pane_width, side_height, rng.random_range(0.024..0.055)),
                        ));
                        y += side_height + side_gap;
                    }
                }
            }
        }
        panels
    }
}

impl IndoorManifest {
    pub(super) fn compose_wall_groups(&mut self) {
        let program = WallComposition::sample(self);
        let mut rng = stream(self.seed, 5811);
        let groups = (rng.random_range(0.5..5.5) * (0.40 + self.density)).round() as usize;
        let walls: Vec<_> = if let Some(e) = &self.envelope {
            e.walls
                .iter()
                .map(|w| {
                    let a = e.footprint[w.edge];
                    let b = e.footprint[(w.edge + 1) % e.footprint.len()];
                    (e.wall_transform(w.edge), a.distance(b))
                })
                .collect()
        } else {
            FacadeSide::ALL
                .into_iter()
                .map(|side| (side.transform(self.room_size), side.span(self.room_size)))
                .collect()
        };
        for _ in 0..groups {
            let family = program.choose(&mut rng);
            for _ in 0..20 {
                let (tf, span) = walls[rng.random_range(0..walls.len())];
                let panels = program.panels(family, span, &mut rng);
                if panels.is_empty() {
                    continue;
                }
                let min = panels
                    .iter()
                    .map(|(_, p, s)| p.x - s.x * 0.5)
                    .fold(f32::INFINITY, f32::min);
                let max = panels
                    .iter()
                    .map(|(_, p, s)| p.x + s.x * 0.5)
                    .fold(f32::NEG_INFINITY, f32::max);
                let range = (span - (max - min)) * 0.5 - 0.08;
                if range <= 0. {
                    continue;
                }
                let along = rng.random_range(-range..range) - (min + max) * 0.5;
                let anchor = tf.transform_point(Vec3::new(along, 0., 0.));
                let bottom = panels
                    .iter()
                    .map(|(_, p, _)| p.y)
                    .fold(f32::INFINITY, f32::min);
                let top = panels
                    .iter()
                    .map(|(_, p, s)| p.y + s.y)
                    .fold(f32::NEG_INFINITY, f32::max);
                let low = self.floor_height(anchor.xz()) + 0.30 - bottom;
                let high = self.ceiling_height(anchor.xz()) - 0.25 - top;
                if low >= high {
                    continue;
                }
                // Fit the complete arrangement, then retain independent group
                // heights. Clamping each picture would recreate shared edges.
                let target = self.floor_height(anchor.xz()) + program.eye_height_m;
                let from = (target - 0.30).max(low);
                let to = (target + 0.30).min(high);
                let eye = if from < to {
                    rng.random_range(from..to)
                } else {
                    rng.random_range(low..high)
                };
                let yaw = tf.rotation.to_euler(EulerRot::YXZ).0;
                let frame_style = rng.random_range(0..6);
                let frame_coherence = rng.random_range(0.45..0.95);
                let mut group = Vec::with_capacity(panels.len());
                for (kind, p, size) in panels {
                    let mut o = self.candidate(
                        kind,
                        tf.transform_point(Vec3::new(along + p.x, eye + p.y, 0.)),
                        size,
                        yaw,
                        &mut rng,
                    );
                    o.id += group.len();
                    o.solid = false;
                    if rng.random_bool(frame_coherence) {
                        o.variant = frame_style;
                    }
                    o.position += tf.rotation
                        * Vec3::Z
                        * super::super::objects::wall::mount_offset(&o).unwrap();
                    let (lo, hi) = o.bounds();
                    if lo.y <= self.floor_height(o.position.xz()) + 0.12
                        || !self.wall_attachment_clear(&o)
                        || self
                            .objects
                            .iter()
                            .chain(group.iter())
                            .map(IndoorObject::bounds)
                            .chain(self.columns())
                            .any(|(a, b)| {
                                lo.cmplt(b + Vec3::splat(0.025)).all()
                                    && hi.cmpgt(a - Vec3::splat(0.025)).all()
                            })
                    {
                        group.clear();
                        break;
                    }
                    group.push(o);
                }
                if !group.is_empty() {
                    self.objects.extend(group);
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
