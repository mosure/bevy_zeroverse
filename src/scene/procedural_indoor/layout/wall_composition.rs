//! Coherent installations on real opaque walls. A group is accepted atomically,
//! with the same reveal, opening, ceiling and occupied-space tests as wall props.
use super::super::architecture::facade::FacadeSide;
use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct WallComposition {
    /// Gallery, presentation and mixed media weights, normalized per room.
    pub weights: [f32; 3],
    pub coverage: f32,
    pub picture_aspect: f32,
    pub spacing_m: f32,
    pub eye_height_m: f32,
    pub asymmetry: f32,
}

impl WallComposition {
    pub(crate) fn sample(scene: &IndoorManifest) -> Self {
        let mut rng = stream(scene.seed, 5810);
        let mut weights = [0, 1, 2].map(|_| rng.random_range(0.12..1.0));
        if scene.layout.prefers_soft_seating() || scene.layout == IndoorLayout::Studio {
            weights[0] *= 2.5;
        } else if matches!(
            scene.layout,
            IndoorLayout::Conference | IndoorLayout::Training
        ) {
            weights[1] *= 2.5;
        }
        let sum = weights.iter().sum::<f32>();
        Self {
            weights: weights.map(|w| w / sum),
            coverage: rng.random_range(0.20..0.72),
            picture_aspect: rng.random_range(0.65..1.65),
            spacing_m: rng.random_range(0.075..0.22),
            eye_height_m: rng.random_range(1.40..1.73),
            asymmetry: rng.random_range(0.0..0.55),
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
        let width = (span * self.coverage).clamp(1.0, 4.6).min(span - 0.24);
        if width < 0.65 {
            return Vec::new();
        }
        let gap = self.spacing_m;
        let mut panels = Vec::new();
        if family == 0 {
            let count = rng
                .random_range(1..=5)
                .min(((width + gap) / (0.36 + gap)).floor() as usize)
                .max(1);
            let pane_width = (width - gap * (count - 1) as f32) / count as f32;
            let pane_height = (pane_width / self.picture_aspect).clamp(0.34, 1.28);
            for i in 0..count {
                let x = (i as f32 - (count - 1) as f32 * 0.5) * (pane_width + gap);
                let dy = if count > 1 {
                    self.asymmetry * 0.20 * ((i as f32 / (count - 1) as f32) * 2. - 1.)
                } else {
                    0.
                };
                panels.push((
                    ObjectKind::WallArt,
                    Vec2::new(x, -pane_height * 0.5 + dy),
                    Vec3::new(pane_width, pane_height, rng.random_range(0.024..0.055)),
                ));
            }
        } else {
            let aspect = rng.random_range(1.55..1.90);
            let screen_width = if family == 2 {
                width * 0.60
            } else {
                width.min(2.5)
            };
            let height = (screen_width / aspect).clamp(0.50, 1.40);
            let x = if family == 2 {
                -(width - screen_width) * 0.5
            } else {
                0.
            };
            let kind = if rng.random_bool(0.58) {
                ObjectKind::Whiteboard
            } else {
                ObjectKind::Display
            };
            panels.push((
                kind,
                Vec2::new(x, -height * 0.5),
                Vec3::new(screen_width, height, 0.07),
            ));
            if family == 2 {
                let side_width = width - screen_width - gap;
                if side_width >= 0.36 {
                    let side_height = (height - gap) * 0.5;
                    for side in [-1., 1.] {
                        panels.push((
                            ObjectKind::WallArt,
                            Vec2::new(
                                width * 0.5 - side_width * 0.5,
                                side * (side_height + gap) * 0.5 - side_height * 0.5,
                            ),
                            Vec3::new(side_width, side_height, 0.035),
                        ));
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
        let groups = (rng.random_range(1.0..3.8) * (0.45 + self.density)).round() as usize;
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
                let eye = program.eye_height_m + rng.random_range(-0.08..0.08);
                let yaw = tf.rotation.to_euler(EulerRot::YXZ).0;
                let frame_style = rng.random_range(0..6);
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
                    o.variant = frame_style;
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
mod tests {
    use super::*;
    #[test]
    fn panel_groups_have_positive_sizes_and_separated_frames() {
        let scene =
            IndoorManifest::generate_with_humans(17, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        for seed in 0..256 {
            let mut input = scene.clone();
            input.seed = seed;
            let p = WallComposition::sample(&input);
            assert_eq!(p, WallComposition::sample(&input));
            assert!((p.weights.iter().sum::<f32>() - 1.).abs() < 1e-5);
            let mut rng = stream(seed, 9810);
            for family in 0..3 {
                for span in [0.35, 0.8, 1.5, 3., 8.] {
                    let panels = p.panels(family, span, &mut rng);
                    for (i, (_, pos, size)) in panels.iter().enumerate() {
                        assert!(size.cmpgt(Vec3::ZERO).all());
                        assert!(pos.x.abs() + size.x * 0.5 <= span * 0.5 - 0.10 + 1e-5);
                        for (_, other, s) in &panels[..i] {
                            assert!(
                                (pos.x - other.x).abs() >= (size.x + s.x) * 0.5 + 0.01
                                    || (pos.y + size.y * 0.5 - other.y - s.y * 0.5).abs()
                                        >= (size.y + s.y) * 0.5 + 0.01
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn installed_groups_preserve_mounting_and_archive_validation() {
        let mut rooms_with_art = 0;
        for seed in 0..64 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, 0.25)
                    .unwrap();
            super::super::super::validation::validate_layout(&scene).unwrap();
            let art: Vec<_> = scene
                .objects
                .iter()
                .filter(|o| o.kind == ObjectKind::WallArt)
                .collect();
            rooms_with_art += usize::from(!art.is_empty());
            for object in art {
                assert!(scene.wall_attachment_clear(object));
            }
        }
        assert!(
            rooms_with_art >= 20,
            "accepted wall groups are unreachable: {rooms_with_art}/64"
        );
    }
}
