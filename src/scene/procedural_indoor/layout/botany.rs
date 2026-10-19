//! Seeded greenery programs fitted to real windows, supports and accepted furniture.
//! Placement never relaxes portal, ceiling, support or circulation constraints.
use super::*;
use rand::seq::SliceRandom;

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct BotanyProgram {
    pub abundance: f32,
    pub window_affinity: f32,
    pub clustering: f32,
    pub tabletop_fraction: f32,
    pub species_weights: [f32; super::super::plants::SPECIES as usize],
}
impl BotanyProgram {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 6260);
        let mut weights = std::array::from_fn(|_| rng.random_range(0.04_f32..1.0).powf(1.4));
        let sum: f32 = weights.iter().sum();
        weights.iter_mut().for_each(|w| *w /= sum);
        Self {
            abundance: rng.random_range(0.0_f32..1.0).powf(1.25),
            window_affinity: rng.random_range(0.18..0.95),
            clustering: rng.random_range(0.0..0.86),
            tabletop_fraction: rng.random_range(0.05..0.80),
            species_weights: weights,
        }
    }

    fn species(&self, rng: &mut ChaCha8Rng) -> u32 {
        let mut value = rng.random::<f32>();
        for (i, weight) in self.species_weights.iter().enumerate() {
            value -= weight;
            if value <= 0. {
                return i as u32;
            }
        }
        super::super::plants::SPECIES - 1
    }
}

fn size(variant: u32, tabletop: bool, rng: &mut ChaCha8Rng) -> Vec3 {
    let (width, height) = if tabletop {
        (
            rng.random_range(0.14..0.33),
            match variant {
                7 => rng.random_range(0.16..0.26),
                8 | 9 => rng.random_range(0.23..0.41),
                4 | 13 | 16 => rng.random_range(0.22..0.40),
                _ => rng.random_range(0.25..0.55),
            },
        )
    } else {
        match variant {
            7 => (rng.random_range(0.30..0.54), rng.random_range(0.29..0.52)),
            8 | 9 => (rng.random_range(0.39..0.65), rng.random_range(0.55..1.28)),
            4 | 13 | 16 => (rng.random_range(0.51..0.87), rng.random_range(0.63..1.14)),
            15 => (rng.random_range(0.60..0.95), rng.random_range(0.90..1.80)),
            _ => (rng.random_range(0.47..0.95), rng.random_range(1.00..2.12)),
        }
    };
    Vec3::new(width, height, width * rng.random_range(0.86..1.12))
}

/// Preserve useful aisles already present after furniture placement. The floor
/// abundance program is decorative: it must not consume the room's longest
/// human-clear travel spaces. This bounded coarse probe runs once, rather than
/// doing navigation searches for each rejected planting proposal.
struct AisleReserve {
    segments: Vec<(Vec3, Vec3)>,
}

impl AisleReserve {
    const RADIUS: f32 = 0.42;
    const HEIGHT: f32 = 1.85;

    fn sample(scene: &IndoorManifest) -> Self {
        let mut obstacles = scene.camera_obstacles();
        if let Some(envelope) = &scene.envelope {
            obstacles.extend(envelope.motion_barriers(scene.room_size));
        }
        obstacles.extend(
            scene
                .objects
                .iter()
                .filter(|o| !o.neighbor)
                .map(IndoorObject::bounds),
        );
        let pad = Vec3::new(Self::RADIUS, 0., Self::RADIUS);
        let clear = |a: Vec3, b: Vec3| {
            scene.envelope.as_ref().is_none_or(|e| {
                e.segment_clear(
                    scene.room_size,
                    a + Vec3::Y * (Self::HEIGHT * 0.5),
                    b + Vec3::Y * (Self::HEIGHT * 0.5),
                    Self::RADIUS,
                )
            }) && obstacles
                .iter()
                .filter(|(lo, hi)| hi.y > 0.04 && lo.y < Self::HEIGHT)
                .all(|(lo, hi)| {
                    !segment_hits_box(
                        a,
                        b,
                        (*lo - pad).with_y(-1.),
                        (*hi + pad).with_y(Self::HEIGHT + 1.),
                    )
                })
        };
        let zones: Vec<_> = scene.program.as_ref().map_or_else(
            || {
                let half = scene.room_size.xz() * 0.5;
                vec![(-half, half)]
            },
            |p| p.zones.iter().map(|z| (z.min, z.max)).collect(),
        );
        let mut segments = Vec::new();
        for (min, max) in zones {
            let inset = Vec2::splat(0.70);
            if (max - min).cmple(inset * 2.).any() {
                continue;
            }
            let mut points = Vec::with_capacity(16);
            for x in 0..4 {
                for z in 0..4 {
                    let p = (min + inset)
                        + (max - min - inset * 2.) * Vec2::new(x as f32 / 3., z as f32 / 3.);
                    let p = Vec3::new(p.x, 0., p.y);
                    if scene.floor_height(p.xz()).abs() < 0.01
                        && scene.ceiling_height(p.xz()) > Self::HEIGHT + 0.10
                        && clear(p, p)
                    {
                        points.push(p);
                    }
                }
            }
            let mut available = Vec::new();
            for (i, &a) in points.iter().enumerate() {
                for &b in points.iter().skip(i + 1) {
                    if a.distance_squared(b) >= 1.6_f32.powi(2) && clear(a, b) {
                        available.push((a, b));
                    }
                }
            }
            available.sort_by(|(a, b), (c, d)| {
                c.distance_squared(*d).total_cmp(&a.distance_squared(*b))
            });
            let mut selected: Vec<(Vec3, Vec3)> = Vec::with_capacity(3);
            for (a, b) in available {
                // Similar parallel chords through the same pocket do not count
                // as additional aisles. Keep distinct positions or directions.
                if selected.iter().any(|(c, d)| {
                    ((a + b) * 0.5).distance_squared((*c + *d) * 0.5) < 0.9_f32.powi(2)
                        && (b - a).normalize().dot((*d - *c).normalize()).abs() > 0.80
                }) {
                    continue;
                }
                selected.push((a, b));
                if selected.len() == 3 {
                    break;
                }
            }
            segments.extend(selected);
        }
        Self { segments }
    }

    fn clear(&self, object: &IndoorObject) -> bool {
        let (lo, hi) = object.bounds();
        let pad = Vec3::new(Self::RADIUS, 0., Self::RADIUS);
        // Even a short pot obstructs foot travel; do not test only torso height.
        self.segments.iter().all(|(a, b)| {
            !segment_hits_box(
                *a,
                *b,
                (lo - pad).with_y(-1.),
                (hi + pad).with_y(Self::HEIGHT + 1.),
            )
        })
    }
}

impl IndoorManifest {
    pub(super) fn furnish_botany(&mut self) {
        let program = BotanyProgram::sample(self.seed);
        let mut rng = stream(self.seed, 6261);
        let aisles = AisleReserve::sample(self);
        let anchors: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor
                    && matches!(
                        o.kind,
                        ObjectKind::Desk
                            | ObjectKind::Sofa
                            | ObjectKind::Cabinet
                            | ObjectKind::Bench
                            | ObjectKind::RoomDivider
                    )
            })
            .cloned()
            .collect();
        let mut windows = Vec::new();
        if let Some(envelope) = &self.envelope {
            for wall in &envelope.walls {
                if let Some(facade) = &wall.facade {
                    let tf = envelope.wall_transform(wall.edge);
                    for opening in &facade.openings {
                        windows.push((tf, (opening.min.x + opening.max.x) * 0.5));
                    }
                }
            }
        } else if let Some(exterior) = &self.exterior {
            for facade in &exterior.facades {
                for opening in &facade.openings {
                    windows.push((
                        facade.side.transform(self.room_size),
                        (opening.min.x + opening.max.x) * 0.5,
                    ));
                }
            }
        }
        let budget = (self.room_size.x * self.room_size.z).sqrt()
            * program.abundance
            * (0.45 + self.density * 0.60);
        let budget = budget.round().clamp(0., 12.) as usize;
        let mut cluster: Option<Vec3> = None;
        for _ in 0..budget {
            let variant = program.species(&mut rng);
            let size = size(variant, false, &mut rng);
            for _ in 0..16 {
                let position = if let Some(centre) =
                    cluster.filter(|_| rng.random_bool(program.clustering as f64))
                {
                    let angle = rng.random_range(0.0..std::f32::consts::TAU);
                    centre
                        + Vec3::new(angle.cos(), 0., angle.sin())
                            * rng.random_range(size.x * 0.90..size.x * 1.85)
                } else if !windows.is_empty() && rng.random_bool(program.window_affinity as f64) {
                    let (tf, centre) = windows[rng.random_range(0..windows.len())];
                    tf.transform_point(Vec3::new(
                        centre + rng.random_range(-0.6..0.6),
                        0.,
                        size.x.max(size.z) * 0.5 + rng.random_range(0.30..0.46),
                    ))
                } else if !anchors.is_empty() && rng.random_bool(0.74) {
                    let anchor = &anchors[rng.random_range(0..anchors.len())];
                    let side = if rng.random_bool(0.5) { -1. } else { 1. };
                    anchor.position
                        + Quat::from_rotation_y(anchor.yaw)
                            * Vec3::new(
                                side * ((anchor.size.x + size.x) * 0.5
                                    + rng.random_range(0.16..0.52)),
                                0.,
                                rng.random_range(-0.65..0.65) * anchor.size.z,
                            )
                } else {
                    Vec3::new(
                        rng.random_range(-0.42..0.42) * self.room_size.x,
                        0.,
                        rng.random_range(-0.42..0.42) * self.room_size.z,
                    )
                }
                .with_y(0.);
                let yaw = rng.random_range(0.0..std::f32::consts::TAU);
                let mut plant =
                    self.floor_candidate(ObjectKind::Plant, position, size, yaw, &mut rng);
                plant.variant = variant;
                if !aisles.clear(&plant) || !self.placement_clear(&plant, 0.06) {
                    self.rejected_placements += 1;
                    continue;
                }
                self.objects.push(plant);
                cluster = Some(position);
                break;
            }
        }
        let mut supports: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor
                    && matches!(
                        o.kind,
                        ObjectKind::Desk
                            | ObjectKind::Table
                            | ObjectKind::SideTable
                            | ObjectKind::Cabinet
                            | ObjectKind::DrawerUnit
                    )
                    && o.size.y <= 1.5
            })
            .cloned()
            .collect();
        supports.shuffle(&mut rng);
        for support in supports {
            if !rng.random_bool((program.tabletop_fraction * program.abundance) as f64) {
                continue;
            }
            let variant = [2, 4, 7, 8, 9, 11, 13][rng.random_range(0..7)];
            let size = size(variant, true, &mut rng);
            for _ in 0..8 {
                let side = if rng.random_bool(0.5) { -1. } else { 1. };
                let offset = Vec3::new(
                    side * support.size.x * rng.random_range(0.25..0.37),
                    0.,
                    -support.size.z * rng.random_range(0.18..0.30),
                );
                let previous = self.objects.len();
                self.prop(
                    &support,
                    ObjectKind::Plant,
                    offset,
                    size,
                    rng.random_range(0.0..std::f32::consts::TAU),
                    &mut rng,
                );
                if self.objects.len() > previous {
                    self.objects[previous].variant = variant;
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
    fn aisle_reserve_keeps_short_pots_and_tall_canopies_out_of_travel_space() {
        let aisles = AisleReserve {
            segments: vec![(Vec3::new(-2., 0., 0.), Vec3::new(2., 0., 0.))],
        };
        for height in [0.25, 1.8] {
            let mut plant = IndoorObject {
                finish_seed: None,
                id: 0,
                kind: ObjectKind::Plant,
                position: Vec3::ZERO,
                size: Vec3::new(0.6, height, 0.6),
                yaw: 0.3,
                variant: 7,
                seed: 4,
                solid: true,
                support: None,
                neighbor: false,
                interaction_target: None,
                cable_path: None,
                cable_target: None,
            };
            assert!(!aisles.clear(&plant));
            plant.position.z = 0.65;
            assert!(!aisles.clear(&plant), "pot consumed body clearance");
            plant.position.z = 1.1;
            assert!(aisles.clear(&plant), "peripheral planting was rejected");
        }
    }

    #[test]
    fn new_botany_preserves_existing_available_aisles_with_bounded_sampling() {
        let mut reserved = 0;
        for seed in 0..16 {
            let mut scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
                    .unwrap();
            let aisles = AisleReserve::sample(&scene);
            let zones = scene.program.as_ref().map_or(1, |p| p.zones.len());
            assert!(aisles.segments.len() <= zones * 3);
            reserved += aisles.segments.len();
            let previous = scene.objects.len();
            scene.furnish_botany();
            for plant in scene
                .objects
                .iter()
                .skip(previous)
                .filter(|o| o.support.is_none())
            {
                assert!(
                    aisles.clear(plant),
                    "seed {seed}: new plant blocked an existing aisle"
                );
                assert!(scene.placement_clear(plant, 0.06));
            }
        }
        assert!(
            reserved > 0,
            "aisle protection never discovered available travel space"
        );
    }

    #[test]
    fn botany_program_replays_and_varies_real_factors() {
        let mut low = 0;
        let mut lush = 0;
        let mut species = std::collections::BTreeSet::new();
        for seed in 0..256 {
            let p = BotanyProgram::sample(seed);
            assert_eq!(
                serde_json::to_string(&p).unwrap(),
                serde_json::to_string(&BotanyProgram::sample(seed)).unwrap()
            );
            assert!((p.species_weights.iter().sum::<f32>() - 1.).abs() < 1e-5);
            low += usize::from(p.abundance < 0.12);
            lush += usize::from(p.abundance > 0.8);
            let mut rng = stream(seed, 6262);
            for _ in 0..12 {
                species.insert(p.species(&mut rng));
            }
        }
        assert!(low >= 20 && lush >= 20);
        assert_eq!(species.len(), super::super::super::plants::SPECIES as usize);
    }

    #[test]
    fn botanical_floor_and_tabletop_placements_keep_support_and_circulation_contracts() {
        let mut floor_counts = std::collections::BTreeSet::new();
        let mut seen = std::collections::BTreeSet::new();
        let mut tabletop = 0;
        for seed in 0..64 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, 0.)
                    .unwrap_or_else(|e| panic!("botanical placement seed {seed}: {e}"));
            let plants: Vec<_> = scene
                .objects
                .iter()
                .filter(|o| !o.neighbor && o.kind == ObjectKind::Plant)
                .collect();
            floor_counts.insert(plants.iter().filter(|o| o.support.is_none()).count());
            for plant in plants {
                seen.insert(plant.variant);
                if let Some(id) = plant.support {
                    tabletop += 1;
                    let support = scene.objects.iter().find(|o| o.id == id).unwrap();
                    assert!(
                        scene.prop_clear(plant, support, 0.012),
                        "seed {seed}: plant {} violated support containment",
                        plant.id
                    );
                } else {
                    assert!(
                        plant.solid && scene.placement_clear(plant, 0.06),
                        "seed {seed}: plant {} violated circulation",
                        plant.id
                    );
                }
            }
        }
        assert!(
            floor_counts.len() >= 4,
            "greenery count failed to diversify: {floor_counts:?}"
        );
        assert_eq!(seen.len(), super::super::super::plants::SPECIES as usize);
        assert!(
            tabletop >= 25,
            "compact plants were not reaching valid surfaces: {tabletop}"
        );
    }
}
