//! Transactional service runs. Open-door envelopes are admitted before furnishing.
use super::super::objects::kitchen as geometry;
use super::*;
use rand::seq::SliceRandom;
use std::f32::consts::{PI, TAU};
#[cfg(test)]
mod diversity_tests;
pub mod sampling;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KitchenAssembly {
    pub zone: usize,
    pub configuration: String,
    pub units: Vec<usize>,
    pub props: Vec<usize>,
    pub requested: Vec<ObjectKind>,
    pub rejected_optional: usize,
    #[serde(default)]
    pub fitted_to_wall: bool,
    #[serde(default)]
    pub worktop_height_metres: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<sampling::KitchenSampling>,
}
/// Compact cabinet modules retain a usable preparation surface. Never scale
/// appliances to make a failed placement pass.
fn counter_width(kind: ObjectKind, width: f32, scale: f32) -> f32 {
    if kind == ObjectKind::KitchenCounter {
        (width * scale).max(0.48).min(width)
    } else {
        width
    }
}

impl IndoorManifest {
    pub(super) fn furnish_kitchen(
        &mut self,
        zone: &super::super::program::Zone,
        zone_index: usize,
    ) -> bool {
        let mut r = stream(self.seed, 8060 + zone_index as u64);
        let mut placement = stream(self.seed, 0x4b49544348504c41 + zone_index as u64);
        let gap = placement.random_range(0.012..0.045);
        let appliance_coherence = placement.random_range(0.30..0.85);
        let start = self.objects.len();
        let worktop_height = r.random_range(0.86..0.94);
        let mut kinds = vec![ObjectKind::KitchenSink, ObjectKind::KitchenCounter];
        if zone.activity == IndoorLayout::Kitchen {
            kinds.push(ObjectKind::Range);
        }
        kinds.shuffle(&mut placement);
        let mut optional = vec![
            ObjectKind::Refrigerator,
            ObjectKind::KitchenCupboard,
            ObjectKind::Dishwasher,
            ObjectKind::Pantry,
        ];
        if zone.activity == IndoorLayout::Kitchen {
            optional.push(ObjectKind::Oven);
        }
        optional.shuffle(&mut r);
        let mut walls = [0, 1, 2, 3];
        walls.shuffle(&mut r);
        let seeds: Vec<u64> = (0..kinds.len()).map(|_| r.random()).collect();
        let extent = zone.max - zone.min;
        let center = (zone.min + zone.max) * 0.5;
        let mut candidates = Vec::new();
        if let Some(envelope) = &self.envelope {
            for wall in &envelope.walls {
                let tf = envelope.wall_transform(wall.edge);
                let span = envelope.footprint[wall.edge]
                    .distance(envelope.footprint[(wall.edge + 1) % envelope.footprint.len()]);
                candidates.push((
                    tf.translation + tf.rotation * Vec3::Z * placement.random_range(0.035..0.085),
                    tf.rotation.to_euler(EulerRot::YXZ).0,
                    span,
                    true,
                ));
            }
        }
        candidates.shuffle(&mut r);
        // Narrow or glazed edges may not admit a fitted run. Try progressively
        // inset service runs with the same full door/aisle collision contract.
        for inset in [
            placement.random_range(0.30..0.65),
            placement.random_range(0.85..1.55),
            placement.random_range(1.80..2.60),
        ] {
            for &wall in &walls {
                let yaw = [
                    0.,
                    std::f32::consts::PI,
                    std::f32::consts::FRAC_PI_2,
                    -std::f32::consts::FRAC_PI_2,
                ][wall];
                let back = match wall {
                    0 => Vec3::new(center.x, 0., zone.min.y + inset),
                    1 => Vec3::new(center.x, 0., zone.max.y - inset),
                    2 => Vec3::new(zone.min.x + inset, 0., center.y),
                    _ => Vec3::new(zone.max.x - inset, 0., center.y),
                };
                candidates.push((back, yaw, if wall < 2 { extent.x } else { extent.y }, false));
            }
        }
        let mut accepted = None;
        // Discrete wall setbacks can miss a valid service strip between a
        // shaped facade, floor step and doorway. Search continuous positions
        // only after those cheap candidates fail, retaining all support,
        // opening, door-sweep and object collision checks below. A separate
        // stream leaves already-admitted kitchens unchanged.
        let mut recovery = stream(self.seed, 8062 + zone_index as u64);
        // A sampled prep cabinet may be too wide for the room's supported
        // floor strips. Keep the original run first; only a failed run tries
        // compact cabinets. Sink/range widths, doors and clearances stay intact.
        'fit: for counter_scale in [1.0, 0.82, 0.64] {
            let total: f32 = kinds
                .iter()
                .zip(&seeds)
                .map(|(&kind, &seed)| {
                    counter_width(kind, geometry::dimensions(kind, seed).x, counter_scale) + gap
                })
                .sum();
            let interior = (0..192).map(|_| {
                let wall = walls[recovery.random_range(0..walls.len())];
                let yaw = [
                    0.,
                    std::f32::consts::PI,
                    std::f32::consts::FRAC_PI_2,
                    -std::f32::consts::FRAC_PI_2,
                ][wall];
                let back = Vec3::new(
                    recovery.random_range(zone.min.x..zone.max.x),
                    0.,
                    recovery.random_range(zone.min.y..zone.max.y),
                );
                (back, yaw, if wall < 2 { extent.x } else { extent.y }, false)
            });
            for (back, yaw, along, fitted) in candidates.iter().copied().chain(interior) {
                for shift in sampling::run_shifts(along, total, &mut placement) {
                    if total > along - 0.85 {
                        continue;
                    }
                    let q = Quat::from_rotation_y(yaw);
                    self.objects.truncate(start);
                    let mut x = -total * 0.5 + shift;
                    for (&kind, &seed) in kinds.iter().zip(&seeds) {
                        let size = geometry::dimensions(kind, seed);
                        let width = counter_width(kind, size.x, counter_scale);
                        let pos = back + q * Vec3::new(x + width * 0.5, 0., size.z * 0.5);
                        let mut o = self.floor_candidate(kind, pos, size, yaw, &mut r);
                        o.seed = seed;
                        geometry::align_worktop(&mut o, worktop_height);
                        if width < size.x {
                            geometry::resize_width(&mut o, width);
                        }
                        o.position += q * Vec3::Z * ((o.size.z - size.z) * 0.5);
                        if !self.placement_clear(&o, 0.008)
                            || self.objects.get(start).is_some_and(|first| {
                                (first.position.y - o.position.y).abs() > 0.001
                            })
                        {
                            self.rejected_placements += 1;
                            break;
                        }
                        self.objects.push(o);
                        x += width + gap;
                    }
                    if self.objects.len() == start + kinds.len() {
                        accepted =
                            Some((back, q, yaw, x, -total * 0.5 + shift, fitted, shift, total));
                        break 'fit;
                    }
                }
            }
        }
        let Some((back, q, yaw, mut x, mut left, fitted_to_wall, run_offset, total)) = accepted
        else {
            self.objects.truncate(start);
            return false;
        };
        let mut requested = kinds.clone();
        let mut rejected_optional = 0;
        for kind in optional {
            if !r.random_bool(if kind == ObjectKind::Refrigerator {
                0.94
            } else {
                0.75
            }) {
                continue;
            }
            requested.push(kind);
            let seed = r.random();
            let size = geometry::dimensions(kind, seed);
            let mut placed = false;
            let mut sides = [1., -1.];
            sides.shuffle(&mut placement);
            for side in sides {
                let local_x = if side > 0. {
                    x + size.x * 0.5
                } else {
                    left - size.x * 0.5 - gap
                };
                let pos = back + q * Vec3::new(local_x, 0., size.z * 0.5);
                let mut o = self.floor_candidate(kind, pos, size, yaw, &mut r);
                o.seed = seed;
                geometry::align_worktop(&mut o, worktop_height);
                o.position += q * Vec3::Z * ((o.size.z - size.z) * 0.5);
                if self.placement_clear(&o, 0.008)
                    && (self.objects[start].position.y - o.position.y).abs() < 0.001
                {
                    self.objects.push(o);
                    placed = true;
                    if side > 0. {
                        x += size.x + gap;
                    } else {
                        left -= size.x + gap;
                    }
                    break;
                }
            }
            if !placed {
                rejected_optional += 1;
            }
        }
        let mut configuration = "straight".to_owned();
        // Additional preparation surfaces sample an island, return or opposing run.
        let choice = r.random_range(0..3);
        let positions = [
            (
                Vec3::new(
                    center.x + extent.x * placement.random_range(-0.12..0.12),
                    0.,
                    center.y + extent.y * placement.random_range(-0.12..0.12),
                ),
                yaw + std::f32::consts::FRAC_PI_2,
            ),
            (
                back + q * Vec3::new(
                    -total * 0.5 - placement.random_range(0.03..0.18),
                    0.,
                    placement.random_range(2.2..3.4),
                ),
                yaw + std::f32::consts::FRAC_PI_2,
            ),
            (
                back + q * Vec3::new(
                    0.,
                    0.,
                    (if yaw.sin().abs() > 0.5 {
                        extent.x
                    } else {
                        extent.y
                    }) - placement.random_range(0.95..1.60),
                ),
                yaw + std::f32::consts::PI,
            ),
        ];
        let (pos, turn) = positions[choice];
        let turn = turn
            + if choice == 0 {
                placement.random_range(-0.18..0.18)
            } else {
                0.
            };
        let seed = r.random();
        let size = geometry::dimensions(ObjectKind::KitchenCounter, seed);
        let width_scale = if choice != 0 {
            r.random_range(1.20..1.75)
        } else {
            1.
        };
        let mut o = self.floor_candidate(ObjectKind::KitchenCounter, pos, size, turn, &mut r);
        o.seed = seed;
        geometry::resize_width(&mut o, size.x * width_scale);
        if choice == 0 {
            geometry::configure_island(&mut o);
        }
        geometry::align_worktop(&mut o, worktop_height);
        requested.push(ObjectKind::KitchenCounter);
        let floor = self.objects[start].position.y;
        // Try nearby positions on the same working level. An island must not
        // become a low block inside a sunken floor or straddle a stair/step.
        let proposed = sampling::island_shifts(&mut placement);
        let shifts: &[Vec2] = if choice == 0 {
            &proposed
        } else {
            &[Vec2::ZERO]
        };
        let placed = shifts.iter().any(|shift| {
            o.position = pos + q * Vec3::new(shift.x, 0., shift.y);
            o.position.y = self.floor_height(o.position.xz());
            (o.position.y - floor).abs() < 0.001
                && (!geometry::is_island(&o) || self.island_floor_clear(&o, floor))
                && self.placement_clear(&o, 0.90)
        });
        if placed {
            configuration = ["island", "perpendicular_station", "parallel"][choice].to_owned();
            self.objects.push(o);
        } else {
            rejected_optional += 1;
        }
        let units: Vec<_> = (start..self.objects.len()).collect();
        let finish_seed = stream(self.seed, 8070 + zone_index as u64).random();
        for &id in &units {
            let o = &mut self.objects[id];
            let appliance = matches!(
                o.kind,
                ObjectKind::Range
                    | ObjectKind::Oven
                    | ObjectKind::Refrigerator
                    | ObjectKind::Dishwasher
            );
            o.finish_seed = if appliance {
                super::assemblies::finishes::matches(finish_seed, o.seed, appliance_coherence)
                    .then_some(finish_seed ^ 0x4150504c46494e49)
            } else {
                Some(finish_seed)
            };
        }
        let prop_start = self.objects.len();
        for id in units
            .iter()
            .copied()
            .filter(|&id| self.objects[id].kind == ObjectKind::KitchenCounter)
            .collect::<Vec<_>>()
        {
            self.dress_kitchen_counter(id, &mut r);
        }
        // Sink-side mats vary in shape/finish through the established rug program.
        if r.random_bool(0.72) {
            let sink = self.objects[*units
                .iter()
                .find(|&&id| self.objects[id].kind == ObjectKind::KitchenSink)
                .unwrap()]
            .clone();
            let mut mat = self.floor_candidate(
                ObjectKind::Rug,
                sink.position + q * Vec3::Z * (sink.size.z * 0.5 + 0.43),
                Vec3::new(
                    r.random_range(0.75..1.25),
                    0.008,
                    r.random_range(0.48..0.72),
                ),
                yaw,
                &mut r,
            );
            mat.solid = false;
            if super::floor::detail_clear(self, &mat) {
                self.objects.push(mat);
            }
        }
        self.assembly_diagnostics.kitchens.push(KitchenAssembly {
            zone: zone_index,
            configuration,
            units,
            props: (prop_start..self.objects.len()).collect(),
            requested,
            rejected_optional,
            fitted_to_wall,
            worktop_height_metres: worktop_height,
            sampling: Some(sampling::KitchenSampling {
                unit_gap_m: gap,
                run_offset_m: run_offset,
                appliance_coherence,
            }),
        });
        true
    }

    /// The entire walking apron has the same support level as the service run.
    /// Exact floor-patch overlap checks also detect narrow steps between samples.
    pub(crate) fn island_floor_clear(&self, o: &IndoorObject, floor: f32) -> bool {
        let (lo, hi) = o.bounds();
        if (lo.y - floor).abs() > 0.001 {
            return false;
        }
        let apron = Vec3::new(0.90, 0., 0.90);
        self.envelope
            .as_ref()
            .is_none_or(|e| e.support_clear(lo - apron, hi + apron))
    }
    fn dress_kitchen_counter(&mut self, id: usize, r: &mut ChaCha8Rng) {
        let support = self.objects[id].clone();
        // Some worktops are tea/coffee stations; others retain preparation and
        // washing tasks. Islands use the same supported furnishing contract.
        if stream(support.seed, 8066).random_bool(0.28) {
            self.dress_refreshment_counter(&support, None);
            return;
        }
        let mut placement_rng = stream(support.seed, 8065);
        let pattern = super::super::placement::SurfacePattern::sample(&mut placement_rng);
        let mut choices = vec![
            ObjectKind::Microwave,
            ObjectKind::Cooktop,
            ObjectKind::DryingRack,
            ObjectKind::KnifeBlock,
            ObjectKind::UtensilHolder,
            ObjectKind::PaperTowel,
            ObjectKind::KitchenTowel,
            ObjectKind::Dishware,
            ObjectKind::CuttingBoard,
            ObjectKind::FruitBowl,
            ObjectKind::Bread,
            ObjectKind::SoapDispenser,
            ObjectKind::Sponge,
        ];
        choices.shuffle(r);
        let budget = r.random_range(2..=5);
        let mut placed = 0;
        for kind in choices {
            if placed >= budget {
                break;
            }
            let seed = r.random();
            let size = geometry::dimensions(kind, seed);
            if size.x > support.size.x * 0.90 || size.z > support.size.z * 0.82 {
                continue;
            }
            for _ in 0..10 {
                // Fitted equipment faces its user; movable supplies have no
                // privileged room axis. Reserve the rotated footprint first.
                let yaw = if matches!(kind, ObjectKind::Microwave | ObjectKind::Cooktop) {
                    placement_rng.random_range(-0.18..0.18)
                } else {
                    pattern.yaw(&mut placement_rng)
                };
                let Some(proposal) = pattern.propose(
                    geometry::counter_top_bounds(&support),
                    size.xz(),
                    yaw,
                    0.015,
                    &mut placement_rng,
                ) else {
                    continue;
                };
                let pos = support.transform().transform_point(Vec3::new(
                    proposal.center.x,
                    support.size.y,
                    proposal.center.y,
                ));
                let mut o = self.candidate(kind, pos, size, support.yaw + yaw, r);
                o.seed = seed;
                o.solid = false;
                o.support = Some(id);
                if self.prop_clear(&o, &support, 0.012) {
                    self.objects.push(o);
                    placed += 1;
                    if kind == ObjectKind::CuttingBoard && r.random_bool(0.65) {
                        let board = self.objects.last().unwrap().clone();
                        let size = geometry::dimensions(ObjectKind::Bread, r.random()) * 0.70;
                        let yaw = placement_rng.random_range(-0.65..0.65);
                        let Some(proposal) = pattern.propose(
                            geometry::supplies::board_top_bounds(&board),
                            size.xz(),
                            yaw,
                            0.006,
                            &mut placement_rng,
                        ) else {
                            break;
                        };
                        let mut bread = self.candidate(
                            ObjectKind::Bread,
                            board.transform().transform_point(Vec3::new(
                                proposal.center.x,
                                board.size.y,
                                proposal.center.y,
                            )),
                            size,
                            board.yaw + yaw,
                            r,
                        );
                        bread.support = Some(board.id);
                        bread.solid = false;
                        if self.prop_clear(&bread, &board, 0.005) {
                            self.objects.push(bread);
                        }
                    }
                    break;
                }
            }
        }
    }
    pub(super) fn furnish_kitchen_dining(
        &mut self,
        zone: &super::super::program::Zone,
        r: &mut ChaCha8Rng,
    ) {
        if !r.random_bool(0.82) {
            return;
        }
        for _ in 0..48 {
            let position = Vec3::new(
                r.random_range(zone.min.x + 1.2..zone.max.x - 1.2),
                0.,
                r.random_range(zone.min.y + 1.2..zone.max.y - 1.2),
            );
            let size = Vec3::new(
                r.random_range(1.1..1.9),
                r.random_range(0.70..0.80),
                r.random_range(0.75..1.15),
            );
            let yaw = r.random_range(-PI..PI);
            let checkpoint = self.objects.len();
            if let Some(id) = self.add(ObjectKind::Table, position, size, yaw, r) {
                let table = self.objects[id].clone();
                let tf = table.transform();
                for (local, angle) in [
                    (Vec3::new(0., 0., size.z * 0.5 + 0.55), 0.),
                    (
                        Vec3::new(0., 0., -size.z * 0.5 - 0.55),
                        std::f32::consts::PI,
                    ),
                    (
                        Vec3::new(size.x * 0.5 + 0.55, 0., 0.),
                        std::f32::consts::FRAC_PI_2,
                    ),
                    (
                        Vec3::new(-size.x * 0.5 - 0.55, 0., 0.),
                        -std::f32::consts::FRAC_PI_2,
                    ),
                ] {
                    if r.random_bool(0.15) {
                        continue;
                    }
                    let local = local
                        + Vec3::new(r.random_range(-0.16..0.16), 0., r.random_range(-0.16..0.16));
                    let angle = angle + r.random_range(-0.30..0.30);
                    let before = self.objects.len();
                    self.chair(tf.transform_point(local), yaw + angle, r);
                    if self.objects.len() > before {
                        self.objects[before].interaction_target = Some(id);
                    }
                }
                if self.objects.len() >= checkpoint + 3 {
                    return;
                }
                self.objects.truncate(checkpoint);
            }
        }
    }
    pub(super) fn dress_kitchen_dining(&mut self, id: usize, r: &mut ChaCha8Rng) {
        let support = self.objects[id].clone();
        if !self.clutter.active(support.seed) {
            return;
        }
        let mut placement_rng = stream(support.seed, 8066);
        let pattern = super::super::placement::SurfacePattern::sample(&mut placement_rng);
        for kind in [
            ObjectKind::Dishware,
            ObjectKind::Dishware,
            ObjectKind::FruitBowl,
            ObjectKind::KitchenTowel,
        ] {
            let seed = r.random();
            let size = geometry::dimensions(kind, seed);
            for _ in 0..12 {
                let yaw = pattern.yaw(&mut placement_rng);
                let Some(proposal) = pattern.propose(
                    (-support.size.xz() * 0.5, support.size.xz() * 0.5),
                    size.xz(),
                    yaw,
                    0.015,
                    &mut placement_rng,
                ) else {
                    continue;
                };
                let mut o = self.candidate(
                    kind,
                    support.transform().transform_point(Vec3::new(
                        proposal.center.x,
                        support.size.y,
                        proposal.center.y,
                    )),
                    size,
                    support.yaw + yaw,
                    r,
                );
                o.seed = seed;
                o.solid = false;
                o.support = Some(id);
                if self.prop_clear(&o, &support, 0.01) {
                    self.objects.push(o);
                    break;
                }
            }
        }
    }
    pub(crate) fn validate_kitchens(&self) -> Result<(), String> {
        for group in &self.assembly_diagnostics.kitchens {
            if group.sampling.as_ref().is_some_and(|s| !s.valid()) {
                return Err("invalid kitchen placement sampling".into());
            }
            if !(0.86..=0.94).contains(&group.worktop_height_metres) {
                return Err("invalid kitchen worktop height".into());
            }
            for kind in [ObjectKind::KitchenSink, ObjectKind::KitchenCounter] {
                if !group
                    .units
                    .iter()
                    .any(|&id| self.objects.get(id).is_some_and(|o| o.kind == kind))
                {
                    return Err(format!(
                        "seed {}: kitchen missing required {kind:?}",
                        self.seed
                    ));
                }
            }
            if group.units.iter().any(|&id| {
                self.objects
                    .get(id)
                    .is_none_or(|o| !o.kind.is_kitchen() || !o.solid)
            }) {
                return Err("invalid kitchen assembly membership".into());
            }
            let floor = self.objects[group.units[0]].position.y;
            let islands: Vec<_> = group
                .units
                .iter()
                .map(|&id| &self.objects[id])
                .filter(|o| geometry::is_island(o))
                .collect();
            if islands.len() != usize::from(group.configuration == "island") {
                return Err("kitchen island configuration lacks its freestanding counter".into());
            }
            for island in islands {
                if (island.size.y - group.worktop_height_metres).abs() > 0.001
                    || !self.island_floor_clear(island, floor)
                    || !self.placement_clear(island, 0.90)
                {
                    return Err(
                        "kitchen island lacks worktop height or level circulation clearance".into(),
                    );
                }
            }
            if self.layout == IndoorLayout::Kitchen
                && !group
                    .units
                    .iter()
                    .any(|&id| self.objects[id].kind == ObjectKind::Range)
            {
                return Err("kitchen lacks a cooking station".into());
            }
            for &id in &group.props {
                let Some(o) = self.objects.get(id) else {
                    return Err("invalid kitchen prop membership".into());
                };
                if let Some(support) = o.support {
                    if !group.units.contains(&support) && !group.props.contains(&support) {
                        return Err("kitchen prop has a support outside its assembly".into());
                    }
                } else if o.kind != ObjectKind::Rug {
                    return Err("kitchen prop lacks support".into());
                }
            }
        }
        Ok(())
    }

    pub(crate) fn missing_kitchen_program(&self) -> bool {
        self.layout.is_kitchen() && self.assembly_diagnostics.kitchens.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_seed_1942_admits_a_complete_kitchen() {
        use crate::scene::procedural_indoor::validation::{validate_geometry, validate_layout};
        for density in [0., 0.65, 1.] {
            for people in [0., 0.25, 1.] {
                let scene = IndoorManifest::generate_with_humans(
                    1942,
                    IndoorLayout::Mixed,
                    density,
                    3,
                    people,
                )
                .unwrap();
                validate_layout(&scene).unwrap();
                validate_geometry(&scene).unwrap();
                assert_eq!(scene.layout, IndoorLayout::Kitchen);
                assert!(!scene.assembly_diagnostics.kitchens.is_empty());
                assert!(!scene.envelope.as_ref().unwrap().floor_patches.is_empty());
                assert_eq!(
                    scene,
                    IndoorManifest::generate_with_humans(
                        1942,
                        IndoorLayout::Mixed,
                        density,
                        3,
                        people
                    )
                    .unwrap(),
                    "recovery must replay without changing the requested seed"
                );
            }
        }
    }
}
