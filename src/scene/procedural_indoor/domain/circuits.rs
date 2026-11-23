//! Ceiling lighting controls. Wiring follows the installed floor plan. Switch
//! state, daylight response and lamp batch are sampled independently of cameras.
use super::*;
use crate::scene::procedural_indoor::{architecture, photometry, program::Zone};
use std::collections::BTreeMap;
mod tasks;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CeilingCircuit {
    /// Fraction of installed output. Zero switches off the entire circuit.
    pub dimmer: f32,
    pub batch_kelvin_offset: f32,
    pub batch_tint: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CeilingCircuits {
    /// Index into `circuits` for each semantic ceiling-light instance in mount order.
    pub mount_circuits: Vec<usize>,
    pub circuits: Vec<CeilingCircuit>,
}

/// A transmitted exterior opening projected onto the floor plan. The control
/// signal weights admitted daylight by distance from the glazed span.
pub(super) struct DaylightOpening {
    pub a: Vec2,
    pub b: Vec2,
    pub reach_m: f32,
    pub admitted_lux: f32,
}

fn zone_at(zones: &[Zone], point: Vec2) -> Option<usize> {
    zones
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let distance = |z: &Zone| point.distance_squared(point.clamp(z.min, z.max));
            distance(a).total_cmp(&distance(b))
        })
        .map(|(index, _)| index)
}

impl DaylightOpening {
    fn signal(&self, p: Vec2, zone: usize, zones: &[Zone]) -> f32 {
        let d = self.b - self.a;
        let t = ((p - self.a).dot(d) / d.length_squared().max(1e-8)).clamp(0., 1.);
        let nearest = self.a + d * t;
        if zone_at(zones, nearest.lerp(p, 0.001)) != Some(zone) {
            return 0.;
        }
        let reach_sq = self.reach_m * self.reach_m;
        self.admitted_lux * reach_sq / (reach_sq + p.distance_squared(nearest))
    }
}

struct Bank {
    zone: usize,
    mounts: usize,
    center: Vec2,
    priority: f32,
    task_demand: f32,
    on: bool,
}

fn wire_mounts(
    scene: &IndoorManifest,
    zones: &[Zone],
    mounts: &[Vec3],
    primary: usize,
) -> (Vec<usize>, Vec<Bank>) {
    let mut wiring = stream(scene.seed, 0x5749_5249_4e47);
    let grids: Vec<_> = zones
        .iter()
        .map(|z| {
            let extent = z.max - z.min;
            let mut pitch = Vec2::new(
                wiring.random_range(3.6..7.2),
                wiring.random_range(7.0..14.0),
            );
            if (extent.x > extent.y) == wiring.random_bool(0.72) {
                pitch = pitch.yx();
            }
            (extent / pitch).round().max(Vec2::ONE)
        })
        .collect();
    let mut keys = BTreeMap::new();
    let mut banks = Vec::<Bank>::new();
    let mut mount_circuits = Vec::with_capacity(mounts.len());
    for (i, &p) in mounts.iter().enumerate() {
        let zone = (i < primary).then(|| zone_at(zones, p.xz())).flatten();
        let cell = zone.map_or(UVec2::ZERO, |z| {
            let bounds = &zones[z];
            let uv = (p.xz() - bounds.min) / (bounds.max - bounds.min);
            (uv * grids[z])
                .floor()
                .clamp(Vec2::ZERO, grids[z] - Vec2::ONE)
                .as_uvec2()
        });
        let under_deck = scene
            .envelope
            .as_ref()
            .and_then(|e| e.mezzanine.as_ref())
            .is_some_and(|m| m.deck.contains(p.xz()) && p.y < m.deck.height);
        let zone = zone.unwrap_or(zones.len()); // Adjoining circulation has its own switch.
        let bank = *keys
            .entry((zone, cell.x, cell.y, under_deck))
            .or_insert_with(|| {
                banks.push(Bank {
                    zone,
                    mounts: 0,
                    center: Vec2::ZERO,
                    priority: 0.,
                    task_demand: 0.,
                    on: false,
                });
                banks.len() - 1
            });
        banks[bank].mounts += 1;
        banks[bank].center += p.xz();
        mount_circuits.push(bank);
    }
    (mount_circuits, banks)
}

impl CeilingCircuits {
    pub(super) fn sample(
        scene: &IndoorManifest,
        mounts: &[Vec3],
        windows: &[DaylightOpening],
        borrowed_lux: f32,
    ) -> Self {
        let program = scene
            .program
            .as_ref()
            .expect("circuits require an indoor program");
        let zones = &program.zones;
        let controls = &program.domain.as_ref().unwrap().photometry;
        let primary = architecture::primary_fixture_count(scene, mounts);
        let (mount_circuits, mut banks) = wire_mounts(scene, zones, mounts, primary);
        let task_weight =
            if (0.25..1.).contains(&controls.active_fraction) && scene.target_lux > 40. {
                tasks::assign(
                    scene,
                    &mounts[..primary],
                    &mount_circuits[..primary],
                    &mut banks,
                    windows,
                    borrowed_lux,
                )
            } else {
                0.
            };

        let mut rng = stream(photometry::lighting_seed(scene), 0x0043_4952_4355_4954);
        // Manual switching and daylight-responsive drivers both occur.
        let harvesting = if rng.random_bool(0.62) {
            rng.random_range(0.38..0.86)
        } else {
            0.
        };
        let mut circuits: Vec<_> = banks
            .iter_mut()
            .map(|b| {
                let response = if harvesting > 0. && b.zone < zones.len() {
                    let center = b.center / b.mounts as f32;
                    let available: f32 = windows
                        .iter()
                        .map(|w| w.signal(center, b.zone, zones))
                        .sum();
                    harvesting * available / (scene.target_lux.max(0.1) + available)
                } else {
                    0.
                };
                b.priority =
                    rng.random::<f32>() - response + 0.7 * b.task_demand / task_weight.max(0.01);
                let manual = 1. - controls.circuit_contrast * rng.random_range(0.04..0.50);
                CeilingCircuit {
                    dimmer: manual * (1. - response),
                    batch_kelvin_offset: rng.random_range(-280.0..280.0),
                    batch_tint: rng.random_range(-0.018..0.018),
                }
            })
            .collect();

        // Select whole banks against a budget in *fixtures*. One occupied bank
        // stays on even in a small suite. The adjoining bay has independent use.
        let mut order: Vec<_> = (0..banks.len())
            .filter(|&i| banks[i].zone < zones.len())
            .collect();
        order.sort_by(|&a, &b| banks[b].priority.total_cmp(&banks[a].priority));
        if let Some(anchor) = order
            .iter()
            .position(|&i| Some(banks[i].zone) == program.primary_zone)
        {
            order[..=anchor].rotate_right(1);
        }
        let target = (controls.active_fraction * primary as f32).round() as usize;
        let mut on: usize = 0;
        for i in order {
            let next = on + banks[i].mounts;
            if controls.active_fraction > 0.
                && (on == 0 || next.abs_diff(target) < on.abs_diff(target))
            {
                on = next;
                banks[i].on = true;
            }
        }
        // In occupied electric-light conditions a single additional switch can
        // cover a bank of unlit worktops. Sparse and daylit spaces keep their budget.
        let unserved: f32 = banks.iter().filter(|b| !b.on).map(|b| b.task_demand).sum();
        let tolerated = 0.04 + 0.52 * (1. - controls.active_fraction);
        if unserved > tolerated * task_weight {
            if let Some(b) = banks
                .iter_mut()
                .filter(|b| !b.on && b.task_demand > 0.)
                .max_by(|a, b| a.task_demand.total_cmp(&b.task_demand))
            {
                b.on = true;
            }
        }
        for (b, circuit) in banks.iter().zip(&mut circuits) {
            let on = if b.zone == zones.len() {
                rng.random_bool(controls.active_fraction as f64)
            } else {
                b.on
            };
            if !on {
                circuit.dimmer = 0.;
            }
        }
        Self {
            mount_circuits,
            circuits,
        }
    }

    pub fn for_mount(&self, index: usize) -> Option<&CeilingCircuit> {
        self.circuits.get(*self.mount_circuits.get(index)?)
    }

    pub(super) fn valid(&self) -> bool {
        self.mount_circuits.iter().all(|&i| i < self.circuits.len())
            && self.circuits.len() <= self.mount_circuits.len()
            && self.circuits.iter().all(|b| {
                (0.0..=1.0).contains(&b.dimmer)
                    && (-280.0..=280.0).contains(&b.batch_kelvin_offset)
                    && (-0.018..=0.018).contains(&b.batch_tint)
            })
    }
}

#[cfg(test)]
mod tests;
