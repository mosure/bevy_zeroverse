//! Accepted functional groups and bounded, transactional meeting construction.
//! Receipts describe actual support/seat relationships; layout names are priors.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssemblyRole {
    Workstation,
    Meeting,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionalAssembly {
    pub role: AssemblyRole,
    pub surface: usize,
    pub seats: Vec<usize>,
    pub requested_capacity: usize,
    pub attempts: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AssemblyDiagnostics {
    #[serde(default)]
    pub mounts: Vec<super::functional::MountedUnit>,
    #[serde(default)]
    pub programs: Vec<super::functional::ProgramAssembly>,
    #[serde(default)]
    pub kitchens: Vec<super::kitchen::KitchenAssembly>,
    pub proposed_workstations: usize,
    pub proposed_meetings: usize,
    pub rejected_surfaces: usize,
    pub rejected_seating: usize,
    pub rejected_meetings: usize,
    pub accepted: Vec<FunctionalAssembly>,
}

impl IndoorManifest {
    pub(super) fn meeting_group(
        &mut self,
        zone: &super::super::program::Zone,
        zone_index: usize,
        usable: Vec2,
        center: Vec2,
    ) {
        // Group retries never consume the next zone's activity/content stream.
        let mut rng = stream(self.seed, 5840 + zone_index as u64);
        self.assembly_diagnostics.proposed_meetings += 1;
        let scale = zone.composition.as_ref().map_or(1., |mix| mix.group_scale);
        let width =
            (rng.random_range(1.05..1.55_f32) * scale).clamp(0.85, (usable.x - 1.8).max(0.85));
        let length = if rng.random_bool(0.28) {
            width
        } else {
            (usable.y - 1.9).min(rng.random_range(2.0..5.8)).max(1.1)
        };
        let rows = ((length - 0.35) / zone.seat_pitch).floor().max(1.) as usize;
        let capacity = ((2 * rows + 2) as f32 * zone.occupancy).ceil().max(2.) as usize;
        let minimum = (capacity as f32 * 0.65).ceil().max(2.) as usize;
        let checkpoint = self.objects.len();
        let family = rng.random_range(0..super::super::objects::chairs::FAMILIES);
        for attempt in 0..12 {
            let yaw = zone.orientation + rng.random_range(-0.22..0.22);
            let tf = Transform::from_xyz(
                center.x + rng.random_range(-0.30..0.30),
                0.,
                center.y + rng.random_range(-0.30..0.30),
            )
            .with_rotation(Quat::from_rotation_y(yaw));
            let factor = (1. - attempt as f32 * 0.022).max(0.80);
            let size = Vec3::new(
                width * factor,
                rng.random_range(0.72..0.78),
                length * factor,
            );
            let Some(surface) = self.add(ObjectKind::Table, tf.translation, size, yaw, &mut rng)
            else {
                self.assembly_diagnostics.rejected_surfaces += 1;
                continue;
            };
            // Opposing seats are interleaved so a constrained group cannot
            // accept only a row on one edge while reporting full capacity.
            for row in 0..rows {
                let z = (row as f32 - (rows - 1) as f32 * 0.5) * (size.z - 0.36) / rows as f32;
                for side in [-1., 1.] {
                    if self.objects.len() - checkpoint > capacity {
                        break;
                    }
                    self.chair_in_family(
                        tf.transform_point(Vec3::new(side * (size.x * 0.5 + 0.57), 0., z)),
                        yaw + side * std::f32::consts::FRAC_PI_2,
                        Some(family),
                        &mut rng,
                    );
                }
            }
            for side in [-1., 1.] {
                if self.objects.len() - checkpoint > capacity {
                    break;
                }
                self.chair_in_family(
                    tf.transform_point(Vec3::new(0., 0., side * (size.z * 0.5 + 0.57))),
                    yaw + if side > 0. { 0. } else { std::f32::consts::PI },
                    Some(family),
                    &mut rng,
                );
            }
            let seats: Vec<_> = (checkpoint + 1..self.objects.len()).collect();
            if seats.len() >= minimum {
                for &id in &seats {
                    self.objects[id].interaction_target = Some(surface);
                }
                self.assembly_diagnostics.accepted.push(FunctionalAssembly {
                    role: AssemblyRole::Meeting,
                    surface,
                    seats,
                    requested_capacity: capacity,
                    attempts: attempt + 1,
                });
                return;
            }
            self.assembly_diagnostics.rejected_seating += 1;
            // Only this proposal owns these IDs. No prop/person/camera exists yet.
            self.objects.truncate(checkpoint);
        }
        self.assembly_diagnostics.rejected_meetings += 1;
    }

    pub(crate) fn validate_assemblies(&self) -> Result<(), String> {
        for group in &self.assembly_diagnostics.accepted {
            let surface = self
                .objects
                .get(group.surface)
                .ok_or("assembly references an absent surface")?;
            if !surface.kind.is_surface()
                || surface.neighbor
                || group.attempts == 0
                || group.seats.is_empty()
                || (group.role == AssemblyRole::Meeting && group.seats.len() < 2)
            {
                return Err("incomplete functional assembly".into());
            }
            let mut ids = std::collections::BTreeSet::new();
            for &id in &group.seats {
                let seat = self
                    .objects
                    .get(id)
                    .ok_or("assembly references an absent seat")?;
                if !ids.insert(id)
                    || seat.kind != ObjectKind::Chair
                    || seat.neighbor
                    || seat.interaction_target != Some(group.surface)
                {
                    return Err("inconsistent assembly seat relationship".into());
                }
            }
        }
        Ok(())
    }

    pub(super) fn assign_assembly_finishes(&mut self) {
        for group in &self.assembly_diagnostics.accepted {
            let seed = self.objects[group.surface].seed;
            self.objects[group.surface].finish_seed = Some(seed);
            for &seat in &group.seats {
                self.objects[seat].finish_seed = Some(seed);
            }
            for object in &mut self.objects {
                if object.support == Some(group.surface)
                    && matches!(object.kind, ObjectKind::Laptop | ObjectKind::Monitor)
                    && !object.seed.is_multiple_of(5)
                {
                    object.finish_seed = Some(seed);
                }
            }
        }
    }
}
