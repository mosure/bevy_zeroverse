//! Accepted functional groups and bounded, transactional meeting construction.
//! Receipts describe actual support/seat relationships; layout names are priors.
use super::*;
mod meeting;
mod perimeter;
#[cfg(test)]
mod tests;

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
