//! Accepted functional groups and bounded, transactional meeting construction.
//! Receipts describe actual support/seat relationships; layout names are priors.
use super::*;
mod boardroom;
pub use boardroom::BoardroomSummary;
pub mod finishes;
mod meeting;
mod perimeter;
pub(super) use perimeter::SeatPath;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssemblyRole {
    Workstation,
    Meeting,
    StandingWorkstation,
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
        let boardroom = self.program.as_ref().and_then(|p| p.executive_boardroom);
        for group in &self.assembly_diagnostics.accepted {
            let surface = self
                .objects
                .get(group.surface)
                .ok_or("assembly references an absent surface")?;
            if !surface.kind.is_surface()
                || surface.neighbor
                || group.attempts == 0
                || if group.role == AssemblyRole::StandingWorkstation {
                    !super::super::objects::tables::is_raised_desk(surface)
                        || !group.seats.is_empty()
                        || group.requested_capacity != 1
                } else {
                    group.seats.is_empty()
                }
                || (group.role == AssemblyRole::Meeting && group.seats.len() < 2)
            {
                return Err("incomplete functional assembly".into());
            }
            if group.role == AssemblyRole::Meeting
                && boardroom.is_some_and(|p| {
                    surface.kind != ObjectKind::Table
                        || surface.variant != super::super::objects::tables::BOARDROOM_VARIANT
                        || !surface.size.xz().abs_diff_eq(p.table_size_m, 0.001)
                        || group.seats.len() < 8
                })
            {
                return Err("installed boardroom furniture disagrees with its program".into());
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
            let mut finish = finishes::parameters(seed);
            if self.objects[group.surface].variant
                == super::super::objects::tables::BOARDROOM_VARIANT
                && self.objects[group.surface].kind == ObjectKind::Table
            {
                finish.seat_coherence = 0.78 + 0.22 * finish.seat_coherence;
            }
            self.objects[group.surface].finish_seed = Some(seed);
            for &seat in &group.seats {
                let object = &mut self.objects[seat];
                object.finish_seed =
                    finishes::matches(seed, object.seed, finish.seat_coherence).then_some(seed);
            }
            for object in &mut self.objects {
                if object.support == Some(group.surface)
                    && matches!(object.kind, ObjectKind::Laptop | ObjectKind::Monitor)
                {
                    object.finish_seed =
                        finishes::matches(seed, object.seed, finish.device_coherence)
                            .then_some(seed);
                }
            }
        }
    }
}
