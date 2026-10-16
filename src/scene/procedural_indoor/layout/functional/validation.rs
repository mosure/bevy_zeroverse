use super::*;
use crate::sample::failure::{CaptureError, SceneRejectionKind};
impl IndoorManifest {
    pub fn validate_public_programs(&self) -> Result<(), String> {
        self.validate_public_programs_typed()
            .map_err(|e| e.to_string())
    }

    pub(crate) fn validate_public_programs_typed(&self) -> Result<(), CaptureError> {
        if !self.layout.is_public() {
            if !self.assembly_diagnostics.programs.is_empty() {
                return Err("public program receipt on an unrelated scene".into());
            }
            return Ok(());
        }
        let zones = &self
            .program
            .as_ref()
            .ok_or("missing public spatial program")?
            .zones;
        if self.assembly_diagnostics.programs.len() != zones.len() {
            return Err("missing functional zone receipt".into());
        }
        let mut rejection = None;
        for p in &self.assembly_diagnostics.programs {
            let reason =
                |s: &str| format!("seed {} {:?} zone {}: {s}", self.seed, p.activity, p.zone);
            let fail = |s: &str| CaptureError::Failed(reason(s));
            if zones.get(p.zone).is_none_or(|z| z.activity != p.activity) {
                return Err(fail("zone identity differs from functional receipt"));
            }
            if p.parameters != ProgramParameters::sample(self.seed, p.zone, self.density) {
                return Err(fail("functional parameters cannot replay from seed"));
            }
            if p.accepted_capacity != p.realized_capacity(&self.objects) {
                return Err(fail("capacity receipt differs from accepted objects"));
            }
            for kind in &p.required {
                if !p
                    .members
                    .iter()
                    .any(|&id| self.objects.get(id).is_some_and(|o| o.kind == *kind))
                {
                    rejection.get_or_insert_with(|| {
                        CaptureError::rejected(
                            self.seed,
                            SceneRejectionKind::MissingProgramAnchor,
                            reason(&format!("missing required {kind:?}")),
                        )
                    });
                }
            }
            if matches!(
                p.activity,
                IndoorLayout::Classroom
                    | IndoorLayout::LectureHall
                    | IndoorLayout::Cafe
                    | IndoorLayout::Cafeteria
            ) && p.accepted_capacity < 2
            {
                rejection.get_or_insert_with(|| {
                    CaptureError::rejected(
                        self.seed,
                        SceneRejectionKind::InsufficientProgramCapacity,
                        reason("fewer than two usable seats"),
                    )
                });
            }
            for &id in &p.members {
                let o = self
                    .objects
                    .get(id)
                    .ok_or_else(|| fail("dangling assembly member"))?;
                let (lo, hi) = o.bounds();
                if o.kind == ObjectKind::ExhibitPlinth
                    && !self.objects.iter().any(|child| child.support == Some(id))
                {
                    return Err(fail("exhibit plinth has no supported exhibit"));
                }
                if lo.x < p.min.x - 0.001
                    || lo.z < p.min.y - 0.001
                    || hi.x > p.max.x + 0.001
                    || hi.z > p.max.y + 0.001
                {
                    return Err(fail("assembly crosses its functional region"));
                }
            }
            for route in &p.reservations {
                if !route.min.is_finite()
                    || !route.max.is_finite()
                    || route.min.cmpge(route.max).any()
                {
                    return Err(fail("invalid reserved route"));
                }
                for o in self.objects.iter().filter(|o| o.solid && !o.neighbor) {
                    let (a, b) = o.bounds();
                    if route.overlaps(a, b) {
                        return Err(fail("reserved circulation obstructed"));
                    }
                }
            }
            for service in &p.services {
                if self.objects.get(service.source).is_none()
                    || self.objects.get(service.target).is_none()
                    || !service.source_point.is_finite()
                    || !service.target_point.is_finite()
                {
                    return Err(fail("invalid service relationship"));
                }
            }
        }
        // A recoverable capacity/placement rejection must never hide a corrupt
        // receipt, service relationship or reservation elsewhere in the room.
        rejection.map_or(Ok(()), Err)
    }
}
