//! A shared printer needs a working surface and room to collect the output.
use super::*;
mod stand;
use crate::scene::procedural_indoor::{items::sizing, objects, placement::SurfacePattern};

fn demand(layout: IndoorLayout) -> f64 {
    use IndoorLayout::*;
    match layout {
        OpenOffice | Coworking => 0.87,
        Reception | Library | Workshop | Studio => 0.70,
        Classroom | Training => 0.39,
        Conference | LectureHall => 0.16,
        _ => 0.,
    }
}

impl IndoorManifest {
    /// Reserve a worktop before loose belongings. Carry an unmatched job to
    /// the powered-wall pass after occupants, outlets and floor details exist.
    pub(super) fn furnish_worktop_printer(&mut self) -> Option<IndoorObject> {
        let mut r = stream(self.seed, u64::from_le_bytes(*b"PrintUse"));
        let chance = demand(self.layout) * (0.38 + 0.62 * f64::from(self.density));
        if chance == 0.
            || !r.random_bool(chance)
            || self
                .objects
                .iter()
                .any(|o| o.kind == ObjectKind::Printer && !o.neighbor)
        {
            return None;
        }
        let size = sizing::loose(ObjectKind::Printer, &mut r).expect("printer dimensions");
        let mut printer = self.candidate(ObjectKind::Printer, Vec3::ZERO, size, 0., &mut r);
        printer.solid = false;
        objects::utilities::printer::reserve_media(&mut printer);
        let (lo, hi) = self.primary_room_bounds();
        let center = (lo + hi) * 0.5;
        let mut supports = Vec::new();
        for o in &self.objects {
            let weight = match o.kind {
                ObjectKind::DrawerUnit | ObjectKind::MobileCart => 2.2,
                ObjectKind::Cabinet => 2.8,
                ObjectKind::Desk => 1.,
                _ => continue,
            };
            let height = o.position.y + o.size.y - self.floor_height(o.position.xz());
            if !o.neighbor
                && (0.46..=1.18).contains(&height)
                && self.in_primary_room(o.position, 0.)
                && self.permits_table_item(o, ObjectKind::Printer)
            {
                // Weighted sampling without replacement. A rejected surface
                // cannot advance another surface's placement stream.
                let u = stream(o.seed, u64::from_le_bytes(*b"PrintTop")).random::<f32>();
                supports.push((-(1. - u).ln() / weight, o.id));
            }
        }
        let structure = self.structural_obstacles_with_ceiling(false);
        supports.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for (_, id) in supports.into_iter().take(5) {
            if self.fit_printer(&mut printer, &self.objects[id], center, &structure) {
                self.objects.push(printer);
                return None;
            }
        }
        Some(printer)
    }

    fn fit_printer(
        &self,
        printer: &mut IndoorObject,
        support: &IndoorObject,
        center: Vec2,
        structure: &[(Vec3, Vec3)],
    ) -> bool {
        let mut r = stream(support.seed, u64::from_le_bytes(*b"PrintPos"));
        let pattern = SurfacePattern::sample(&mut r);
        let mut half = support.size.xz() * 0.5;
        if support.kind == ObjectKind::Cabinet {
            let p = objects::storage::parameters(support);
            half.y -= support.size.x / p.columns as f32 * p.opening_radians.sin();
        }
        let toward = center - support.position.xz();
        let front = toward.x.atan2(toward.y) - support.yaw;
        for _ in 0..8 {
            let yaw = if r.random_bool(0.68) {
                front + r.random_range(-0.51..0.51)
            } else {
                pattern.yaw(&mut r)
            };
            let Some(place) = pattern.propose((-half, half), printer.size.xz(), yaw, 0.021, &mut r)
            else {
                continue;
            };
            printer.position = support.transform().transform_point(Vec3::new(
                place.center.x,
                support.size.y,
                place.center.y,
            ));
            printer.yaw = support.yaw + yaw;
            printer.support = Some(support.id);
            if self.prop_clear(printer, support, 0.012)
                && self.print_station_clear(printer, support.id, structure)
            {
                return true;
            }
        }
        false
    }

    fn print_station_clear(
        &self,
        printer: &IndoorObject,
        support: usize,
        structure: &[(Vec3, Vec3)],
    ) -> bool {
        // Include a hand's reach in front of the slot. Covers adjacent tall
        // furniture, other worktops and the people already seated at this desk.
        let forward = printer.transform().rotation * Vec3::Z;
        let (lo, hi) = printer.bounds();
        let reach = forward * 0.16;
        let lo = lo + reach.min(Vec3::ZERO);
        let hi = hi + reach.max(Vec3::ZERO);
        let overlaps = |(a, b): (Vec3, Vec3)| {
            lo.cmplt(b + Vec3::splat(0.008)).all() && hi.cmpgt(a - Vec3::splat(0.008)).all()
        };
        self.in_primary_room(
            printer.position + forward * (printer.size.z * 0.5 + 0.16),
            0.04,
        ) && !structure.iter().copied().any(overlaps)
            && !self.humans.iter().any(|h| overlaps(h.bounds()))
            && !self.objects.iter().any(|o| {
                o.id != support
                    && o.id != printer.id
                    && !o.neighbor
                    && overlaps(o.bounds())
                    && o.obstacle_bounds().into_iter().any(overlaps)
            })
    }
}

#[cfg(test)]
mod tests;
