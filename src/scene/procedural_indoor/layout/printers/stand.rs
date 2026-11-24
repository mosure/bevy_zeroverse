//! Where worktops are occupied, a small office can dedicate a rolling cart
//! or file pedestal to printing. Place both objects or neither.
use super::*;

impl IndoorManifest {
    pub(in crate::scene::procedural_indoor::layout) fn furnish_print_stand(
        &mut self,
        mut printer: IndoorObject,
    ) {
        let mut r = stream(self.seed, u64::from_le_bytes(*b"PrintPod"));
        if !r.random_bool(0.66)
            || self
                .objects
                .iter()
                .any(|o| !o.neighbor && o.kind == ObjectKind::Printer)
        {
            return;
        }
        // Anchor the furniture along a powered wall without covering the outlet.
        let mut outlets: Vec<_> = self
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::WallOutlet && !o.neighbor)
            .map(|o| {
                (
                    stream(o.seed, u64::from_le_bytes(*b"PrintBay")).random::<u32>(),
                    o.transform(),
                )
            })
            .collect();
        outlets.sort_unstable_by_key(|o| o.0);
        if outlets.is_empty() {
            return;
        }
        let structure = self.structural_obstacles_with_ceiling(false);
        let kind = if r.random_bool(0.32) {
            ObjectKind::MobileCart
        } else {
            ObjectKind::DrawerUnit
        };
        let size = Vec3::new(
            printer.size.x + r.random_range(0.12..0.31),
            r.random_range(0.61..0.85),
            printer.size.z + r.random_range(0.12..0.26),
        );
        let mut stand = self.candidate(kind, Vec3::ZERO, size, 0., &mut r);
        printer.id = stand.id + 1;
        for (_, outlet) in outlets.into_iter().take(4) {
            let first = if r.random_bool(0.5) { 1. } else { -1. };
            let normal = outlet.rotation * Vec3::Z;
            for side in [first, -first] {
                let point = outlet.transform_point(Vec3::new(
                    side * (size.x * 0.5 + r.random_range(0.14..0.56)),
                    0.,
                    size.z * 0.5 + r.random_range(0.13..0.34),
                ));
                stand.position = point.with_y(self.floor_height(point.xz()));
                stand.yaw = normal.x.atan2(normal.z) + r.random_range(-0.16..0.16);
                let (lo, hi) = stand.bounds();
                if !self.in_primary_room(stand.position, 0.5 * size.x.max(size.z))
                    || !self.placement_clear(&stand, 0.13)
                    || self
                        .humans
                        .iter()
                        .map(|h| h.bounds())
                        .chain(
                            self.objects
                                .iter()
                                .filter(|o| !o.neighbor)
                                .map(IndoorObject::bounds),
                        )
                        .any(|(a, b)| {
                            lo.cmplt(b + Vec3::splat(0.009)).all()
                                && hi.cmpgt(a - Vec3::splat(0.009)).all()
                        })
                {
                    continue;
                }
                // The stock is fixed across attempts. Only placement changes.
                let approach = stand.position.xz() + normal.xz();
                if self.fit_printer(&mut printer, &stand, approach, &structure) {
                    self.objects.push(stand);
                    self.objects.push(printer);
                    return;
                }
            }
        }
    }
}
