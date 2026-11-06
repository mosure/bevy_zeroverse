//! Wall displays and supported clutter use free-space rejection, not fixed anchors.
use super::super::architecture::facade::{self, FacadeSide};
use super::*;
mod electrical;
pub use electrical::{electrical_parameters, ElectricalProgram};

pub(super) fn drink(rng: &mut ChaCha8Rng) -> (ObjectKind, Vec3) {
    let kind = [
        ObjectKind::Mug,
        ObjectKind::CoffeeCup,
        ObjectKind::WaterBottle,
        ObjectKind::SodaCan,
    ][rng.random_range(0..4)];
    (
        kind,
        super::super::items::sizing::loose(kind, rng).expect("drink size"),
    )
}

impl IndoorManifest {
    /// Reserve an aligned service appliance before small accessories compete for
    /// its space. Sampling a big appliance as late uniform clutter almost never
    /// fits a realistic narrow cart, even when its geometry is perfectly valid.
    pub(super) fn appliance_group(&mut self, support: &IndoorObject) {
        let mut rng = stream(support.seed, 5530);
        let program = super::super::items::parameters(self, support);
        if support.size.x < 0.52
            || support.size.z < 0.35
            || !rng.random_bool((0.38 + program.task_weights[2] * 0.55) as f64)
        {
            return;
        }
        // Preserve the domain's optional-clutter thinning when choosing the
        // assembly. Functional kitchen/cafe stations use a separate entry point.
        if self
            .domain()
            .is_some_and(|d| !rng.random_bool((0.12 + 0.88 * d.clutter) as f64))
        {
            return;
        }
        self.dress_refreshment_counter(support, None);
    }

    pub(super) fn attachment_wall(&self, rng: &mut ChaCha8Rng) -> (Transform, f32) {
        if let Some(e) = &self.envelope {
            let wall = &e.walls[rng.random_range(0..e.walls.len())];
            let a = e.footprint[wall.edge];
            let b = e.footprint[(wall.edge + 1) % e.footprint.len()];
            (e.wall_transform(wall.edge), a.distance(b))
        } else {
            let side = FacadeSide::ALL[rng.random_range(0..FacadeSide::ALL.len())];
            (side.transform(self.room_size), side.span(self.room_size))
        }
    }
    fn attachment_clear(&self, tf: Transform, span: f32, obj: &IndoorObject) -> bool {
        let local = tf.rotation.inverse() * (obj.position - tf.translation);
        let (lo, hi) = obj.bounds();
        let trim = super::super::architecture::trim::TrimProgram::for_scene(self);
        let base = trim
            .and_then(|t| t.base.as_ref().map(|s| s.height_m + t.joint_m + 0.01))
            .unwrap_or(0.);
        let crown = trim
            .and_then(|t| t.crown.as_ref().map(|s| s.height_m + t.joint_m + 0.01))
            .unwrap_or(0.08);
        local.x.abs() + obj.size.x * 0.5 < span * 0.5 - 0.05
            && lo.y > base
            && [
                lo.xz(),
                hi.xz(),
                Vec2::new(lo.x, hi.z),
                Vec2::new(hi.x, lo.z),
            ]
            .into_iter()
            .all(|p| hi.y < self.ceiling_height(p) - crown)
            && !(self.envelope.is_none()
                && super::super::architecture::details::overlaps_niche(self, lo, hi))
            && !facade::overlaps_opening(self, lo, hi, 0.05)
    }

    /// Same mounted-wall contract for generation and archived-manifest audits.
    pub(crate) fn wall_attachment_clear(&self, obj: &IndoorObject) -> bool {
        let Some(offset) = super::super::objects::wall::mount_offset(obj) else {
            return false;
        };
        let normal = Quat::from_rotation_y(obj.yaw) * Vec3::Z;
        let clear = |tf: Transform, span: f32| {
            let local = tf.rotation.inverse() * (obj.position - tf.translation);
            normal.dot(tf.rotation * Vec3::Z) > 0.99999
                && (local.z - offset).abs() < 0.001
                && self.attachment_clear(tf, span, obj)
        };
        if let Some(e) = &self.envelope {
            e.walls.iter().any(|wall| {
                let a = e.footprint[wall.edge];
                let b = e.footprint[(wall.edge + 1) % e.footprint.len()];
                clear(e.wall_transform(wall.edge), a.distance(b))
            })
        } else {
            FacadeSide::ALL
                .into_iter()
                .any(|side| clear(side.transform(self.room_size), side.span(self.room_size)))
        }
    }

    /// Preserve each category's former proposal-count marginal, but remove the
    /// shared categorical draw that made extra pictures imply fewer displays.
    pub(super) fn wall_decoration_count(&self, kind: ObjectKind) -> usize {
        let public = self.layout.is_public()
            && !matches!(
                self.layout,
                IndoorLayout::Classroom | IndoorLayout::LectureHall
            );
        let dining = matches!(self.layout, IndoorLayout::Cafe | IndoorLayout::Cafeteria);
        let (domain, probability) = match kind {
            ObjectKind::WallArt => (
                sampling::Domain::WallArt,
                if public && !dining { 0.6 } else { 0.2 },
            ),
            ObjectKind::Display => (
                sampling::Domain::Displays,
                if public && dining { 0.6 } else { 0.2 },
            ),
            ObjectKind::Whiteboard => (
                sampling::Domain::Whiteboards,
                if public { 0.0 } else { 0.4 },
            ),
            ObjectKind::Clock => (sampling::Domain::Clocks, 0.2),
            _ => unreachable!("not standalone wall decor"),
        };
        let mut r = domain.rng(self.seed);
        let opportunities = r.random_range(1..=13);
        (0..opportunities)
            .filter(|_| r.random_bool(probability))
            .count()
    }

    pub(super) fn wall_decoration_kind(&mut self, kind: ObjectKind) {
        let domain = match kind {
            ObjectKind::WallArt => sampling::Domain::WallArt,
            ObjectKind::Display => sampling::Domain::Displays,
            ObjectKind::Whiteboard => sampling::Domain::Whiteboards,
            ObjectKind::Clock => sampling::Domain::Clocks,
            _ => unreachable!("not standalone wall decor"),
        };
        let count = self.wall_decoration_count(kind);
        if count == 0 {
            return;
        }
        let mut occupied: Vec<_> = self
            .objects
            .iter()
            .map(IndoorObject::bounds)
            .chain(self.columns())
            .collect();
        for ordinal in 0..count {
            let mut local = domain.proposal(self.seed, ordinal);
            let rng = &mut local;
            let (tf, span) = self.attachment_wall(rng);
            let size = if kind == ObjectKind::Clock {
                super::super::objects::clocks::dimensions(rng)
            } else {
                super::wall_composition::standalone_size(kind, rng)
            };
            if span < size.x + 0.72 {
                continue;
            }
            let along = rng
                .random_range(-span * 0.5 + size.x * 0.5 + 0.35..span * 0.5 - size.x * 0.5 - 0.35);
            let anchor = tf.transform_point(Vec3::new(along, 0., 0.));
            let floor = self.floor_height(anchor.xz());
            let roof = self.ceiling_height(anchor.xz());
            let y = if kind == ObjectKind::WallArt {
                let lo = (floor + 1.15).max(floor + 0.25 + size.y * 0.5);
                let hi = (floor + 2.05).min(self.ceiling_height(anchor.xz()) - 0.25 - size.y * 0.5);
                if lo >= hi {
                    continue;
                }
                rng.random_range(lo..hi) - size.y * 0.5
            } else {
                let high = (roof - size.y - 0.30).min(floor + 2.15);
                if high <= floor + 0.95 {
                    continue;
                }
                rng.random_range(floor + 0.95..high)
            };
            let position = tf.transform_point(Vec3::new(along, y, 0.));
            let yaw = tf.rotation.to_euler(EulerRot::YXZ).0;
            let mut object = self.candidate(kind, position, size, yaw, rng);
            object.solid = false;
            object.position +=
                tf.rotation * Vec3::Z * super::super::objects::wall::mount_offset(&object).unwrap();
            let (lo, hi) = object.bounds();
            if !self.attachment_clear(tf, span, &object) {
                continue;
            }
            if occupied.iter().copied().any(|(a, b)| {
                lo.cmplt(b + Vec3::splat(0.06)).all() && hi.cmpgt(a - Vec3::splat(0.06)).all()
            }) {
                continue;
            }
            occupied.push((lo, hi));
            self.objects.push(object);
        }
    }
}

#[cfg(test)]
mod tests;
