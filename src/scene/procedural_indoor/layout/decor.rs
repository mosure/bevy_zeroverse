//! Wall displays and supported clutter use free-space rejection, not fixed anchors.
use super::super::architecture::facade::{self, FacadeSide};
use super::*;

pub(super) fn drink(rng: &mut ChaCha8Rng) -> (ObjectKind, Vec3) {
    match rng.random_range(0..4) {
        0 => (
            ObjectKind::Mug,
            Vec3::new(
                rng.random_range(0.10..0.14),
                rng.random_range(0.072..0.13),
                rng.random_range(0.075..0.11),
            ),
        ),
        1 => {
            let d = rng.random_range(0.068..0.095);
            (
                ObjectKind::CoffeeCup,
                Vec3::new(d, rng.random_range(0.085..0.15), d),
            )
        }
        2 => {
            let d = rng.random_range(0.055..0.09);
            (
                ObjectKind::WaterBottle,
                Vec3::new(d, rng.random_range(0.16..0.31), d),
            )
        }
        _ => {
            let d = rng.random_range(0.057..0.075);
            (
                ObjectKind::SodaCan,
                Vec3::new(d, rng.random_range(0.09..0.18), d),
            )
        }
    }
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
        let kind = if rng.random_bool(0.55) {
            ObjectKind::CoffeeMaker
        } else {
            ObjectKind::Kettle
        };
        let size = if kind == ObjectKind::CoffeeMaker {
            Vec3::new(
                rng.random_range(0.24..0.31),
                rng.random_range(0.29..0.41),
                (support.size.z * 0.69).clamp(0.27, 0.38),
            )
        } else {
            Vec3::new(0.18, rng.random_range(0.20..0.27), 0.25)
        };
        let side = if rng.random_bool(0.5) { -1. } else { 1. };
        self.prop(
            support,
            kind,
            Vec3::new(side * support.size.x * 0.16, 0., -support.size.z * 0.03),
            size,
            rng.random_range(-0.06..0.06),
            &mut rng,
        );
        let (drink, size) = drink(&mut rng);
        self.prop(
            support,
            drink,
            Vec3::new(-side * support.size.x * 0.28, 0., support.size.z * 0.16),
            size,
            rng.random_range(-0.8..0.8),
            &mut rng,
        );
    }

    fn attachment_wall(&self, rng: &mut ChaCha8Rng) -> (Transform, f32) {
        if let Some(e) = &self.envelope {
            let wall = &e.walls[rng.random_range(0..e.walls.len())];
            let a = e.footprint[wall.edge];
            let b = e.footprint[(wall.edge + 1) % e.footprint.len()];
            (e.wall_transform(wall.edge), a.distance(b))
        } else {
            let side = FacadeSide::ALL[rng.random_range(0..3)];
            (side.transform(self.room_size), side.span(self.room_size))
        }
    }
    fn attachment_clear(&self, tf: Transform, span: f32, obj: &IndoorObject) -> bool {
        let local = tf.rotation.inverse() * (obj.position - tf.translation);
        let (lo, hi) = obj.bounds();
        local.x.abs() + obj.size.x * 0.5 < span * 0.5 - 0.05
            && [
                lo.xz(),
                hi.xz(),
                Vec2::new(lo.x, hi.z),
                Vec2::new(hi.x, lo.z),
            ]
            .into_iter()
            .all(|p| hi.y < self.ceiling_height(p) - 0.08)
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

    pub(super) fn wall_hardware(&mut self, rng: &mut ChaCha8Rng) {
        for i in 0..rng.random_range(4..10) {
            let switch = i < 2;
            let kind = if switch {
                ObjectKind::LightSwitch
            } else {
                ObjectKind::WallOutlet
            };
            let gangs = rng.random_range(1..=3);
            let size = Vec3::new(0.075 * gangs as f32, rng.random_range(0.10..0.125), 0.018);
            let (tf, span) = self.attachment_wall(rng);
            let along = rng.random_range(-0.40..0.40);
            let height = if switch {
                rng.random_range(1.05..1.22)
            } else {
                rng.random_range(0.22..0.38)
            };
            let position = tf.transform_point(Vec3::new(along * span, height, 0.));
            let yaw = tf.rotation.to_euler(EulerRot::YXZ).0;
            let mut obj = self.candidate(kind, position, size, yaw, rng);
            obj.solid = false;
            obj.position +=
                tf.rotation * Vec3::Z * super::super::objects::wall::mount_offset(&obj).unwrap();
            // One electrical convention per scene; finishes/gang counts can vary.
            obj.variant = (self.seed % 4) as u32;
            let (lo, hi) = obj.bounds();
            if !self.attachment_clear(tf, span, &obj)
                || self
                    .objects
                    .iter()
                    .map(IndoorObject::bounds)
                    .chain(self.columns())
                    .any(|(a, b)| {
                        lo.cmplt(b + Vec3::splat(0.015)).all()
                            && hi.cmpgt(a - Vec3::splat(0.015)).all()
                    })
            {
                continue;
            }
            self.objects.push(obj);
        }
    }
    pub(super) fn wall_decorations(&mut self, rng: &mut ChaCha8Rng) {
        let h = self.room_size.y;
        let count = rng.random_range(2..=9);
        for _ in 0..count {
            let (tf, span) = self.attachment_wall(rng);
            let (kind, size) = match rng.random_range(0..5) {
                0 => (
                    ObjectKind::Display,
                    Vec3::new(
                        rng.random_range(0.85..2.1),
                        rng.random_range(0.55..1.18),
                        0.07,
                    ),
                ),
                1 | 2 => (
                    ObjectKind::Whiteboard,
                    Vec3::new(
                        rng.random_range(0.7..2.3),
                        rng.random_range(0.65..1.35),
                        rng.random_range(0.12..0.18),
                    ),
                ),
                3 => (
                    ObjectKind::WallArt,
                    Vec3::new(
                        rng.random_range(0.35..1.2),
                        rng.random_range(0.35..1.25),
                        0.045,
                    ),
                ),
                _ => (
                    ObjectKind::Clock,
                    Vec3::splat(rng.random_range(0.22..0.43)).with_z(0.045),
                ),
            };
            if span < size.x + 0.72 {
                continue;
            }
            let along = rng
                .random_range(-span * 0.5 + size.x * 0.5 + 0.35..span * 0.5 - size.x * 0.5 - 0.35);
            let y = rng.random_range(0.95..(h - size.y - 0.3).clamp(0.96, 2.15));
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
            if self
                .objects
                .iter()
                .map(IndoorObject::bounds)
                .chain(self.columns())
                .any(|(a, b)| {
                    lo.cmplt(b + Vec3::splat(0.06)).all() && hi.cmpgt(a - Vec3::splat(0.06)).all()
                })
            {
                continue;
            }
            self.objects.push(object);
        }
    }
    pub(super) fn scatter_clutter(&mut self, _rng: &mut ChaCha8Rng) {
        let clutter = self.domain().map_or(0.5, |d| d.clutter) * self.density;
        let surfaces: Vec<_> = self
            .objects
            .iter()
            .filter(|o| o.kind.is_surface())
            .cloned()
            .collect();
        for support in surfaces {
            let program = super::super::items::parameters(self, &support);
            let mut rng = stream(support.seed, 5510);
            let attempts = (clutter * program.occupancy * (12.0 + 6.0 * support.size.xz().length()))
                .clamp(0.0, 48.0) as usize;
            for _ in 0..attempts {
                let (kind, size) = program.item(&support, &mut rng);
                // Large appliances are occasional service equipment, not piles
                // of repeated identical machines on a single support.
                if matches!(
                    kind,
                    ObjectKind::CoffeeMaker
                        | ObjectKind::Kettle
                        | ObjectKind::Printer
                        | ObjectKind::DeskLamp
                        | ObjectKind::DeskFan
                ) && self
                    .objects
                    .iter()
                    .any(|o| o.support == Some(support.id) && o.kind == kind)
                {
                    continue;
                }
                let (mut offset, mut yaw) = super::surfaces::scatter_proposal(
                    &self.objects,
                    &support,
                    kind,
                    size,
                    &mut rng,
                );
                if rng.random_bool(program.clustering as f64) {
                    offset = offset.lerp(
                        Vec3::new(
                            program.focal_point.x * support.size.x,
                            0.,
                            program.focal_point.y * support.size.z,
                        ),
                        0.35,
                    );
                }
                if rng.random_bool(program.alignment as f64) {
                    yaw = program.rotation + rng.random_range(-0.22..0.22);
                }
                self.prop(&support, kind, offset, size, yaw, &mut rng);
            }
        }
    }
}
