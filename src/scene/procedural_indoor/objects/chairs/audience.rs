//! Teaching seats. The writing surface, real hinge and back shell use the same
//! metric installation as placement clearance, including left-handed tablets.
use super::*;
mod backrest;
mod frame;
mod tablet;
use tablet::Tablet;
#[cfg(test)]
mod tests;

pub fn is_folded(o: &IndoorObject) -> bool {
    o.variant == 11 && stream(o.seed, 8260).random_bool(0.22)
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct TeachingSeat {
    pub family: &'static str,
    pub folded: bool,
    pub seat_height_metres: f32,
    pub width_metres: f32,
    pub height_metres: f32,
    pub depth_metres: f32,
    seat_width: f32,
    pub(crate) seat_depth: f32,
    pad: f32,
    crown: f32,
    dish: f32,
    hinge: Vec3,
    pub(crate) fold_angle: f32,
    back: backrest::Backrest,
    pub(crate) tablet: Option<Tablet>,
    face_stock: Surface,
    shell_stock: Surface,
    frame_stock: Surface,
}

impl TeachingSeat {
    pub(crate) fn sample(o: &IndoorObject) -> Self {
        let fixed = o.variant == 11;
        let mut r = stream(o.seed, u64::from_le_bytes(*b"TeachFit"));
        let mut finish = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"SeatSkin") ^ o.variant as u64,
        );
        let shell_stock = if finish.random_bool(if fixed { 0.48 } else { 0.54 }) {
            Surface::Wood
        } else {
            Surface::Plastic
        };
        let face_stock = if fixed {
            if finish.random_bool(0.64) {
                Surface::Fabric
            } else {
                Surface::FabricAlt
            }
        } else {
            shell_stock
        };
        let frame_stock = if !fixed && finish.random_bool(0.37) {
            Surface::Chrome
        } else {
            Surface::Metal
        };
        let pad = if fixed {
            r.random_range(0.058..0.079)
        } else {
            r.random_range(0.027..0.039)
        };
        let seat_height_metres = 0.47;
        let dish = if fixed { 0.006 } else { 0.003 };
        let seat_y = seat_height_metres - pad * 0.5 + dish;
        let seat_width = (o.size.x * r.random_range(0.64..0.73)).min(0.59);
        let fraction = r.random_range(0.29..0.39);
        let fold_angle: f32 = r.random_range(1.16..1.39);
        let (sin, cos) = fold_angle.sin_cos();
        let mut seat_depth = (o.size.z * r.random_range(0.60..0.70)).min(0.51);
        if fixed {
            // The same physical seat fits in either state, including short
            // rows. The upholstered front edge reaches highest when folded.
            seat_depth = seat_depth
                .min((o.size.y - 0.009 - seat_y - 0.5 * pad * cos) / ((0.5 + fraction) * sin));
        }
        let hinge = Vec3::new(0., seat_y, fraction * seat_depth);
        // The rear seat corner initially swings slightly behind its open
        // position. Clear the complete circular sweep, not just the end poses.
        let back_z = hinge.z + (0.5 * seat_depth - hinge.z).hypot(0.5 * pad) + 0.008;
        let back = backrest::Backrest::sample(o, seat_width, back_z, fixed, &mut r);
        let tablet = (!fixed).then(|| Tablet::sample(o, shell_stock, &mut r));
        Self {
            family: if fixed { "auditorium" } else { "tablet_arm" },
            folded: is_folded(o),
            seat_height_metres,
            width_metres: o.size.x,
            height_metres: o.size.y,
            depth_metres: o.size.z,
            seat_width,
            seat_depth,
            pad,
            dish,
            crown: r.random_range(2.9..4.9),
            hinge,
            fold_angle,
            back,
            tablet,
            face_stock,
            shell_stock,
            frame_stock,
        }
    }

    pub(crate) fn carry_grip_size(&self) -> Option<Vec2> {
        self.back.grip_dimensions()
    }

    fn seat_pose(&self, angle: f32) -> Transform {
        let rotation = Quat::from_rotation_x(angle);
        Transform::from_translation(self.hinge - rotation * (Vec3::Z * self.hinge.z))
            .with_rotation(rotation)
    }

    fn pose(&self) -> Transform {
        self.seat_pose(if self.folded { self.fold_angle } else { 0. })
    }

    fn build(&self, a: &mut Assembly) {
        let size = Vec3::new(self.seat_width, self.pad, self.seat_depth);
        // Both shells have a soft waterfall nose. The rigid tablet-chair pan is
        // shallower; the fixed seat exposes a protective shell when tipped up.
        let mut underside = Geometry::default();
        let separate = self.face_stock != self.shell_stock;
        a.part(self.face_stock, "chair").seat_panels(
            size,
            super::super::super::geometry::CushionProfile {
                roundness: if self.tablet.is_some() { 5.5 } else { 4.2 },
                crown: self.crown,
                dish: self.dish,
            },
            self.pose(),
            separate.then_some(&mut underside),
        );
        if separate {
            a.part(self.shell_stock, "chair").append(underside);
        }
        self.back.build(a, self.face_stock, self.shell_stock);
        frame::build(a, self);
        if let Some(t) = self.tablet {
            t.build(a);
        }
    }
}

pub(super) fn clearance_boxes(o: &IndoorObject) -> Vec<(Vec3, Vec3)> {
    let p = TeachingSeat::sample(o);
    let half = o.size * 0.5;
    let mut boxes = vec![
        (
            Vec3::new(-half.x, 0., -half.z),
            Vec3::new(half.x, 0.50, half.z),
        ),
        (
            Vec3::new(-half.x, 0.48, p.back.z - 0.017),
            Vec3::new(half.x, o.size.y, half.z),
        ),
    ];
    if let Some(t) = p.tablet {
        let outer = half.x * t.side();
        let extent = t.projected_half();
        let inner = t.center.x - extent.x * t.side();
        boxes.push((
            Vec3::new(outer.min(inner), 0.48, t.center.z - extent.y),
            Vec3::new(
                outer.max(inner),
                t.center.y + 0.5 * t.thickness,
                t.center.z + extent.y,
            ),
        ));
    }
    if p.folded {
        let tf = p.pose();
        let extent = tf.rotation.mul_vec3(Vec3::Y).abs() * p.pad * 0.5
            + tf.rotation.mul_vec3(Vec3::Z).abs() * p.seat_depth * 0.5
            + Vec3::X * p.seat_width * 0.5;
        boxes.push((tf.translation - extent, tf.translation + extent));
    }
    boxes
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    TeachingSeat::sample(o).build(a);
}
