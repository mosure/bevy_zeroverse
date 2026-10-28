//! Furniture-related proposals, expressed in the accepted anchor's frame.
//! Rear service storage, end tables and conversational footrests have different
//! priors. Admission remains the full room/circulation/OBB clearance policy.
use super::*;
use std::f32::consts::{FRAC_PI_2, PI};

pub(super) struct Placement {
    sides: [f32; 4], // left, right, rear, front
    kind: ObjectKind,
    spread: f32,
    setback: f32,
    along: f32,
}

impl Placement {
    pub(super) fn sample(anchor: &IndoorObject, kind: ObjectKind, r: &mut impl Rng) -> Self {
        use ObjectKind::*;
        let lounge = matches!(anchor.kind, Sofa | CoffeeTable);
        let sides = match kind {
            DrawerUnit => [1., 1., 0.50, 0.12],
            SideTable if anchor.kind == Sofa => [1., 1., 0.22, 0.16],
            Ottoman if lounge => [0.35, 0.35, 0.08, 1.8],
            RoomDivider => [1., 1., 1.1, 0.12],
            Bench => [1., 1., 0.65, 0.45],
            _ => [1., 1., 0.75, 0.75],
        };
        Self {
            sides: sides.map(|weight| weight * r.random_range(0.55..1.65)),
            kind,
            spread: r.random_range(0.025..0.48),
            setback: r.random_range(0.14..0.52),
            along: r.random_range(0.40..0.90),
        }
    }

    pub(super) fn propose(
        &self,
        anchor: &IndoorObject,
        size: Vec3,
        r: &mut impl Rng,
    ) -> (Vec3, f32) {
        let mut choice = r.random_range(0.0..self.sides.iter().sum::<f32>());
        let side = self
            .sides
            .iter()
            .position(|w| {
                choice -= w;
                choice <= 0.
            })
            .unwrap_or(3);
        let outward = [-FRAC_PI_2, FRAC_PI_2, PI, 0.][side];
        let heading = match self.kind {
            // Screens extend along the selected furniture edge; benches and
            // footrests may face a conversation group or the surrounding room.
            ObjectKind::RoomDivider => outward,
            ObjectKind::Bench | ObjectKind::Ottoman => {
                outward + if r.random_bool(0.60) { PI } else { 0. }
            }
            _ if r.random_bool(0.54) => 0.,
            _ => outward,
        } + r.random_range(-self.spread..self.spread);
        let (sin, cos) = heading.sin_cos();
        let extent = Vec2::new(
            size.x * cos.abs() + size.z * sin.abs(),
            size.x * sin.abs() + size.z * cos.abs(),
        );
        let gap = self.setback * r.random_range(0.70..1.35);
        let along = r.random_range(-self.along..self.along);
        let local = if side < 2 {
            Vec3::new(
                if side == 0 { -1. } else { 1. } * ((anchor.size.x + extent.x) * 0.5 + gap),
                0.,
                along * anchor.size.z,
            )
        } else {
            Vec3::new(
                along * anchor.size.x,
                0.,
                if side == 2 { -1. } else { 1. } * ((anchor.size.z + extent.y) * 0.5 + gap),
            )
        };
        (
            anchor.position + Quat::from_rotation_y(anchor.yaw) * local,
            anchor.yaw + heading,
        )
    }
}
