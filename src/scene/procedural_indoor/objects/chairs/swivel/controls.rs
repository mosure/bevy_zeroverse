//! The release shaft enters the seat mechanism and its grip is carried by the
//! swivelling seat. Endpoints are in the chair's local frame, in metres.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Grip {
    Paddle,
    Sleeve,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ReleaseLever {
    pub grip: Grip,
    pub left: bool,
    pub reach_fraction: f32,
    pub fore_aft_fraction: f32,
    pub drop_m: f32,
    pub grip_length_m: f32,
    pub grip_width_m: f32,
}

impl ReleaseLever {
    pub(super) fn sample(r: &mut impl Rng) -> Self {
        let grip = if r.random_bool(0.63) {
            Grip::Paddle
        } else {
            Grip::Sleeve
        };
        Self {
            grip,
            left: r.random_bool(0.22),
            reach_fraction: r.random_range(0.41..0.54),
            fore_aft_fraction: r.random_range(-0.31..0.13),
            drop_m: r.random_range(0.012..0.037),
            grip_length_m: r.random_range(0.029..0.053),
            grip_width_m: match grip {
                Grip::Paddle => r.random_range(0.019..0.034),
                Grip::Sleeve => r.random_range(0.010..0.017),
            },
        }
    }

    pub(super) fn build(self, a: &mut Assembly, pan: Vec2, head: Vec2, y: f32) {
        let side = if self.left { -1. } else { 1. };
        let root = Vec3::new(side * (head.x * 0.5 - 0.006), y, -head.y * 0.13);
        let tip = Vec3::new(
            side * pan.x * self.reach_fraction,
            y - self.drop_m,
            pan.y * self.fore_aft_fraction,
        );
        let along = (tip - root).with_y(0.).normalize();
        let length = self
            .grip_length_m
            .min(root.with_y(tip.y).distance(tip) * 0.52);
        let neck = tip - along * length;
        let bend = root.lerp(neck, 0.48).with_y(y - self.drop_m * 0.13);
        a.part(Surface::Metal, "chair")
            .tube(&[root, bend, neck, neck + along * 0.006], 0.0036, 8);
        let g = a.part(Surface::Plastic, "chair");
        let width = self.grip_width_m * 0.5;
        match self.grip {
            Grip::Sleeve => {
                g.lathe(
                    &[
                        (0., 0.),
                        (width * 0.81, 0.),
                        (width, length * 0.16),
                        (width * 0.94, length * 0.85),
                        (width * 0.68, length),
                        (0., length),
                    ],
                    12,
                    Transform::from_translation(neck)
                        .with_rotation(Quat::from_rotation_arc(Vec3::Y, along)),
                );
            }
            Grip::Paddle => {
                let corners = [
                    (-0.40, 0.),
                    (0.40, 0.),
                    (0.57, 0.13),
                    (1., 0.62),
                    (0.87, 0.91),
                    (0.53, 1.),
                    (-0.53, 1.),
                    (-0.87, 0.91),
                    (-1., 0.62),
                    (-0.57, 0.13),
                ]
                .map(|(x, z)| Vec2::new(x * width, (z - 0.5) * length));
                g.profile_slab(
                    &corners,
                    0.0105,
                    0.0015,
                    Transform::from_translation(neck + along * (0.5 * length))
                        .with_rotation(Quat::from_rotation_y(along.x.atan2(along.z))),
                );
            }
        }
    }
}
