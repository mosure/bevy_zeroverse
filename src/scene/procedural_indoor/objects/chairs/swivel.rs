//! Swivel chairs share an axial bearing between the rolling base and the seat.
//! Product stock determines the column and control, while the installed seat
//! pan determines the bearing elevation and the length of exposed piston.
use super::*;

mod blade;
mod controls;
#[cfg(test)]
mod tests;
pub use blade::CastSpoke;
pub use controls::{Grip, ReleaseLever};

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct SwivelPedestal {
    pub table_order: bool,
    pub hub_height_m: f32,
    pub barrel_radius_m: f32,
    pub head_height_m: f32,
    pub head_width_fraction: f32,
    pub head_depth_fraction: f32,
    pub column_finish: Surface,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gas_lift: Option<GasLift>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cast_spoke: Option<CastSpoke>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disc: Option<DiscFoot>,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct GasLift {
    pub piston_radius_m: f32,
    /// Fraction of the clear length above the hub occupied by polished rod.
    pub exposed_fraction: f32,
    pub lever: ReleaseLever,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct DiscFoot {
    pub rim_height_m: f32,
    pub shoulder_fraction: f32,
    pub shoulder_height_fraction: f32,
}

impl ChairProgram {
    pub fn swivel_pedestal(&self, o: &IndoorObject, version: u32) -> Option<SwivelPedestal> {
        if version < 260
            || o.kind != ObjectKind::Chair
            || !matches!(o.variant, 0 | 1 | 5 | 9)
            || self.base_style >= 4
        {
            return None;
        }
        let mut item = stream(o.seed, u64::from_le_bytes(*b"LiftSeat"));
        let order = o
            .finish_seed
            .zip(o.interaction_target)
            .filter(|_| item.random_bool(0.76));
        let mut lot = order.map(|(finish, table)| {
            stream(
                finish,
                u64::from_le_bytes(*b"LiftTool")
                    ^ (table as u64).rotate_left(17)
                    ^ u64::from(self.base_style),
            )
        });
        let rng = lot.as_mut().unwrap_or(&mut item);
        let barrel_radius_m = rng.random_range(0.022..0.036);
        let fixed_probability = if self.base_style == 3 {
            0.61
        } else if self.casters {
            0.025
        } else {
            0.14
        };
        let gas_lift = (!rng.random_bool(fixed_probability)).then(|| GasLift {
            piston_radius_m: barrel_radius_m * rng.random_range(0.54..0.73),
            exposed_fraction: rng.random_range(0.28..0.68),
            lever: ReleaseLever::sample(rng),
        });
        Some(SwivelPedestal {
            table_order: order.is_some(),
            hub_height_m: rng.random_range(0.125..0.194),
            barrel_radius_m,
            head_height_m: rng.random_range(0.020..0.042),
            head_width_fraction: rng.random_range(0.29..0.43),
            head_depth_fraction: rng.random_range(0.32..0.50),
            column_finish: if rng.random_bool(0.66) {
                Surface::Metal
            } else {
                Surface::Chrome
            },
            gas_lift,
            cast_spoke: (self.base_style == 2).then(|| CastSpoke::sample(rng)),
            disc: (self.base_style == 3).then(|| DiscFoot {
                rim_height_m: rng.random_range(0.008..0.025),
                shoulder_fraction: rng.random_range(0.46..0.93),
                shoulder_height_fraction: rng.random_range(0.28..0.59),
            }),
        })
    }
}

impl SwivelPedestal {
    pub(crate) fn bearing_y(&self, pan_bottom: f32) -> f32 {
        pan_bottom - self.head_height_m
    }

    pub(super) fn hub_radius(&self) -> f32 {
        self.barrel_radius_m * 1.42
    }

    pub(super) fn upper(&self, a: &mut Assembly, pan: Vec2, pan_bottom: f32) {
        let bearing = self.bearing_y(pan_bottom);
        let top = pan_bottom + 0.0012;
        let head = pan * Vec2::new(self.head_width_fraction, self.head_depth_fraction);
        a.box_part(
            Surface::Metal,
            "chair",
            Vec3::Y * ((bearing + top) * 0.5),
            Vec3::new(head.x, top - bearing, head.y),
            0.004,
        );
        if let Some(lift) = self.gas_lift {
            lift.lever
                .build(a, pan, head, bearing + self.head_height_m * 0.51);
        }
    }

    pub(super) fn column(&self, a: &mut Assembly, pan_bottom: f32) {
        let h = self.hub_height_m;
        let barrel = self.barrel_radius_m;
        let hub = self.hub_radius();
        let bearing = self.bearing_y(pan_bottom);
        let end = self.gas_lift.map_or(bearing + 0.0015, |lift| {
            bearing - lift.exposed_fraction * (bearing - h - 0.024) - 0.0028
        });
        a.part(self.column_finish, "chair").lathe(
            &[
                (0., h - 0.022),
                (hub * 0.85, h - 0.022),
                (hub, h - 0.015),
                (hub, h + 0.014),
                (barrel, h + 0.024),
                (barrel, end - 0.003),
                (barrel * 0.94, end),
                (0., end),
            ],
            20,
            Transform::IDENTITY,
        );
        if let Some(lift) = self.gas_lift {
            // The rod seats in the rotating head. Its lower end and wiper
            // overlap the barrel internally, leaving one visible sealing lip.
            let r = lift.piston_radius_m;
            a.part(Surface::Chrome, "chair").lathe(
                &[
                    (0., end - 0.003),
                    (r, end - 0.003),
                    (r, bearing + 0.0015),
                    (0., bearing + 0.0015),
                ],
                16,
                Transform::IDENTITY,
            );
            a.part(Surface::Rubber, "chair").lathe(
                &[
                    (r * 0.985, end - 0.001),
                    (barrel * 0.92, end - 0.001),
                    (barrel * 0.86, end + 0.002),
                    (r * 0.985, end + 0.0028),
                    (r * 0.985, end - 0.001),
                ],
                16,
                Transform::IDENTITY,
            );
        }
    }

    pub(super) fn disc_foot(&self, disc: DiscFoot, g: &mut Geometry, radius: f32) {
        let h = self.hub_height_m;
        let neck = self.barrel_radius_m * 1.12;
        g.lathe(
            &[
                (0., 0.),
                (radius * 0.96, 0.),
                (radius, disc.rim_height_m * 0.32),
                (radius * 0.985, disc.rim_height_m),
                (
                    radius * disc.shoulder_fraction,
                    h * disc.shoulder_height_fraction,
                ),
                (neck, h - 0.039),
                (neck, h - 0.017),
                (0., h - 0.017),
            ],
            32,
            Transform::IDENTITY,
        );
    }

    pub(super) fn spoke_root(&self, radial: Vec3) -> Vec3 {
        radial * (self.hub_radius() * 0.43) + Vec3::Y * self.hub_height_m
    }
}
