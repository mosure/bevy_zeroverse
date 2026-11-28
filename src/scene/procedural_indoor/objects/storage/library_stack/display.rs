//! Merchandising one title within a double-sided stack. A featured cover claims
//! its own width on the deck before the neighbouring spine-out run is packed.
use super::{stock, *};

mod rests;
#[cfg(test)]
mod tests;

pub(super) const INTRODUCED: u32 = 321;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum BookRest {
    BentWire,
    GroovedTimber,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct FeaturedBook {
    pub bay: usize,
    pub level: usize,
    /// +1 and -1 are the two aisles on the stack's local Z axis.
    pub side: i8,
    pub rest: BookRest,
    pub edition: u64,
    /// Centre of the bottom edge of the back cover, in stack-local metres.
    pub back_foot_m: Vec3,
    pub cover_width_m: f32,
    pub cover_height_m: f32,
    pub thickness_m: f32,
    pub lift_m: f32,
    /// Inclination back from vertical and turn relative to the shelf front.
    pub recline_rad: f32,
    pub turn_rad: f32,
}

#[derive(Clone, Copy)]
pub(super) struct Selection {
    chance: f64,
    rest: BookRest,
}

impl Selection {
    pub fn for_stack(o: &IndoorObject, version: u32) -> Option<Self> {
        if version < INTRODUCED {
            return None;
        }
        let mut r = stream(o.seed, u64::from_le_bytes(*b"LibFront"));
        r.random_bool(0.74).then(|| Self {
            chance: r.random_range(0.19..0.53),
            // Nearby stacks can share a purchased style of rest without stocking
            // the same titles or featuring the same bays.
            rest: if stream(
                o.finish_seed.unwrap_or(o.seed),
                u64::from_le_bytes(*b"LibEasel"),
            )
            .random_bool(0.56)
            {
                BookRest::BentWire
            } else {
                BookRest::GroovedTimber
            },
        })
    }

    pub fn fit(
        self,
        seed: u64,
        bay: usize,
        level: usize,
        cell: &mut stock::Cell,
    ) -> Option<FeaturedBook> {
        let mut r = stream(
            seed,
            u64::from_le_bytes(*b"LibTitle") + cell.index(bay, level),
        );
        let eye_level = cell.floor + cell.height * 0.5;
        let frequency = if (0.40..1.62).contains(&eye_level) {
            self.chance
        } else {
            0.28 * self.chance
        };
        if !r.random_bool(frequency) {
            return None;
        }
        let thickness = r.random_range(0.012..0.044);
        let lift = r.random_range(0.021..0.037);
        let recline =
            r.random_range(0.13_f32..0.32) + 0.095 * (1. - eye_level / 1.62).clamp(0., 1.);
        let (sin, cos) = recline.sin_cos();
        let height = r
            .random_range(0.178_f32..0.303)
            .min((cell.height - 0.013 - lift - thickness * sin) / cos);
        if height < 0.15 {
            return None;
        }
        let width = height * r.random_range(0.59..0.86);
        let turn: f32 = r.random_range(-0.18..0.18);
        let (st, ct) = turn.sin_cos();
        let front = thickness + 0.021;
        let rear = heel(height, sin).max(height * sin);
        // Enclose the *turned* book, lip and heel before choosing the side gap.
        let half_x = width * 0.5 * ct + st.abs() * rear.max(front);
        let front_z = front * ct + width * 0.5 * st.abs();
        let rear_z = rear * ct + width * 0.5 * st.abs();
        let slack = cell.depth - front_z - rear_z - 0.008;
        let edge = r.random_range(0.014..0.065);
        let end = edge + 2. * half_x + r.random_range(0.016..0.048);
        if slack < 0.006 || cell.width - end < 0.14 {
            return None;
        }
        let setback = r.random_range(0.005..slack.min(0.063));
        let left = r.random_bool(0.5);
        let x = if left {
            cell.left + edge + half_x
        } else {
            cell.left + cell.width - edge - half_x
        };
        if left {
            cell.left += end;
        }
        cell.width -= end;
        Some(FeaturedBook {
            bay,
            level,
            side: cell.side,
            rest: self.rest,
            edition: r.random(),
            back_foot_m: Vec3::new(
                x,
                cell.floor + lift,
                f32::from(cell.side) * (cell.outer - setback - front_z),
            ),
            cover_width_m: width,
            cover_height_m: height,
            thickness_m: thickness,
            lift_m: lift,
            recline_rad: recline,
            turn_rad: turn,
        })
    }
}

fn heel(height: f32, sin: f32) -> f32 {
    (height * sin * 0.80 + 0.022).max(0.068)
}

impl FeaturedBook {
    fn yaw(&self) -> Quat {
        Quat::from_rotation_y(if self.side < 0 {
            PI + self.turn_rad
        } else {
            self.turn_rad
        })
    }

    fn frame(&self) -> Transform {
        Transform::from_translation(self.back_foot_m)
            .with_rotation(self.yaw() * Quat::from_rotation_x(-self.recline_rad))
    }

    /// Transform of the existing closed book shell. The +X side of that shell
    /// is the front of the unfolded jacket and +Z is its bound spine.
    pub fn book_transform(&self) -> Transform {
        let frame = self.frame();
        Transform::from_translation(frame.transform_point(Vec3::new(
            0.,
            self.cover_height_m * 0.5,
            self.thickness_m * 0.5,
        )))
        .with_rotation(frame.rotation * Quat::from_rotation_y(-FRAC_PI_2))
    }

    pub(super) fn build(self, a: &mut Assembly, version: u32) {
        self.build_rest(a);
        stock::Book {
            edition: self.edition,
            size: Vec3::new(self.thickness_m, self.cover_height_m, self.cover_width_m),
            tf: self.book_transform(),
        }
        .build(a, version);
    }
}
