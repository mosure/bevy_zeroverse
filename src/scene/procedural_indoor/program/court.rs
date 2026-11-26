//! Indoor volleyball. Local X crosses the sidelines and local Z follows play.
//! Full-size installations use FIVB court, attack-line and centre net dimensions.
use super::*;

pub const LINE: f32 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VolleyballCourt {
    pub full_size: bool,
    pub long_x: bool,
    pub center: Vec2,
    pub width: f32,
    pub length: f32,
    pub attack_distance: f32,
    pub net_height: f32,
    pub post_outset: f32,
    pub pad_radius: f32,
    pub free_zone: f32,
}

fn design(seed: u64) -> (bool, f32, f32) {
    let mut r = stream(seed, u64::from_le_bytes(*b"VBDesign"));
    if r.random_bool(0.59) {
        (true, 9., 18.)
    } else {
        (
            false,
            r.random_range(6.35..8.70),
            r.random_range(12.5..17.25),
        )
    }
}
pub(crate) fn room_size(seed: u64) -> Vec3 {
    let (full, w, l) = design(seed);
    let mut r = stream(seed, u64::from_le_bytes(*b"VBShell "));
    let a = w + r.random_range(8.5..11.8);
    let b = l + r.random_range(7.9..10.4);
    let h = if full {
        r.random_range(9.3..10.9)
    } else {
        r.random_range(7.1..9.5)
    };
    if long_x(seed) {
        Vec3::new(b, h, a)
    } else {
        Vec3::new(a, h, b)
    }
}
fn long_x(seed: u64) -> bool {
    stream(seed, u64::from_le_bytes(*b"VBCourse")).random()
}
impl VolleyballCourt {
    pub(crate) fn fit(seed: u64, size: Vec3) -> Option<Self> {
        let (full_size, width, length) = design(seed);
        let mut r = stream(seed, u64::from_le_bytes(*b"VBLayout"));
        let free_zone = if full_size {
            3.
        } else {
            r.random_range(1.75..2.75)
        };
        let long_x = long_x(seed);
        let ext = if long_x {
            Vec2::new(length, width)
        } else {
            Vec2::new(width, length)
        };
        let slack = (size.xz() - ext) * 0.5 - Vec2::splat(free_zone + 0.55);
        if !size.is_finite() || slack.min_element() < 0. || size.y < 5.5 {
            return None;
        }
        Some(Self {
            full_size,
            long_x,
            width,
            length,
            free_zone,
            center: slack * Vec2::new(r.random_range(-0.83..0.83), r.random_range(-0.83..0.83)),
            attack_distance: if full_size {
                3.
            } else {
                length * r.random_range(0.15..0.18)
            },
            net_height: if full_size {
                if r.random_bool(0.5) {
                    2.43
                } else {
                    2.24
                }
            } else {
                r.random_range(1.99..2.40)
            },
            post_outset: r.random_range(0.76..0.91),
            pad_radius: r.random_range(0.088..0.143),
        })
    }
    /// Convert court coordinates (across, along) to the manifest's floor axes.
    pub fn world(self, local: Vec2) -> Vec2 {
        self.center
            + if self.long_x {
                Vec2::new(local.y, local.x)
            } else {
                local
            }
    }
    pub fn protected_bounds(self) -> (Vec2, Vec2) {
        let half = Vec2::new(self.width, self.length) * 0.5 + self.free_zone;
        (self.world(-half), self.world(half))
    }
    pub fn clear_height(self) -> f32 {
        if self.full_size {
            7.
        } else {
            5.5
        }
    }
    pub(crate) fn net_size(self) -> Vec3 {
        Vec3::new(
            self.width + 2. * (self.post_outset + self.pad_radius),
            self.net_height + 0.17,
            self.pad_radius * 2.,
        )
    }
    pub(crate) fn net_yaw(self) -> f32 {
        if self.long_x {
            std::f32::consts::FRAC_PI_2
        } else {
            0.
        }
    }
}

/// Absolute lower limit for a ceiling fitting over the court or its run-off.
pub(crate) fn clear_height_at(scene: &IndoorManifest, lo: Vec2, hi: Vec2) -> Option<f32> {
    let court = scene.program.as_ref()?.volleyball?;
    let (min, max) = court.protected_bounds();
    (lo.cmplt(max).all() && hi.cmpgt(min).all()).then(|| court.clear_height())
}
