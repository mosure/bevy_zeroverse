//! Vision openings in the wall between the room and its adjoining circulation.
//! Intervals run away from the entry jamb on each side, so a sidelight stays
//! attached to its own door even when the entrance changes position.
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SharedOpenings {
    /// Fractions of the available wall from each jamb towards its corner.
    pub jamb_intervals: [Vec2; 2],
    /// Vertical interval within the clear height above the sill and below the header.
    pub height_interval: Vec2,
}

impl SharedOpenings {
    pub(super) fn sample(seed: u64) -> Self {
        let mut r = stream(seed, 0x5649_5349_4f4e_5741);
        let design = r.random::<f32>();
        let height_interval = if design < 0.20 {
            Vec2::new(r.random_range(0.42..0.74), r.random_range(0.9..1.0))
        } else {
            Vec2::new(r.random_range(0.0..0.14), r.random_range(0.78..1.0))
        };
        let mut jamb_intervals = [Vec2::ZERO; 2];
        let primary = usize::from(r.random::<bool>());
        for (i, interval) in jamb_intervals.iter_mut().enumerate() {
            let width = match design {
                // Private walls can have only the entry itself; other installations
                // range from one vision panel to nearly continuous office glazing.
                d if d > 0.93 || (d > 0.64 && i != primary) => continue,
                d if d > 0.64 => r.random_range(0.20..0.72),
                d if d > 0.30 => r.random_range(0.67..1.0),
                _ => r.random_range(0.35..0.96),
            };
            let offset = if design > 0.64 {
                (1. - width) * r.random_range(0.0..0.25)
            } else {
                (1. - width) * r.random::<f32>()
            };
            *interval = Vec2::new(offset, offset + width);
        }
        Self {
            jamb_intervals,
            height_interval,
        }
    }

    pub(super) fn valid(self) -> bool {
        let bounded = |v: Vec2| v.is_finite() && v.x >= 0. && v.y <= 1. && v.x <= v.y;
        self.jamb_intervals.into_iter().all(bounded)
            && bounded(self.height_interval)
            && self.height_interval.y > self.height_interval.x
    }
}
