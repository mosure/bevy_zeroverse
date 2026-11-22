//! Metric floor plan on the far side of the shared entry wall. Coordinates are
//! room-local X/Z metres. The primary reconstruction region ends at the entry.
use super::*;
mod elevation;
pub use elevation::{AdjoiningElevation, AdjoiningWindows};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdjoiningFloor {
    #[default]
    Continuous,
    Concrete,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AdjoiningReturn {
    pub split_x_m: f32,
    pub positive_x: bool,
    pub extra_depth_m: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AdjoiningSpace {
    /// Clear depth across the entire shared wall. A side return can extend farther.
    pub depth_m: f32,
    pub side_return: Option<AdjoiningReturn>,
    pub floor: AdjoiningFloor,
}

impl Default for AdjoiningSpace {
    fn default() -> Self {
        Self {
            depth_m: NEIGHBOR_DEPTH,
            side_return: None,
            floor: AdjoiningFloor::Continuous,
        }
    }
}

impl AdjoiningSpace {
    pub(super) fn sample(seed: u64, width: f32) -> Self {
        let mut r = stream(seed, 0x4241_5946_4c4f_4f52);
        // Compact circulation and deeper occupied space share an overlapping metric prior.
        let depth_m = if r.random_bool(0.44) {
            r.random_range(2.30..3.60)
        } else {
            r.random_range(2.95..5.35)
        };
        let side_return = r.random_bool(0.46).then(|| AdjoiningReturn {
            split_x_m: width * r.random_range(-0.22..0.22),
            positive_x: r.random(),
            extra_depth_m: r.random_range(0.80..2.55),
        });
        let floor = if stream(seed, 0x4241_5946_494e_4953).random_bool(0.29) {
            AdjoiningFloor::Concrete
        } else {
            AdjoiningFloor::Continuous
        };
        Self {
            depth_m,
            side_return,
            floor,
        }
    }

    pub(crate) fn validate(self, width: f32) -> Result<(), String> {
        if !(2.0..=6.0).contains(&self.depth_m)
            || self.side_return.is_some_and(|r| {
                !r.split_x_m.is_finite()
                    || r.split_x_m.abs() + 0.8 >= width * 0.5
                    || !(0.5..=3.0).contains(&r.extra_depth_m)
            })
        {
            return Err("invalid adjoining floor dimensions".into());
        }
        Ok(())
    }

    pub fn max_depth(self) -> f32 {
        self.depth_m + self.side_return.map_or(0., |r| r.extra_depth_m)
    }

    pub fn area(self, width: f32) -> f32 {
        width * self.depth_m
            + self.side_return.map_or(0., |r| {
                let return_width = width * 0.5
                    + if r.positive_x {
                        -r.split_x_m
                    } else {
                        r.split_x_m
                    };
                return_width * r.extra_depth_m
            })
    }

    /// Shortest available depth across a closed X interval. Straddling the
    /// inner corner requires clearing the nearer wall.
    pub fn depth_over(self, left: f32, right: f32) -> f32 {
        if self.side_return.is_some_and(|r| {
            if r.positive_x {
                left >= r.split_x_m
            } else {
                right <= r.split_x_m
            }
        }) {
            self.max_depth()
        } else {
            self.depth_m
        }
    }

    pub fn contains_bounds(self, size: Vec3, lo: Vec2, hi: Vec2, margin: f32) -> bool {
        let lo = lo - Vec2::splat(margin);
        let hi = hi + Vec2::splat(margin);
        let half = size.xz() * 0.5;
        lo.x >= -half.x
            && hi.x <= half.x
            && lo.y >= half.y
            && hi.y <= half.y + self.depth_over(lo.x, hi.x)
    }

    /// Counterclockwise polygon, starting along the shared wall. Remaining
    /// edges form the exterior walls in their construction order.
    pub fn footprint(self, size: Vec3) -> Vec<Vec2> {
        let h = size.xz() * 0.5;
        let far_z = h.y + self.max_depth();
        let near_z = h.y + self.depth_m;
        let mut p = Vec::with_capacity(6);
        p.extend([Vec2::new(-h.x, h.y), Vec2::new(h.x, h.y)]);
        if let Some(r) = self.side_return {
            let (right_z, left_z) = if r.positive_x {
                (far_z, near_z)
            } else {
                (near_z, far_z)
            };
            p.extend([
                Vec2::new(h.x, right_z),
                Vec2::new(r.split_x_m, right_z),
                Vec2::new(r.split_x_m, left_z),
                Vec2::new(-h.x, left_z),
            ]);
        } else {
            p.extend([Vec2::new(h.x, far_z), Vec2::new(-h.x, far_z)]);
        }
        p
    }

    /// Outside corner cut out of the bounding rectangle. A swept path cannot
    /// take a diagonal shortcut through this corner.
    pub fn recess(self, size: Vec3) -> Option<(Vec2, Vec2)> {
        let r = self.side_return?;
        let h = size.xz() * 0.5;
        let (x0, x1) = if r.positive_x {
            (-h.x, r.split_x_m)
        } else {
            (r.split_x_m, h.x)
        };
        Some((
            Vec2::new(x0, h.y + self.depth_m),
            Vec2::new(x1, h.y + self.max_depth()),
        ))
    }
}

impl IndoorManifest {
    /// Legacy envelopes with no archived adjoining plan used a full-width 3.2 m bay.
    pub fn adjoining_space(&self) -> AdjoiningSpace {
        self.adjoining
            .filter(|_| self.envelope.is_some())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::envelope::polygon;

    #[test]
    fn footprint_queries_match_independent_polygon_and_replay() {
        let mut depths = std::collections::BTreeSet::new();
        let mut returns = [0; 2];
        let mut concrete = 0;
        for seed in 0..256 {
            let width = 5.8 + (seed % 57) as f32 * 0.18;
            let size = Vec3::new(width, 3., 10.);
            let bay = AdjoiningSpace::sample(seed, width);
            bay.validate(width).unwrap();
            assert_eq!(bay, AdjoiningSpace::sample(seed, width));
            let polygon = bay.footprint(size);
            assert!((polygon::area(&polygon) - bay.area(width)).abs() < 0.0001);
            depths.insert((bay.depth_m * 100.).round() as u32);
            if let Some(r) = bay.side_return {
                returns[usize::from(r.positive_x)] += 1;
            }
            concrete += usize::from(bay.floor == AdjoiningFloor::Concrete);
            let mut rng = stream(seed, 9304);
            for _ in 0..96 {
                let center = Vec2::new(
                    rng.random_range(-0.55..0.55) * width,
                    5. + rng.random_range(-0.5..bay.max_depth() + 0.5),
                );
                let h = Vec2::new(rng.random_range(0.1..0.8), rng.random_range(0.1..0.8));
                let lo = center - h;
                let hi = center + h;
                assert_eq!(
                    bay.contains_bounds(size, lo, hi, 0.),
                    polygon::box_inside(&polygon, lo, hi, 0.)
                );
            }
        }
        assert!(
            depths.len() > 145 && returns.iter().all(|v| *v > 35) && (45..105).contains(&concrete)
        );
    }

    #[test]
    fn archived_and_invalid_plans_have_explicit_behavior() {
        let room = IndoorManifest::generate(86, IndoorLayout::Mixed, 0.6, 0).unwrap();
        let mut archived = serde_json::to_value(room).unwrap();
        let fields = archived.as_object_mut().unwrap();
        fields.remove("adjoining");
        fields.remove("adjoining_elevation");
        let replay: IndoorManifest = serde_json::from_value(archived).unwrap();
        assert_eq!(replay.adjoining_space(), AdjoiningSpace::default());
        assert_eq!(replay.adjoining_space().depth_m, 3.2);
        assert!(replay.adjoining_elevation.is_none());
        let damaged = AdjoiningSpace {
            side_return: Some(AdjoiningReturn {
                split_x_m: f32::NAN,
                positive_x: false,
                extra_depth_m: 1.,
            }),
            ..Default::default()
        };
        assert!(damaged.validate(8.).is_err());
    }

    #[test]
    fn adjoining_windows_replay_in_their_recorded_wall_and_reject_invalid_installations() {
        let room = IndoorManifest::generate(44316, IndoorLayout::Mixed, 0.65, 0).unwrap();
        let sampled = AdjoiningElevation::sample(&room);
        assert_eq!(Some(&sampled), room.adjoining_elevation.as_ref());
        let encoded = serde_json::to_vec(&room).unwrap();
        let replay: IndoorManifest = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(replay.adjoining_elevation.as_ref(), Some(&sampled));
        let mut window = sampled.windows.unwrap();
        window.validate(&replay).unwrap();
        let installed = window.edge;
        for invalid in [0, usize::MAX] {
            window.edge = invalid;
            assert!(window.validate(&replay).is_err());
        }
        window.edge = installed;
        window.facade.recess += 1.;
        assert!(window.validate(&replay).is_err());
    }
}
