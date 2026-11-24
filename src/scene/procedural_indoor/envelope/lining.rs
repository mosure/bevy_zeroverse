//! Finish-side construction inside the structural wall section. The room-side
//! datum stays fixed so wall-mounted objects, glazing and clearances still fit.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, IndoorLayout},
    materials::{variants, Surface},
};
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LiningStock {
    Veneer,
    AcousticFabric,
    CeramicSlab,
    FiredBrick,
    ConcreteBlock,
}

impl LiningStock {
    pub(crate) fn surface(self) -> Surface {
        match self {
            Self::Veneer => Surface::Wood,
            Self::AcousticFabric => Surface::FabricAlt,
            Self::CeramicSlab => Surface::Ceramic,
            Self::FiredBrick => Surface::Terracotta,
            Self::ConcreteBlock => Surface::Concrete,
        }
    }

    pub(crate) fn is_masonry(self) -> bool {
        matches!(self, Self::FiredBrick | Self::ConcreteBlock)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WallCourses {
    pub height_m: f32,
    /// Height of the first horizontal joint as a fraction of the module.
    pub phase: f32,
    /// Displacement in module widths. Panels alternate; masonry can also
    /// advance by a signed third on each successive course.
    pub bond_offset: f32,
}

impl WallCourses {
    pub(crate) fn phase_count(self, stock: LiningStock) -> usize {
        if self.bond_offset == 0. {
            1
        } else if stock.is_masonry() && self.bond_offset.abs() < 0.5 {
            3
        } else {
            2
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WallLining {
    pub stock: LiningStock,
    pub height_m: f32,
    pub module_width_m: f32,
    /// Fraction of a stock module at the wall origin.
    pub phase: f32,
    pub joint_width_m: f32,
    pub joint_depth_m: f32,
    pub finish_slot: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub courses: Option<WallCourses>,
}

impl WallLining {
    pub(crate) fn joint_bottom_fraction(&self) -> f32 {
        match self.stock {
            // Wrapped edges turn through the relief. A wide open joint exposes
            // more of the dark substrate than a narrow felt groove.
            LiningStock::AcousticFabric => {
                (1. - 1.5 * self.joint_depth_m / self.joint_width_m).clamp(0.16, 0.82)
            }
            LiningStock::FiredBrick | LiningStock::ConcreteBlock => 0.66,
            _ => 0.44,
        }
    }

    pub(crate) fn installed_height(&self, left_ceiling: f32, right_ceiling: f32) -> f32 {
        self.height_m.min(left_ceiling.min(right_ceiling))
    }

    pub(super) fn valid(&self) -> bool {
        let masonry = self.stock.is_masonry();
        let dimensions = match self.stock {
            LiningStock::FiredBrick => (0.19..=0.46, 0.06..=0.115),
            LiningStock::ConcreteBlock => (0.29..=0.52, 0.14..=0.25),
            _ => (0.24..=1.60, 0.14..=1.20),
        };
        (0.55..=if masonry { 6.8 } else { 2.25 }).contains(&self.height_m)
            && dimensions.0.contains(&self.module_width_m)
            && (0.0..1.0).contains(&self.phase)
            && (0.002..=0.02).contains(&self.joint_width_m)
            && (0.001..=0.012).contains(&self.joint_depth_m)
            && self.finish_slot < variants::COUNT
            && (!masonry || self.courses.is_some())
            && self.courses.is_none_or(|c| {
                dimensions.1.contains(&c.height_m)
                    && (0.0..1.0).contains(&c.phase)
                    && if masonry {
                        [0., 0.5, 1. / 3., -1. / 3.].contains(&c.bond_offset)
                    } else {
                        (0.0..=0.5).contains(&c.bond_offset)
                    }
            })
    }
}

fn sample_modules(lining: &mut WallLining, seed: u64) {
    use LiningStock::*;
    let mut r = stream(seed, 0x46494e495348524f);
    let installed = match lining.stock {
        CeramicSlab => 0.84,
        AcousticFabric => 0.66,
        Veneer => 0.32,
        FiredBrick | ConcreteBlock => return,
    };
    if !r.random_bool(installed) {
        return;
    }
    let (height, width, stagger) = match lining.stock {
        CeramicSlab if r.random_bool(0.48) => (0.14..0.32, 0.24..0.58, 0.74),
        CeramicSlab => (0.32..0.68, 0.52..1.22, 0.46),
        AcousticFabric => (0.44..1.12, 0.48..1.36, 0.19),
        Veneer => (0.32..0.72, 0.70..1.56, 0.62),
        FiredBrick | ConcreteBlock => return,
    };
    lining.module_width_m = r.random_range(width);
    lining.courses = Some(WallCourses {
        height_m: r.random_range(height),
        phase: r.random(),
        bond_offset: if r.random_bool(stagger) {
            if r.random_bool(0.57) {
                0.5
            } else {
                r.random_range(0.22..0.44)
            }
        } else {
            0.
        },
    });
}

// Retrofit and exposed structural finishes are more common in conversions,
// public circulation and workshops. A kitchen can still contain an accent wall.
fn sample_masonry(scene: &IndoorManifest) -> Option<WallLining> {
    use IndoorLayout::*;
    let mut r = stream(scene.seed, u64::from_le_bytes(*b"WallKiln"));
    let (chance, clay) = match scene.layout {
        Cafe | Gallery | Studio | Coworking => (0.48, 0.78),
        Workshop | ElevatorLanding | LectureHall | Classroom => (0.46, 0.25),
        Kitchen | Kitchenette | Breakroom => (0.10, 0.67),
        _ => (0.22, 0.52),
    };
    if !r.random_bool(chance) {
        return None;
    }
    let fired = r.random_bool(clay);
    let (height, width) = if fired && r.random_bool(0.24) {
        (r.random_range(0.062..0.087), r.random_range(0.30..0.453))
    } else if fired {
        let h = r.random_range(0.067_f32..0.109);
        (h, (h * r.random_range(2.2..3.1)).clamp(0.195, 0.335))
    } else {
        let h = r.random_range(0.151_f32..0.244);
        (h, (h * r.random_range(1.68..2.30)).clamp(0.295, 0.515))
    };
    let offset = if r.random_bool(if fired { 0.10 } else { 0.27 }) {
        0.
    } else if r.random_bool(if fired { 0.41 } else { 0.16 }) {
        if r.random_bool(0.5) {
            1. / 3.
        } else {
            -1. / 3.
        }
    } else {
        0.5
    };
    let joint = r.random_range(0.008_f32..0.016);
    Some(WallLining {
        stock: if fired {
            LiningStock::FiredBrick
        } else {
            LiningStock::ConcreteBlock
        },
        height_m: if r.random_bool(0.23) {
            r.random_range(1.05..1.72)
        } else {
            (scene.room_size.y * r.random_range(0.74..1.12)).clamp(1.8, 6.8)
        },
        module_width_m: width,
        phase: 0.,
        joint_width_m: joint,
        joint_depth_m: joint * r.random_range(0.19..0.44),
        finish_slot: r.random_range(0..variants::COUNT),
        courses: Some(WallCourses {
            height_m: height,
            phase: r.random(),
            bond_offset: offset,
        }),
    })
}

pub(super) fn install(e: &mut EnvelopeProgram, scene: &IndoorManifest) {
    use IndoorLayout::*;
    use LiningStock::*;
    let mut r = stream(scene.seed, 0x57414c4c4c4f5453);
    if !r.random_bool(0.58) {
        return;
    }
    let veneer = (Veneer, 0.74..1.68, 0.36..1.22, 0.004..0.013, 0.002..0.008);
    let fabric = (
        AcousticFabric,
        1.06..2.15,
        0.44..1.36,
        0.005..0.018,
        0.003..0.010,
    );
    let slab = (
        CeramicSlab,
        0.82..1.72,
        0.43..1.54,
        0.003..0.009,
        0.0015..0.004,
    );
    let (stock, height, pitch, width, depth) = match (scene.layout, r.random_range(0..10)) {
        (Kitchen | Kitchenette | Cafeteria | Breakroom, 0..=7) => slab,
        (Training | LectureHall | Classroom | OpenOffice, 0..=6) => fabric,
        (Lounge | Conference | Reception | Library, 0..=6) => veneer,
        (_, 0..=3) => fabric,
        (_, 4..=6) => slab,
        _ => veneer,
    };
    let mut series = WallLining {
        stock,
        height_m: r.random_range(height),
        module_width_m: r.random_range(pitch),
        phase: 0.,
        joint_width_m: r.random_range(width),
        joint_depth_m: r.random_range(depth),
        finish_slot: r.random_range(0..variants::COUNT),
        courses: None,
    };
    sample_modules(&mut series, scene.seed);
    let series = sample_masonry(scene).unwrap_or(series);
    let mut candidates = Vec::new();
    for (index, wall) in e.walls.iter().enumerate() {
        let a = e.footprint[wall.edge];
        let b = e.footprint[(wall.edge + 1) % e.footprint.len()];
        let span = a.distance(b);
        let height = if series.stock.is_masonry() {
            series.installed_height(
                e.ceiling_height(scene.room_size, a),
                e.ceiling_height(scene.room_size, b),
            )
        } else {
            series.height_m
        };
        let area = span * height;
        let covered = wall.facade.as_ref().map_or(0., |f| {
            f.openings
                .iter()
                .map(|o| (o.max.x - o.min.x) * (o.max.y.min(height) - o.min.y.max(0.)).max(0.))
                .sum::<f32>()
        });
        let solid = area - covered;
        if span > 2. && solid > area * 0.38 && solid > 2.2 {
            candidates.push((index, solid.sqrt() * r.random_range(0.22_f32..1.)));
        }
    }
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let count = if r.random_bool(0.35) { 2 } else { 1 };
    let mut return_origin = None;
    if series.stock.is_masonry() && count == 2 && candidates.len() >= 2 {
        let n = e.footprint.len();
        let anchor = e.walls[candidates[0].0].edge;
        let next = (anchor + 1) % n;
        let prev = (anchor + n - 1) % n;
        // Some two-wall installations continue around a corner. Independent
        // feature walls can still occur on opposite sides of a room.
        if stream(scene.seed, u64::from_le_bytes(*b"BondTurn")).random_bool(0.71) {
            if let Some(i) = candidates
                .iter()
                .position(|&(i, _)| [next, prev].contains(&e.walls[i].edge))
            {
                candidates.swap(1, i);
            }
        }
        let second = e.walls[candidates[1].0].edge;
        if second == next || second == prev {
            let length = |i: usize| e.footprint[i].distance(e.footprint[(i + 1) % n]);
            let signed_distance =
                (length(anchor) + length(second)) * if second == next { -0.5 } else { 0.5 };
            return_origin = Some(signed_distance / series.module_width_m);
        }
    }
    let first_phase: f32 = r.random();
    for (order, (i, _)) in candidates.into_iter().take(count).enumerate() {
        let phase = if order == 0 {
            first_phase
        } else if let Some(delta) = return_origin {
            (first_phase + delta).rem_euclid(1.)
        } else {
            r.random()
        };
        e.walls[i].lining = Some(WallLining { phase, ..series });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archived_plain_walls_replay_and_invalid_finish_geometry_is_rejected() {
        let plain: EnvelopeWall = serde_json::from_str(r#"{"edge":2,"facade":null}"#).unwrap();
        assert!(plain.lining.is_none());
        assert!(serde_json::to_value(plain).unwrap().get("lining").is_none());
        let mut sample = WallLining {
            stock: LiningStock::CeramicSlab,
            height_m: 1.25,
            module_width_m: 0.8,
            phase: 0.28,
            joint_width_m: 0.008,
            joint_depth_m: 0.003,
            finish_slot: 1,
            courses: None,
        };
        assert!(sample.valid());
        let archive = serde_json::to_value(sample).unwrap();
        assert!(archive.get("courses").is_none());
        assert_eq!(
            serde_json::from_value::<WallLining>(archive).unwrap(),
            sample
        );
        sample.module_width_m = 0.;
        assert!(!sample.valid());
        sample.module_width_m = 0.8;
        sample.joint_depth_m = f32::NAN;
        assert!(!sample.valid());
    }

    #[test]
    fn each_stock_admits_upright_stacked_and_staggered_installations() {
        for stock in [
            LiningStock::AcousticFabric,
            LiningStock::CeramicSlab,
            LiningStock::Veneer,
        ] {
            let mut counts = [0_u32; 3];
            for seed in 0..192 {
                let mut s = WallLining {
                    stock,
                    height_m: 1.48,
                    module_width_m: 0.72,
                    phase: 0.13,
                    joint_width_m: 0.008,
                    joint_depth_m: 0.004,
                    finish_slot: 0,
                    courses: None,
                };
                sample_modules(&mut s, seed);
                assert!(s.valid());
                counts[match s.courses {
                    None => 0,
                    Some(c) if c.bond_offset == 0. => 1,
                    Some(_) => 2,
                }] += 1;
                if let Some(c) = &mut s.courses {
                    c.height_m = f32::NAN;
                    assert!(!s.valid());
                }
            }
            assert!(counts.iter().all(|&v| v >= 7), "{stock:?} {counts:?}");
        }
    }

    #[test]
    fn mineral_stock_replays_measured_units_and_each_bond_direction() {
        let mut scene =
            IndoorManifest::generate_with_humans(13, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let mut counts = [[0_u32; 4]; 2];
        let mut tall = [false; 2];
        for seed in 0..2048 {
            scene.seed = seed;
            scene.layout = [IndoorLayout::Workshop, IndoorLayout::Cafe][seed as usize % 2];
            let Some(mut wall) = sample_masonry(&scene) else {
                continue;
            };
            assert_eq!(Some(wall), sample_masonry(&scene));
            assert!(wall.valid());
            let stock = usize::from(wall.stock == LiningStock::FiredBrick);
            let courses = wall.courses.unwrap();
            let phases = courses.phase_count(wall.stock);
            counts[stock][if courses.bond_offset < 0. {
                3
            } else {
                phases - 1
            }] += 1;
            tall[usize::from(wall.height_m > 1.73)] = true;
            let json = serde_json::to_vec(&wall).unwrap();
            assert_eq!(serde_json::from_slice::<WallLining>(&json).unwrap(), wall);
            wall.courses = None;
            assert!(!wall.valid());
        }
        assert!(counts.iter().flatten().all(|&n| n >= 5), "{counts:?}");
        assert_eq!(tall, [true; 2]);
    }

    #[test]
    fn adjacent_masonry_uses_the_same_course_and_wraps_the_pitch_at_the_corner() {
        let mut s =
            IndoorManifest::generate_with_humans(13, IndoorLayout::Cafe, 0.65, 0, 0.).unwrap();
        let mut e = s.envelope.clone().unwrap();
        e.footprint = vec![
            Vec2::new(-4., -3.),
            Vec2::new(4., -3.),
            Vec2::new(4., 3.),
            Vec2::new(-4., 3.),
        ];
        e.walls = (0..4)
            .map(|edge| EnvelopeWall {
                edge,
                facade: None,
                lining: None,
            })
            .collect();
        e.ceiling_drop = Vec2::ZERO;
        e.soffits.clear();
        let n = e.footprint.len();
        let mut turns = 0;
        for seed in 0..400 {
            s.seed = seed;
            e.walls.iter_mut().for_each(|w| w.lining = None);
            install(&mut e, &s);
            for w in &e.walls {
                let Some(first) = w.lining.filter(|v| v.stock.is_masonry()) else {
                    continue;
                };
                let j = (w.edge + 1) % n;
                let Some(second) = e.walls.iter().find(|a| a.edge == j).and_then(|v| v.lining)
                else {
                    continue;
                };
                assert_eq!(first.courses, second.courses);
                assert_eq!(first.finish_slot, second.finish_slot);
                let length = |i: usize| e.footprint[i].distance(e.footprint[(i + 1) % n]);
                let delta = (length(w.edge) + length(j)) * 0.5 / first.module_width_m
                    + second.phase
                    - first.phase;
                assert!((delta - delta.round()).abs() < 1e-4);
                turns += 1;
            }
        }
        assert!(turns >= 8, "{turns}");
    }
}
