//! One installed lifting station. Local X follows the bar, and lifters enter from +Z.
//! Collisions reserve the entire station. Camera proposals see its open frame.
use super::*;
use crate::scene::procedural_indoor::materials::strength::Finish;

mod mesh;
pub(super) use mesh::build;
#[cfg(test)]
mod tests;

pub(crate) const VERSION: u32 = 292;
/// Clear floor past both sleeve ends, behind the frame, and in front of the lifter.
pub(crate) const ACCESS: (f32, f32, f32) = (0.72, 0.24, 0.91);
const INNER_SLEEVE: f32 = 0.655;
const BORE: f32 = 0.02525;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PlateStock {
    CastIron,
    BlackBumper,
    ColourBumper,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct PowerRack {
    pub full_cage: bool,
    pub size: Vec3,
    pub upright_span: f32,
    pub post: f32,
    pub pitch: f32,
    pub first_hole: f32,
    pub holes: usize,
    pub cup_pin: usize,
    pub safety_pin: usize,
    pub plate_stock: PlateStock,
    /// Pairs ordered from the sleeve flange out. Zero denotes an empty position.
    pub plate_pairs_kg: [u8; 3],
}

pub(super) fn dimensions(r: &mut impl Rng) -> Vec3 {
    Vec3::new(
        if r.random_bool(0.69) { 2.20 } else { 2.01 },
        r.random_range(2.04..2.42),
        r.random_range(1.18..1.69),
    )
}

impl PowerRack {
    pub fn for_object(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"RackFit "));
        let mut load = stream(o.seed, u64::from_le_bytes(*b"BarLoads"));
        let mut kit = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"GymDiscs"),
        );
        let pitch = r.random_range(0.058..0.078);
        let first_hole = r.random_range(0.51..0.61);
        let pin = |height: f32| ((height - first_hole) / pitch).round() as usize;
        let cup_pin = pin(if r.random_bool(0.27) {
            r.random_range(0.90..1.14)
        } else {
            r.random_range(1.23..1.65)
        });
        let safety_pin = pin(first_hole + cup_pin as f32 * pitch - r.random_range(0.27..0.55));
        let mut plates = [0; 3];
        let count = match load.random_range(0..100) {
            0..=24 => 0,
            25..=63 => 1,
            64..=89 => 2,
            _ => 3,
        };
        for plate in &mut plates[..count] {
            *plate = [5, 10, 15, 20, 20, 25][load.random_range(0..6)];
        }
        plates[..count].sort_unstable_by(|a, b| b.cmp(a));
        Self {
            full_cage: r.random_bool(0.52),
            size: o.size,
            upright_span: r.random_range(1.035..1.165),
            post: r.random_range(0.062..0.080),
            pitch,
            first_hole,
            holes: pin((o.size.y - 0.23).min(1.87)) + 1,
            cup_pin,
            safety_pin,
            plate_stock: match kit.random_range(0..10) {
                0..=2 => PlateStock::CastIron,
                3..=6 => PlateStock::BlackBumper,
                _ => PlateStock::ColourBumper,
            },
            plate_pairs_kg: plates,
        }
    }
    fn work_z(self) -> f32 {
        self.size.z * if self.full_cage { -0.33 } else { -0.105 }
    }
    fn opposite_z(self) -> f32 {
        self.size.z * if self.full_cage { 0.34 } else { -0.38 }
    }
    fn guard_end_z(self) -> f32 {
        if self.full_cage {
            self.opposite_z()
        } else {
            self.size.z * 0.40
        }
    }
    fn pin_y(self, i: usize) -> f32 {
        self.first_hole + i as f32 * self.pitch
    }
    fn shaft_radius(self) -> f32 {
        if self.size.x < 2.1 {
            0.0125
        } else {
            0.014
        }
    }
    /// Centre of the bar where its shaft rests on both lined J-cups.
    pub fn bar_center(self) -> Vec3 {
        Vec3::new(
            0.,
            self.pin_y(self.cup_pin) + 0.030 + self.shaft_radius(),
            self.work_z() + self.post * 0.5 + 0.064,
        )
    }
    fn plate(self, kg: u8) -> (f32, f32, Finish) {
        let iron = self.plate_stock == PlateStock::CastIron;
        let finish = match self.plate_stock {
            PlateStock::CastIron => Finish::Iron,
            PlateStock::BlackBumper => Finish::Rubber,
            PlateStock::ColourBumper => match kg {
                10 => Finish::Green,
                15 => Finish::Yellow,
                20 => Finish::Blue,
                25 => Finish::Red,
                _ => Finish::Rubber,
            },
        };
        let radius = match (kg, iron) {
            (5, _) => 0.116,
            (10, true) => 0.154,
            (15, true) => 0.184,
            (20, true) => 0.210,
            _ => 0.225,
        };
        let width = if iron {
            0.014 + f32::from(kg) * 0.00125
        } else {
            0.012 + f32::from(kg) * 0.00275
        };
        (radius, width, finish)
    }
    fn discs(self, mut add: impl FnMut(f32, f32, f32, Finish)) -> f32 {
        let mut edge = INNER_SLEEVE + 0.030;
        for kg in self.plate_pairs_kg.into_iter().take_while(|&kg| kg != 0) {
            let (radius, width, stock) = self.plate(kg);
            add(edge + width * 0.5, radius, width, stock);
            edge += width;
        }
        edge
    }
    /// Shared axis-aligned frame members. The two perforated working posts are built separately.
    fn frame(self, mut add: impl FnMut(Vec3, Vec3)) {
        let (s, w, work, other, guard) = (
            self.size,
            self.post,
            self.work_z(),
            self.opposite_z(),
            self.guard_end_z(),
        );
        for side in [-1., 1.] {
            let x = side * self.upright_span * 0.5;
            add(Vec3::new(x, 0.033, 0.), Vec3::new(w + 0.055, 0.066, s.z));
            let top = if self.full_cage { s.y } else { s.y * 0.52 };
            add(
                Vec3::new(x, (top + 0.066) * 0.5, other),
                Vec3::new(w, top - 0.066, w),
            );
            let join_z = (work + other) * 0.5;
            add(
                Vec3::new(x, top - 0.09, join_z),
                Vec3::new(w * 0.82, w, (work - other).abs() - w),
            );
            let start = work + w * 0.5;
            let end = guard - if self.full_cage { w * 0.5 } else { 0. };
            add(
                Vec3::new(x, self.pin_y(self.safety_pin), (start + end) * 0.5),
                Vec3::new(w * 0.83, 0.058, end - start),
            );
        }
        // A high rear tie on a cage; a low rear stabilizer on a half rack.
        let z = if self.full_cage { work } else { other };
        let y = if self.full_cage { s.y - 0.10 } else { 0.13 };
        add(
            Vec3::new(0., y, z),
            Vec3::new(self.upright_span - w, w, w * 0.86),
        );
    }
    pub(crate) fn view_boxes(self, mut emit: impl FnMut(Vec3, Vec3)) {
        self.frame(|p, s| emit(p, s * 0.5));
        let base = 0.066;
        let top = self.size.y;
        for side in [-1., 1.] {
            emit(
                Vec3::new(
                    side * self.upright_span * 0.5,
                    (top + base) * 0.5,
                    self.work_z(),
                ),
                Vec3::new(self.post, top - base, self.post) * 0.5,
            );
        }
        let cross = if self.full_cage {
            self.opposite_z()
        } else {
            self.work_z()
        };
        emit(
            Vec3::new(0., top - 0.12, cross),
            Vec3::new(self.upright_span * 0.5, 0.017, 0.017),
        );
        let bar = self.bar_center();
        let rr = self.shaft_radius();
        emit(bar, Vec3::new(INNER_SLEEVE, rr, rr));
        let end = self.size.x * 0.5;
        for sign in [-1., 1.] {
            emit(
                bar + Vec3::X * (sign * (INNER_SLEEVE + end) * 0.5),
                Vec3::new((end - INNER_SLEEVE) * 0.5, 0.025, 0.025),
            );
        }
        self.discs(|x, r, w, _| {
            for side in [-1., 1.] {
                // Keep the square corners around a disc transparent to alignment rays.
                for (height, half_depth) in [(-0.70, 0.72), (0., 1.), (0.70, 0.72)] {
                    let half_height = if height == 0. { 0.4 } else { 0.3 };
                    emit(
                        bar + Vec3::new(side * x, height * r, 0.),
                        Vec3::new(w * 0.5, half_height * r, half_depth * r),
                    );
                }
            }
        });
    }
}
