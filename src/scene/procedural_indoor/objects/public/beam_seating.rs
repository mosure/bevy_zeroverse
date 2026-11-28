//! Floor-standing waiting seats carried on a common beam. The passenger faces local +Z.
use super::*;

mod shell;
#[cfg(test)]
mod tests;

const LABEL: &str = "bench";
pub(crate) const VERSION: u32 = 315;
const FITTED_BACK_VERSION: u32 = 327;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SeatStock {
    BentPlywood,
    MouldedPolymer,
    Upholstered,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Armrests {
    None,
    Ends,
    EachSeat,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct BeamSeating {
    pub bays: usize,
    pub table_bay: Option<usize>,
    pub pitch_m: f32,
    pub seat_height_m: f32,
    pub seat_width_m: f32,
    pub wrap_m: f32,
    pub shell_thickness_m: f32,
    pub armrests: Armrests,
    pub stock: SeatStock,
    pub facing: Surface,
    pub finish_slot: usize,
    pub splayed_feet: bool,
}

pub(super) fn dimensions(r: &mut impl Rng) -> Vec3 {
    let bays = match r.random_range(0..10) {
        0..=2 => 2,
        3..=6 => 3,
        7..=8 => 4,
        _ => 5,
    };
    Vec3::new(
        bays as f32 * r.random_range(0.565..0.650),
        r.random_range(0.77..0.985),
        r.random_range(0.575..0.755),
    )
}

impl BeamSeating {
    pub fn for_object(o: &IndoorObject) -> Self {
        let mut kit = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"WaitSeat"),
        );
        let mut bank = stream(o.seed, u64::from_le_bytes(*b"BeamBays"));
        let bays = (o.size.x / 0.61).round().clamp(2., 5.) as usize;
        let pitch_m = o.size.x / bays as f32;
        let stock = match kit.random_range(0..3) {
            0 => SeatStock::BentPlywood,
            1 => SeatStock::MouldedPolymer,
            _ => SeatStock::Upholstered,
        };
        Self {
            bays,
            table_bay: (bays > 2 && bank.random_bool(0.32)).then(|| bank.random_range(0..bays)),
            pitch_m,
            seat_height_m: kit.random_range(0.414..0.472),
            seat_width_m: pitch_m - kit.random_range(0.069..0.112),
            wrap_m: kit.random_range(0.018..0.045),
            shell_thickness_m: kit.random_range(0.009..0.015),
            armrests: match kit.random_range(0..6) {
                0 => Armrests::None,
                1..=2 => Armrests::Ends,
                _ => Armrests::EachSeat,
            },
            facing: match stock {
                SeatStock::BentPlywood => Surface::Wood,
                SeatStock::MouldedPolymer => Surface::Plastic,
                SeatStock::Upholstered if kit.random_bool(0.26) => Surface::Leather,
                SeatStock::Upholstered => Surface::Fabric,
            },
            stock,
            finish_slot: kit
                .random_range(0..crate::scene::procedural_indoor::materials::variants::COUNT),
            splayed_feet: kit.random_bool(0.55),
        }
    }

    pub fn seat_count(self) -> usize {
        self.bays - usize::from(self.table_bay.is_some())
    }

    pub(crate) fn center_x(self, bay: usize) -> f32 {
        (bay as f32 + 0.5 - self.bays as f32 * 0.5) * self.pitch_m
    }

    /// Arm at a joint between bays, including the end caps.
    pub(crate) fn arm_x(self, bank: &IndoorObject, joint: usize) -> Option<f32> {
        let end = joint == 0 || joint == self.bays;
        if self.armrests == Armrests::None
            || (self.armrests == Armrests::Ends && !end)
            || (joint == 0 && self.table_bay == Some(0))
            || (joint == self.bays && self.table_bay == Some(joint - 1))
        {
            return None;
        }
        Some(
            ((joint as f32 - self.bays as f32 * 0.5) * self.pitch_m)
                .clamp(-bank.size.x * 0.5 + 0.034, bank.size.x * 0.5 - 0.034),
        )
    }

    fn beam_y(self) -> f32 {
        self.seat_height_m - 0.137
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    let p = BeamSeating::for_object(o);
    let beam = p.beam_y();
    a.box_part(
        Surface::Metal,
        LABEL,
        Vec3::Y * beam,
        Vec3::new(o.size.x - 0.105, 0.083, 0.068),
        0.006,
    );
    let supports = if p.bays >= 5 { 3 } else { 2 };
    for i in 0..supports {
        let x = (i as f32 / (supports - 1) as f32 - 0.5) * (o.size.x - 0.70 * p.pitch_m);
        for side in [-1., 1.] {
            let z = side * o.size.z * 0.37;
            a.box_part(
                Surface::Rubber,
                LABEL,
                Vec3::new(x, 0.010, z),
                Vec3::new(0.07, 0.02, 0.085),
                0.004,
            );
            if p.splayed_feet {
                a.part(Surface::Metal, LABEL).tapered_tube(
                    &[
                        Vec3::new(x, 0.022, z),
                        Vec3::new(x, 0.054, z * 0.91),
                        Vec3::new(x, beam - 0.011, side * 0.015),
                    ],
                    0.020,
                    0.027,
                    8,
                );
            }
        }
        if !p.splayed_feet {
            a.box_part(
                Surface::Metal,
                LABEL,
                Vec3::new(x, 0.038, 0.),
                Vec3::new(0.062, 0.042, o.size.z * 0.80),
                0.006,
            );
            a.box_part(
                Surface::Metal,
                LABEL,
                Vec3::new(x, (0.048 + beam) * 0.5, 0.),
                Vec3::new(0.053, beam - 0.048, 0.065),
                0.004,
            );
        }
    }

    // Each seat in an installation is made from the same tooling. Build one
    // local blank including its hardware and translate its material batches.
    let blank = shell::seat(o, p, version);
    let mut sheet = stream(o.seed, u64::from_le_bytes(*b"SeatCuts"));
    for i in 0..p.bays {
        let x = p.center_x(i);
        if p.table_bay == Some(i) {
            let top = p.seat_height_m - 0.012;
            a.box_part(
                Surface::Metal,
                LABEL,
                Vec3::new(x, (top + beam) * 0.5, 0.),
                Vec3::new(0.105, top - beam - 0.025, 0.105),
                0.004,
            );
            a.box_part(
                Surface::Wood,
                LABEL,
                Vec3::new(x, top - 0.014, 0.),
                Vec3::new(p.pitch_m - 0.075, 0.028, o.size.z * 0.75),
                0.010,
            );
        } else {
            let cut = Vec2::new(sheet.random_range(0.0..4.0), sheet.random_range(0.0..4.0));
            for ((surface, label), g) in &blank.parts {
                a.part(*surface, label)
                    .append_translated(g, Vec3::X * x, cut);
            }
        }
    }

    for i in 0..=p.bays {
        let Some(x) = p.arm_x(o, i) else {
            continue;
        };
        let y = p.seat_height_m + 0.193;
        // The inboard toe joins the beam even at the two capped ends.
        let root_x = x.clamp(-o.size.x * 0.5 + 0.065, o.size.x * 0.5 - 0.065);
        a.part(Surface::Metal, LABEL).tapered_tube(
            &[
                Vec3::new(root_x, beam, 0.),
                Vec3::new(x, beam + 0.074, -0.035),
                Vec3::new(x, y - 0.085, -0.091),
                Vec3::new(x, y - 0.050, -0.102),
                Vec3::new(x, y - 0.027, -0.099),
                Vec3::new(x, y - 0.010, -0.085),
                Vec3::new(x, y, -0.058),
                Vec3::new(x, y, o.size.z * 0.24),
            ],
            0.018,
            0.014,
            10,
        );
    }
}
