//! One free-weight station. Pairs share a physical product specification. An
//! unoccupied saddle remains attached to its shelf. Local +Z is the access side.
use super::*;

mod mesh;
#[cfg(test)]
mod tests;

pub(crate) const VERSION: u32 = 312;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Shelf {
    OpenRails,
    LinedTray,
    Saddles,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Head {
    Hex,
    TwelveSided,
    Round,
}

impl Head {
    fn sides(self) -> u32 {
        match self {
            Self::Hex => 6,
            Self::TwelveSided => 12,
            Self::Round => 20,
        }
    }

    /// Put a facet flat on a rail/tray. The saddle instead supports both
    /// flanks of the round head at the actual tessellated contact points.
    fn phase(self) -> f32 {
        if self == Self::Round {
            0.
        } else {
            FRAC_PI_2 - PI / self.sides() as f32
        }
    }

    fn contact(self, slope: f32) -> f32 {
        (0..self.sides())
            .map(|i| {
                let (y, x) = (self.phase() + TAU * i as f32 / self.sides() as f32).sin_cos();
                y + slope * x.abs()
            })
            .fold(0., f32::max)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum HeadFinish {
    CastIron,
    Rubber,
    ColourCodedEnds,
}

impl HeadFinish {
    pub(crate) fn for_object(o: &IndoorObject) -> Self {
        let mut finish = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"BellCoat"),
        );
        match finish.random_range(0..12) {
            0..=2 => Self::CastIron,
            3..=5 => Self::ColourCodedEnds,
            _ => Self::Rubber,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Slot {
    pub x: f32,
    pub depth_offset: f32,
    pub yaw: f32,
    /// Clocking of the two matching end marks, in fractions of a turn.
    pub face_turn: f32,
    pub present: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct WeightPair {
    pub radius: f32,
    pub head_width: f32,
    pub grip_radius: f32,
    pub nominal_kg: u8,
    pub slots: [Slot; 2],
}

#[derive(Clone, Debug, Serialize)]
pub struct Tier {
    /// Top of the rail or tray liner, or valley of the installed V saddle.
    pub support_y: f32,
    pub center_z: f32,
    pub pairs: Vec<WeightPair>,
}

#[derive(Debug, Serialize)]
pub struct FreeWeightRack {
    pub shelf: Shelf,
    pub head: Head,
    pub head_finish: HeadFinish,
    pub cantilever: bool,
    pub half_grip: f32,
    pub saddle_slope: f32,
    pub tiers: Vec<Tier>,
    contact: f32,
}

impl Tier {
    fn sample(
        usable: f32,
        max_radius: f32,
        rack: &FreeWeightRack,
        reverse: bool,
        taken: f64,
        r: &mut impl Rng,
        use_state: &mut impl Rng,
    ) -> Self {
        let n = ((usable / (3.65 * max_radius + 0.076)).round() as usize).clamp(2, 6);
        // Bound the swept half-width as well as the head. A rotated handle
        // moves its two heads in opposite directions along the shelf.
        let margin = 0.025;
        let cap = max_radius.min(usable / (4. * n as f32) - margin);
        let small = r.random_range(0.60..0.84);
        let mut pairs = Vec::with_capacity(n);
        for j in 0..n {
            let rank = if reverse { n - 1 - j } else { j };
            let t = rank as f32 / (n - 1) as f32;
            let radius = cap * (small + (1. - small) * t);
            let head_width = 0.025 + radius * r.random_range(0.40..0.49);
            let grip_radius = r.random_range(0.0135..0.0165) + t * 0.0015;
            let out = use_state.random::<f64>();
            let single = out >= taken && out < taken * 1.55;
            let remaining = use_state.random_range(0..2);
            let slots = std::array::from_fn(|k| {
                let yaw = match rack.shelf {
                    Shelf::Saddles => 0.,
                    Shelf::OpenRails => use_state.random_range(-0.065..0.065),
                    Shelf::LinedTray => use_state.random_range(-0.135..0.135),
                };
                Slot {
                    x: 0.,
                    depth_offset: use_state.random_range(-0.004..0.004),
                    yaw,
                    face_turn: if use_state.random_bool(0.45) {
                        0.
                    } else {
                        use_state.random()
                    },
                    present: out >= taken && (!single || k == remaining),
                }
            });
            pairs.push(WeightPair {
                radius,
                head_width,
                grip_radius,
                nominal_kg: rack.nominal_kg(radius, head_width, grip_radius),
                slots,
            });
        }
        let occupied: f32 = pairs.iter().map(|p| 4. * (p.radius + margin)).sum();
        let spare = (usable - occupied).max(0.);
        let gap = spare / (2 * n + 1) as f32;
        let mut x = -usable * 0.5 + gap * r.random_range(0.55..1.45);
        for pair in &mut pairs {
            let half = pair.radius + margin;
            for slot in &mut pair.slots {
                slot.x = x + half;
                x += 2. * half + gap;
            }
        }
        Self {
            support_y: 0.,
            center_z: 0.,
            pairs,
        }
    }
}

impl FreeWeightRack {
    pub fn for_object(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"BellRack"));
        let mut use_state = stream(o.seed, u64::from_le_bytes(*b"BellLoan"));
        let shelf = match r.random_range(0..10) {
            0..=3 => Shelf::Saddles,
            4..=6 => Shelf::LinedTray,
            _ => Shelf::OpenRails,
        };
        let head = if shelf == Shelf::Saddles {
            Head::Round
        } else if r.random_bool(0.72) {
            Head::Hex
        } else {
            Head::TwelveSided
        };
        let saddle_slope = if shelf == Shelf::Saddles {
            r.random_range(0.46..0.76)
        } else {
            0.
        };
        let contact = head.contact(saddle_slope);
        let half_grip = r.random_range(0.062..0.078);
        let count = if o.size.y >= 0.85 && r.random_bool(0.72) {
            3
        } else {
            2
        };
        let reverse = r.random_bool(0.5);
        let snake = r.random_bool(0.44);
        let heavy = r.random_range(0.085..0.118);
        let taken: f64 = match use_state.random_range(0..10) {
            0..=2 => 0.,
            3..=7 => use_state.random_range(0.08..0.25),
            _ => use_state.random_range(0.26..0.48),
        };
        let mut rack = Self {
            shelf,
            head,
            half_grip,
            saddle_slope,
            contact,
            head_finish: HeadFinish::for_object(o),
            cantilever: r.random_bool(0.49),
            tiers: Vec::with_capacity(count),
        };
        for i in 0..count {
            let t = i as f32 / (count - 1) as f32;
            let max_radius = heavy * (1. - t * r.random_range(0.11..0.29));
            let tier = Tier::sample(
                o.size.x - 0.15,
                max_radius,
                &rack,
                reverse ^ (snake && i % 2 == 1),
                taken,
                &mut r,
                &mut use_state,
            );
            rack.tiers.push(tier);
        }
        let tiers = &mut rack.tiers;
        let crown = head.contact(0.);
        let extent = |tier: &Tier| {
            tier.pairs
                .iter()
                .map(|p| p.radius * (contact + crown))
                .fold(0., f32::max)
        };
        // Reserve room to grasp a handle below the hardware of the next tier.
        let headroom = 0.122;
        let ceiling = o.size.y - 0.008;
        if tiers.len() == 3
            && 0.14 + 2. * headroom + tiers.iter().map(extent).sum::<f32>() > ceiling
        {
            tiers.remove(1);
        }
        if !tiers
            .iter()
            .flat_map(|tier| &tier.pairs)
            .any(|p| p.slots.iter().all(|s| s.present))
        {
            let tier = use_state.random_range(0..tiers.len());
            let pair = use_state.random_range(0..tiers[tier].pairs.len());
            for slot in &mut tiers[tier].pairs[pair].slots {
                slot.present = true;
            }
        }
        let steps = (tiers.len() - 1) as f32;
        let spare = ceiling - tiers.iter().map(extent).sum::<f32>();
        let mut y = r
            .random_range(0.155..0.232_f32)
            .min(spare - steps * headroom);
        let gap_y = (spare - y) / steps;
        let setback =
            (o.size.z * 0.5 - Self::tray_depth(half_grip) - 0.005) * r.random_range(0.60..0.98);
        for (i, tier) in tiers.iter_mut().enumerate() {
            tier.support_y = y;
            tier.center_z = setback * (1. - 2. * i as f32 / steps);
            y += extent(tier) + gap_y;
        }
        rack
    }

    fn nominal_kg(&self, radius: f32, width: f32, grip_radius: f32) -> u8 {
        // Nominal catalogue marking for a chamfered iron-core pair of heads
        // and steel grip, rounded to an even-kg increment.
        let n = self.head.sides() as f32;
        let area = 0.5 * n * (TAU / n).sin() * radius.powi(2);
        let heads = 2. * area * width * 0.93 * 7100.;
        let grip = 2. * self.half_grip * PI * grip_radius.powi(2) * 7800.;
        (0.5 * (heads + grip)).round().clamp(1., 32.) as u8 * 2
    }

    fn tray_depth(half_grip: f32) -> f32 {
        half_grip + 0.112
    }

    fn rail_offset(&self) -> f32 {
        self.half_grip + 0.031
    }
}

pub(super) use mesh::build;
