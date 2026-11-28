//! Shop rails in local metres. +Z faces the reserved browsing aisle. The same
//! arm and hook positions drive the hardware, cloth, catalog and dataset metrics.
use super::*;
use rand::seq::SliceRandom;

mod hardware;
mod stock;
#[cfg(test)]
mod tests;

pub const DISPLAY_RAIL_VERSION: u32 = 305;
const WIRE: f32 = 0.0022;
const ARM_RADIUS: f32 = 0.014;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum RailForm {
    SideHang,
    StraightFaceOut,
    Waterfall,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct StockArm {
    pub start_m: Vec3,
    pub end_m: Vec3,
    /// Horizontal distances from the socket to the retaining pins.
    pub stops_m: Vec<f32>,
}

impl StockArm {
    fn run(&self) -> f32 {
        self.start_m.xz().distance(self.end_m.xz())
    }

    fn point(&self, distance: f32) -> Vec3 {
        self.start_m.lerp(self.end_m, distance / self.run())
    }

    fn hook_frame(&self) -> (Vec3, Vec3) {
        let axis = (self.end_m - self.start_m).normalize();
        let across = axis.cross(Vec3::Y).normalize();
        (across, across.cross(axis))
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct HungItem {
    pub arm: usize,
    pub collection: usize,
    pub along_m: f32,
    pub yaw_rad: f32,
    pub size_grade: f32,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ApparelCollection {
    pub cut: Garment,
    pub sleeve_finish_slot: Option<usize>,
}

impl ApparelCollection {
    fn build(&self, a: &mut Assembly) {
        self.cut.build_with_sleeve_dye(a, self.sleeve_finish_slot);
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ApparelRail {
    pub form: RailForm,
    pub frame: Surface,
    pub hardware: Surface,
    pub hanger_stock: Surface,
    pub spine_z_m: f32,
    pub spine_y_m: f32,
    pub post_half_span_m: f32,
    pub arms: Vec<StockArm>,
    /// Sewn templates at nominal size about the hanger neck in local space.
    pub collections: Vec<ApparelCollection>,
    pub hangers: Vec<HungItem>,
}

/// Resolve the construction of a tall display for the given archived generator.
pub fn apparel_rail(o: &IndoorObject, generator_version: u32) -> Option<ApparelRail> {
    if generator_version < DISPLAY_RAIL_VERSION || o.kind != ObjectKind::ApparelDisplay {
        return None;
    }
    match parameters(o) {
        Installation::Rail {
            hangers,
            length,
            frame,
        } => Some(ApparelRail::sample(o, hangers, length, frame)),
        _ => None,
    }
}

impl ApparelRail {
    pub(in super::super) fn sample(
        o: &IndoorObject,
        count: usize,
        length: f32,
        frame: Surface,
    ) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"ShopArms"));
        let form = match r.random_range(0..100) {
            0..55 => RailForm::SideHang,
            55..76 => RailForm::StraightFaceOut,
            _ => RailForm::Waterfall,
        };
        let side_hang = form == RailForm::SideHang;
        let s = o.size;
        let half_span = s.x * 0.5 - r.random_range(0.040..0.065);
        let spine_z = if side_hang {
            0.
        } else {
            -s.z * r.random_range(0.27..0.39)
        };
        let spine_y = s.y - 0.050;
        let bays = if side_hang {
            1
        } else {
            ((2. * half_span / r.random_range(0.54..0.70)).floor() as usize).clamp(1, 3)
        };
        let mut p = Self {
            form,
            frame,
            spine_y_m: spine_y,
            spine_z_m: spine_z,
            post_half_span_m: half_span,
            hardware: if r.random_bool(0.60) {
                Surface::Chrome
            } else {
                Surface::Metal
            },
            hanger_stock: [Surface::Wood, Surface::Plastic, Surface::Metal][r.random_range(0..3)],
            arms: Vec::with_capacity(bays),
            collections: Vec::new(),
            hangers: Vec::with_capacity(count),
        };
        let bay_width = 2. * half_span / bays as f32;
        for bay in 0..bays {
            let center = -half_span + (bay as f32 + 0.5) * bay_width;
            let x = center + r.random_range(-0.026..0.026) * bay_width;
            let (start, end) = if side_hang {
                (
                    Vec3::new(-half_span, spine_y, spine_z),
                    Vec3::new(half_span, spine_y, spine_z),
                )
            } else {
                let reach = r
                    .random_range(0.31_f32..0.54)
                    .min(s.z * 0.5 - spine_z - 0.065);
                let fall = if form == RailForm::Waterfall {
                    r.random_range(0.17..0.48) * reach
                } else {
                    0.
                };
                let root = Vec3::new(x, spine_y, spine_z);
                (root, root + Vec3::new(0., -fall, reach))
            };
            let mut arm = StockArm {
                start_m: start,
                end_m: end,
                stops_m: Vec::new(),
            };
            let budget = count / bays + usize::from(bay < count % bays);
            let spots = stocking_positions(&mut arm, form, budget, &mut r);
            let cross_space = if side_hang {
                s.z
            } else {
                bay_width - 2. * (x - center).abs()
            };
            p.dress_arm(o, &arm, &spots, cross_space, length);
            p.arms.push(arm);
        }
        p
    }

    pub(in super::super) fn build(&self, a: &mut Assembly, size: Vec3) {
        self.frame(a, size);
        // Grade each copy of a retail line from the same sewn pattern. A rail
        // stocking six sizes need only tessellate that line once.
        let templates: Vec<_> = self
            .collections
            .iter()
            .map(|g| {
                let mut mesh = Assembly::default();
                g.build(&mut mesh);
                mesh
            })
            .collect();
        for h in &self.hangers {
            let neck = self.hanger(a, h);
            let rotation = h.yaw_rad.sin_cos();
            for ((surface, label), g) in &templates[h.collection].parts {
                append_facing(a.part(*surface, label), g, neck, rotation, h.size_grade);
            }
        }
    }
}

fn stocking_positions(
    arm: &mut StockArm,
    form: RailForm,
    budget: usize,
    r: &mut ChaCha8Rng,
) -> Vec<f32> {
    let run = arm.run();
    if form == RailForm::Waterfall {
        let stations = ((run - 0.068) / r.random_range(0.072..0.095))
            .floor()
            .clamp(2., 7.) as usize;
        let pitch = (run - 0.068) / stations as f32;
        arm.stops_m = (1..=stations).map(|i| 0.045 + i as f32 * pitch).collect();
        let cos_slope = run / arm.start_m.distance(arm.end_m);
        let mut places: Vec<_> = arm
            .stops_m
            .iter()
            .map(|&stop| stop - (hardware::STOP_RADIUS + WIRE) * cos_slope)
            .collect();
        places.shuffle(r);
        places.truncate(budget.min(stations));
        places.sort_by(f32::total_cmp);
        return places;
    }
    let inset = if form == RailForm::SideHang {
        0.083
    } else {
        0.053
    };
    let span = run - 2. * inset;
    let n = budget.min((span / 0.090).floor() as usize + 1).max(1);
    // Leave a variable gap between collections, or extra room at an end after
    // browsing. Adjacent garments always retain a minimum physical pitch.
    let slack = span - (n - 1) as f32 * 0.090;
    let mut gaps: Vec<f32> = (0..=n).map(|_| r.random_range(0.18..1.0)).collect();
    let gap = r.random_range(0..=n);
    gaps[gap] *= r.random_range(1.5..7.0);
    let unit = slack / gaps.iter().sum::<f32>();
    let mut at = inset + unit * gaps[0];
    (0..n)
        .map(|i| {
            let current = at;
            at += 0.090 + unit * gaps[i + 1];
            current
        })
        .collect()
}
