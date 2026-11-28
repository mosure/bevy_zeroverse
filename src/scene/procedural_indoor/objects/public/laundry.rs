//! Commercial front loaders. Local +Z is the loading face. A bank buys the same
//! chassis and controls; pocket contents and indicated times vary per machine.
use super::*;
use crate::scene::procedural_indoor::materials::{appliance_panel::Film, worktop_finish};
use std::f32::consts::TAU;

mod cabinet;
pub mod cart;
mod drum;
#[cfg(test)]
mod tests;

pub fn dimensions(kind: ObjectKind, seed: u64) -> Vec3 {
    let mut r = stream(seed, u64::from_le_bytes(*b"LdrySize"));
    let w = r.random_range(0.66..0.97);
    match kind {
        ObjectKind::WashingMachine => Vec3::new(
            w,
            w * r.random_range(1.31..1.55),
            w * r.random_range(0.95..1.17),
        ),
        ObjectKind::TumbleDryer => {
            let height = if r.random_bool(0.48) {
                r.random_range(1.91..2.15)
            } else {
                w * r.random_range(1.48..1.91)
            };
            Vec3::new(w, height, w * r.random_range(1.02..1.27))
        }
        ObjectKind::LaundryFoldingTable => Vec3::new(
            r.random_range(1.25..2.05),
            r.random_range(0.84..0.98),
            r.random_range(0.68..0.88),
        ),
        _ => unreachable!("laundry stock"),
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Pocket {
    pub center_y_m: f32,
    pub radius_m: f32,
    /// Depth from the front sheet to the pierced drum's backplate.
    pub depth_m: f32,
    pub load_fraction: f32,
    pub remaining_minutes: u8,
    pub phase_rad: f32,
    pub fabric_slot: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Machine {
    pub pockets: Vec<Pocket>,
    pub finish: CabinetFinish,
    pub hinge_right: bool,
    pub dial: bool,
    pub film: Film,
    pub front_z_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum CabinetFinish {
    BrushedSteel,
    WhiteEnamel,
    AlmondEnamel,
    GraphiteEnamel,
    TintedCoating,
}

impl Machine {
    pub fn for_object(o: &IndoorObject) -> Self {
        let dry = o.kind == ObjectKind::TumbleDryer;
        let mut stock = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"WashFlet"),
        );
        let mut load = stream(o.seed, u64::from_le_bytes(*b"WashLoad"));
        let n = if dry && o.size.y >= 1.90 { 2 } else { 1 };
        let h = (o.size.y - cabinet::TOE) / n as f32;
        let fraction = if dry {
            stock.random_range(0.317..0.381)
        } else {
            stock.random_range(0.277..0.348)
        };
        let radius = (fraction * o.size.x).min(0.335 * h);
        let center = h * stock.random_range(0.44..0.51);
        let depth = o.size.z * stock.random_range(0.50..0.69);
        let pockets = (0..n)
            .map(|i| {
                let filled = load.random_bool(0.62);
                Pocket {
                    center_y_m: cabinet::TOE + h * i as f32 + center,
                    radius_m: radius,
                    depth_m: depth,
                    load_fraction: if filled {
                        load.random_range(0.21..0.83)
                    } else {
                        0.
                    },
                    remaining_minutes: if filled && load.random_bool(0.80) {
                        load.random_range(1..=58)
                    } else {
                        0
                    },
                    phase_rad: load.random_range(0.0..TAU),
                    fabric_slot:
                        crate::scene::procedural_indoor::materials::garment_stock::sample_slot(
                            &mut load,
                        ),
                }
            })
            .collect();
        let finish = if stock.random_bool(if dry { 0.56 } else { 0.42 }) {
            match stock.random_range(0..12) {
                0..=6 => CabinetFinish::WhiteEnamel,
                7..=8 => CabinetFinish::AlmondEnamel,
                9..=10 => CabinetFinish::GraphiteEnamel,
                _ => CabinetFinish::TintedCoating,
            }
        } else {
            CabinetFinish::BrushedSteel
        };
        Self {
            pockets,
            finish,
            hinge_right: stock.random_bool(0.37),
            dial: stock.random_bool(0.56),
            film: if stock.random_bool(0.54) {
                Film::Led
            } else {
                Film::ReflectiveLcd
            },
            front_z_m: o.size.z * 0.5 - 0.044,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct FoldingTable {
    pub top: Surface,
    pub shelf: bool,
    pub setback_m: f32,
    pub thickness_m: f32,
}
impl FoldingTable {
    pub fn for_object(o: &IndoorObject) -> Self {
        let mut r = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"FoldDesk"),
        );
        Self {
            top: if r.random_bool(0.69) {
                Surface::Plastic
            } else {
                Surface::Wood
            },
            shelf: r.random_bool(0.59),
            setback_m: r.random_range(0.07..0.14),
            thickness_m: r.random_range(0.025..0.044),
        }
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    if o.kind == ObjectKind::LaundryFoldingTable {
        let p = FoldingTable::for_object(o);
        let (w, h, d) = (o.size.x, o.size.y, o.size.z);
        let t = p.thickness_m;
        a.box_part(
            p.top,
            &worktop_finish::table_label(o, p.top),
            Vec3::Y * (h - 0.5 * t),
            Vec3::new(w, t, d),
            0.006,
        );
        let x = w * 0.5 - p.setback_m;
        let z = d * 0.5 - 0.065;
        let rail = "table#finish111#uv_local";
        for sx in [-x, x] {
            for sz in [-z, z] {
                a.box_part(
                    Surface::Metal,
                    rail,
                    Vec3::new(sx, (h - t + 0.014) * 0.5, sz),
                    Vec3::new(0.037, h - t - 0.014, 0.037),
                    0.003,
                );
                a.box_part(
                    Surface::Rubber,
                    "table",
                    Vec3::new(sx, 0.007, sz),
                    Vec3::new(0.039, 0.014, 0.039),
                    0.002,
                );
            }
            a.box_part(
                Surface::Metal,
                rail,
                Vec3::new(sx, 0.24, 0.),
                Vec3::new(0.03, 0.034, 2. * z - 0.037),
                0.002,
            );
        }
        for sz in [-z, z] {
            a.box_part(
                Surface::Metal,
                rail,
                Vec3::new(0., h - t - 0.032, sz),
                Vec3::new(2. * x - 0.037, 0.064, 0.023),
                0.002,
            );
        }
        if p.shelf {
            a.box_part(
                Surface::Metal,
                rail,
                Vec3::new(0., 0.265, 0.),
                Vec3::new(2. * x + 0.026, 0.016, 2. * z - 0.05),
                0.003,
            );
        }
        return;
    }
    let machine = Machine::for_object(o);
    cabinet::build(a, o, &machine);
}
