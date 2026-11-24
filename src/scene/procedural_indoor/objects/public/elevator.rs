//! Sliding door banks with fitted pockets, a bounded car, and separate finishes.
use super::*;
use crate::scene::procedural_indoor::materials::worktop_finish::BrushedHardware;
mod legacy;
mod sill;
mod station;
#[cfg(test)]
mod tests;
pub(crate) use sill::Sill;
pub(crate) use station::{Levels, Station};

pub(crate) const LANDING_VERSION: u32 = 229;

pub(crate) const MIN_FOOTPRINT: Vec2 = Vec2::new(2.05, 1.50);
/// Preserve a 2.50 m car plus the enclosing core's roof structure.
pub(crate) const MIN_CORE_HEIGHT: f32 = 2.66;

#[derive(Clone, Debug, serde::Serialize)]
pub struct ElevatorProgram {
    pub centered: bool,
    pub opening_side: f32,
    pub leaves_per_side: usize,
    pub opening: f32,
    pub portal_left: f32,
    pub portal_right: f32,
    pub threshold: f32,
    pub portal_top: f32,
    pub recess: f32,
    pub surround: Surface,
    pub doors: Surface,
    pub lining: Surface,
    pub floor: Surface,
    pub lining_bays: usize,
    pub rail_height: f32,
    pub rail_sides: bool,
    pub light_width: f32,
    pub light_depth: f32,
    pub control_side: f32,
    pub control_height: f32,
    pub indicator_side: bool,
    pub floor_number: u32,
}

fn sample(o: &IndoorObject) -> ElevatorProgram {
    let mut r = stream(o.seed, 8240);
    let mut finish = stream(o.finish_seed.unwrap_or(o.seed), 8241);
    let centered = r.random_bool(0.68);
    let leaves = if centered {
        r.random_range(1..=2)
    } else {
        r.random_range(2..=3)
    };
    // At full travel every panel fits behind a jamb, including its complete
    // width. Closing and pocketed states share exactly the same solid envelope.
    let (left, right) = if centered {
        let half = r.random_range(0.195..(0.46 * leaves as f32 / (leaves + 1) as f32 - 0.008));
        (-half, half)
    } else {
        let left = r.random_range(-0.41..-0.35);
        let width =
            r.random_range(0.46..((0.46 - left) * leaves as f32 / (leaves + 1) as f32 - 0.008));
        (left, left + width)
    };
    let opening_side = if r.random_bool(0.5) { -1. } else { 1. };
    let (left, right) = if !centered && opening_side < 0. {
        (-right, -left)
    } else {
        (left, right)
    };
    ElevatorProgram {
        centered,
        opening_side,
        leaves_per_side: leaves,
        opening: if r.random_bool(0.30) {
            0.
        } else {
            r.random_range(0.08..1.)
        },
        portal_left: left,
        portal_right: right,
        // A small metric lip, not a percentage-height step into the car.
        threshold: r.random_range(0.003..0.012) / o.size.y,
        portal_top: r.random_range(0.78..0.89),
        recess: r.random_range(0.06..0.13),
        surround: [
            Surface::Metal,
            Surface::Wood,
            Surface::Concrete,
            Surface::Paint,
        ][finish.random_range(0..4)],
        doors: [Surface::Metal, Surface::Chrome, Surface::Plastic][finish.random_range(0..3)],
        lining: [Surface::Metal, Surface::Wood, Surface::Paint][finish.random_range(0..3)],
        floor: [Surface::Concrete, Surface::Ceramic, Surface::Wood][finish.random_range(0..3)],
        lining_bays: r.random_range(2..=6),
        rail_height: r.random_range(0.33..0.45),
        rail_sides: r.random_bool(0.65),
        light_width: r.random_range(0.22..0.78),
        light_depth: r.random_range(0.10..0.58),
        control_side: if r.random_bool(0.5) { -1. } else { 1. },
        control_height: r.random_range(0.38..0.53),
        indicator_side: r.random_bool(0.35),
        floor_number: r.random_range(0..30),
    }
}

/// Local fractional bounds of each leaf, also used by clearance qualification.
pub fn leaf_bounds(p: &ElevatorProgram) -> Vec<([f32; 3], [f32; 3])> {
    let sides: &[f32] = if p.centered {
        &[-1., 1.]
    } else {
        std::slice::from_ref(&p.opening_side)
    };
    let bank = (p.portal_right - p.portal_left) / sides.len() as f32;
    let w = bank / p.leaves_per_side as f32;
    let z = 0.46 - p.recess;
    let h = p.portal_top - p.threshold;
    let mut leaves = Vec::with_capacity(sides.len() * p.leaves_per_side);
    for &side in sides {
        for i in 0..p.leaves_per_side {
            let closed = if p.centered {
                side * (i as f32 + 0.5) * w
            } else {
                p.portal_left + (i as f32 + 0.5) * w
            };
            let parked = if side < 0. {
                p.portal_left - w * 0.5
            } else {
                p.portal_right + w * 0.5
            };
            let x = closed + (parked - closed) * p.opening;
            leaves.push((
                [x, p.threshold + h * 0.5, z - i as f32 * 0.026],
                [w - 0.0015, h - 0.003, 0.014],
            ));
        }
    }
    leaves
}

pub fn is_core(o: &IndoorObject) -> bool {
    o.kind == ObjectKind::Elevator && (o.variant == 20 || Levels::decode(o.variant).is_some())
}

pub fn car_height(o: &IndoorObject) -> f32 {
    if is_core(o) {
        (o.size.y - 0.16).min(2.72)
    } else {
        o.size.y
    }
}

fn car_size(o: &IndoorObject) -> Vec3 {
    if is_core(o) {
        Vec3::new(o.size.x * 0.84, car_height(o), o.size.z * 0.92)
    } else {
        o.size
    }
}

pub fn parameters(o: &IndoorObject) -> ElevatorProgram {
    let mut p = sample(o);
    let car = car_size(o);
    let station = Station::sample(o, car, &p);
    p.floor_number = u32::from(station.displayed_floor);
    p.control_side = station.button_center_m.x.signum();
    p.control_height = station.button_center_m.y / car.y;
    p.indicator_side = station.mount != station::ReadoutMount::Header;
    p
}

pub(crate) fn installation(o: &IndoorObject) -> (Station, Sill) {
    let size = car_size(o);
    let mut p = sample(o);
    p.threshold *= o.size.y / size.y;
    (Station::sample(o, size, &p), Sill::sample(o, size, &p))
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    if !is_core(o) {
        car(a, o, o, version);
        return;
    }
    let mut inner = o.clone();
    inner.variant = 0;
    inner.size = car_size(o);
    let p = car(a, &inner, o, version);
    if version < LANDING_VERSION {
        for ((_, label), g) in &mut a.parts {
            if label == "control_panel" {
                for point in &mut g.positions {
                    point[2] += o.size.z * (0.485 - 0.493 * 0.92);
                }
            }
        }
    }
    let l = "wall";
    // Full-height masonry/service-core enclosure, with a genuine recessed portal.
    // It replaces the exposed exterior of the car; no wall covers its doorway.
    for x in [-0.46, 0.46] {
        panel(a, o, Surface::Paint, l, [x, 0.5, -0.02], [0.08, 1., 0.96]);
    }
    panel(a, o, Surface::Paint, l, [0., 0.5, -0.48], [0.84, 1., 0.04]);
    let left = p.portal_left * 0.84 - 0.016;
    let right = p.portal_right * 0.84 + 0.016;
    let portal = p.portal_top * inner.size.y;
    for (lo, hi) in [(-0.42, left), (right, 0.42)] {
        if hi > lo {
            panel(
                a,
                o,
                Surface::Paint,
                l,
                [(lo + hi) * 0.5, 0.5, 0.44],
                [hi - lo, 1., 0.08],
            );
        }
    }
    a.box_part(
        Surface::Paint,
        l,
        Vec3::new(
            (left + right) * o.size.x * 0.5,
            (portal + o.size.y) * 0.5 + 0.008,
            o.size.z * 0.44,
        ),
        Vec3::new(
            (right - left) * o.size.x,
            o.size.y - portal - 0.016,
            o.size.z * 0.08,
        ),
        0.002,
    );
    // Structural roof closes the shaft over the bounded car; contact planes meet
    // rather than overlap. The whole volume was reserved before furnishing.
    a.box_part(
        Surface::Paint,
        "ceiling",
        Vec3::new(0., o.size.y - 0.025, -o.size.z * 0.02),
        Vec3::new(o.size.x * 0.84, 0.05, o.size.z * 0.88),
        0.001,
    );
}

fn threshold_back(p: &ElevatorProgram) -> f32 {
    0.46 - p.recess - (p.leaves_per_side - 1) as f32 * 0.026 - 0.02
}

fn car(a: &mut Assembly, o: &IndoorObject, outer: &IndoorObject, version: u32) -> ElevatorProgram {
    let p = sample(o);
    // The car floor ends where the threshold starts: overlapping top faces
    // would flicker even though both slabs fit the elevator envelope.
    let threshold_back = threshold_back(&p);
    for x in [-0.48, 0.48] {
        panel(
            a,
            o,
            p.lining,
            "elevator#exterior",
            [x, 0.48, -0.01],
            [0.04, 0.96, 0.94],
        );
    }
    panel(
        a,
        o,
        p.lining,
        "elevator#exterior",
        [0., 0.48, -0.48],
        [0.92, 0.96, 0.04],
    );
    panel(
        a,
        o,
        p.floor,
        "floor#exterior",
        [0., p.threshold * 0.5, (-0.48 + threshold_back) * 0.5],
        [0.92, p.threshold, threshold_back + 0.48],
    );
    panel(
        a,
        o,
        Surface::Paint,
        "ceiling#exterior",
        [0., 0.97, -0.01],
        [0.92, 0.06, 0.94],
    );
    panel(
        a,
        o,
        Surface::Light,
        "lamp#exterior",
        [0., 0.935, -0.10],
        [p.light_width, 0.008, p.light_depth],
    );
    for (lo, hi) in [(-0.5, p.portal_left), (p.portal_right, 0.5)] {
        panel(
            a,
            o,
            p.surround,
            "elevator",
            [(lo + hi) * 0.5, 0.5, 0.455],
            [hi - lo, 1., 0.07],
        );
    }
    let cx = (p.portal_left + p.portal_right) * 0.5;
    let width = p.portal_right - p.portal_left;
    panel(
        a,
        o,
        p.surround,
        "elevator",
        [cx, (1. + p.portal_top) * 0.5, 0.455],
        [width, 1. - p.portal_top, 0.07],
    );
    for (pos, size) in leaf_bounds(&p) {
        panel(a, o, p.doors, "door", pos, size);
        // Thin stile sits on the leaf front, never on a coincident surface.
        panel(
            a,
            o,
            Surface::Rubber,
            "door",
            [pos[0] - size[0] * 0.46, pos[1], pos[2] + 0.008],
            [0.002, size[1] * 0.97, 0.001],
        );
    }
    if version < LANDING_VERSION {
        legacy::hall(a, o, &p, threshold_back);
    } else {
        Sill::sample(outer, o.size, &p).build(a);
        Station::sample(outer, o.size, &p).build(a);
    }
    let radius = o.size.min_element() * 0.008;
    rod(
        a,
        o,
        Surface::Chrome,
        "elevator#exterior",
        [-0.39, p.rail_height, -0.435],
        [0.39, p.rail_height, -0.435],
        radius,
    );
    for x in [-0.32, 0.32] {
        rod(
            a,
            o,
            Surface::Chrome,
            "elevator#exterior",
            [x, p.rail_height, -0.46],
            [x, p.rail_height, -0.435],
            radius * 0.7,
        );
    }
    if p.rail_sides {
        for side in [-1., 1.] {
            rod(
                a,
                o,
                Surface::Chrome,
                "elevator#exterior",
                [side * 0.435, p.rail_height, -0.35],
                [side * 0.435, p.rail_height, 0.18],
                radius,
            );
            for z in [-0.29, 0.12] {
                rod(
                    a,
                    o,
                    Surface::Chrome,
                    "elevator#exterior",
                    [side * 0.46, p.rail_height, z],
                    [side * 0.435, p.rail_height, z],
                    radius * 0.7,
                );
            }
        }
    }
    for i in 1..p.lining_bays {
        panel(
            a,
            o,
            Surface::Rubber,
            "elevator#exterior",
            [-0.46 + 0.92 * i as f32 / p.lining_bays as f32, 0.49, -0.458],
            [0.002, 0.86, 0.001],
        );
    }
    p
}
