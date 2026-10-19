//! Sliding door banks with fitted pockets, a bounded car, and separate finishes.
use super::*;

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

pub fn parameters(o: &IndoorObject) -> ElevatorProgram {
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
        doors: [Surface::Metal, Surface::Chrome, Surface::Paint][finish.random_range(0..3)],
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

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    // The car floor ends where the threshold starts: overlapping top faces
    // would flicker even though both slabs fit the elevator envelope.
    let threshold_back = 0.46 - p.recess - (p.leaves_per_side - 1) as f32 * 0.026 - 0.02;
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
    panel(
        a,
        o,
        Surface::Metal,
        "elevator",
        [cx, p.threshold * 0.5, (threshold_back + 0.5) * 0.5],
        [width, p.threshold, 0.5 - threshold_back],
    );
    for i in 0..p.leaves_per_side {
        panel(
            a,
            o,
            Surface::Rubber,
            "elevator",
            [cx, p.threshold + 0.0007, 0.46 - p.recess - i as f32 * 0.026],
            [width, 0.001, 0.004],
        );
    }
    let side = p.control_side;
    let jamb = if side < 0. {
        (-0.5 + p.portal_left) * 0.5
    } else {
        (0.5 + p.portal_right) * 0.5
    };
    panel(
        a,
        o,
        Surface::Metal,
        "control_panel",
        [jamb, p.control_height, 0.493],
        [0.040, 0.10, 0.006],
    );
    for delta in [-0.023, 0.023] {
        a.part(Surface::Chrome, "control_panel").cylinder(
            o.size.x * 0.009,
            o.size.z * 0.0015,
            Transform::from_xyz(
                jamb * o.size.x,
                (p.control_height + delta) * o.size.y,
                o.size.z * 0.497,
            )
            .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
        );
    }
    let indicator = if p.indicator_side {
        [jamb, p.portal_top * 0.84]
    } else {
        [cx, (1. + p.portal_top) * 0.5]
    };
    panel(
        a,
        o,
        Surface::Ink,
        "control_panel",
        [indicator[0], indicator[1], 0.493],
        [0.054, 0.045, 0.006],
    );
    const DIGITS: [u8; 10] = [0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f];
    for (i, digit) in [p.floor_number / 10, p.floor_number % 10]
        .into_iter()
        .enumerate()
    {
        for (bit, (x, y, w, h)) in [
            (0., 1., 0.010, 0.0015),
            (1., 0.5, 0.0015, 0.011),
            (1., -0.5, 0.0015, 0.011),
            (0., -1., 0.010, 0.0015),
            (-1., -0.5, 0.0015, 0.011),
            (-1., 0.5, 0.0015, 0.011),
            (0., 0., 0.010, 0.0015),
        ]
        .into_iter()
        .enumerate()
        {
            if DIGITS[digit as usize] & (1 << bit) != 0 {
                panel(
                    a,
                    o,
                    Surface::Light,
                    "control_panel",
                    [
                        indicator[0] + (i as f32 - 0.5) * 0.022 + x * 0.006,
                        indicator[1] + y * 0.013,
                        0.497,
                    ],
                    [w, h, 0.001],
                );
            }
        }
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
}
