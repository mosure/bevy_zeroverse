//! Stock dimensions, finishes and suspension for currently installed fans.
use super::*;
use crate::scene::procedural_indoor::objects::equipment::pendant::Canopy;

// Older manifests used small untagged styles and derive their shape from the instance seed.
pub(super) const AXLE_CONTACT: &str = "contact_fan_axle";
pub(super) const INSTALLED: u32 = 1 << 31;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BladeFamily {
    Paddle,
    Carved,
    Swept,
    Commercial,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BladeProfile {
    pub family: BladeFamily,
    pub root_chord_ratio: f32,
    pub shoulder_span: f32,
    pub tip_chord_ratio: f32,
    pub tip_pitch_radians: f32,
    pub camber_m: f32,
    pub tip_rise_m: f32,
}

pub(crate) fn installed_variant(seed: u64, diameter: f32) -> u32 {
    let mut stock = stream(seed, u64::from_le_bytes(*b"FanStock"));
    let multi = 0.09 + 0.26 * ((diameter - 0.85) / 1.10).clamp(0., 1.);
    let draw: f32 = stock.random();
    let style = if draw < 0.27 {
        0
    } else if draw < 0.54 {
        1
    } else if draw < 1. - multi {
        2
    } else {
        3
    };
    INSTALLED | (stock.random::<u32>() & !3) | style
}

pub(super) struct Fit {
    pub y: f32,
    pub band: f32,
    pub top: f32,
    pub mount: f32,
}

impl Fit {
    pub(super) fn new(size: Vec3) -> Self {
        // A shortened downrod changes the suspension, not the stocked rotor.
        let band = (0.037 + 0.014 * size.x.min(size.z))
            .clamp(0.043, 0.069)
            .min(size.y * 0.25);
        let y = band * 1.40;
        Self {
            y,
            band,
            top: y + band,
            mount: (size.x.min(size.z) * 0.083 + 0.032).clamp(0.09, 0.20),
        }
    }
}

pub(crate) fn canopy(o: &IndoorObject) -> Canopy {
    let mount = Fit::new(o.size).mount;
    let stock = ((o.variant >> 8) & 255) as f32 / 255.;
    let r = mount * (0.57 + 0.29 * stock);
    let height_m = 0.026 + 0.24 * r;
    Canopy {
        radius_m: r,
        height_m,
        pad_radius_m: r * 0.25,
        pad_height_m: height_m * 0.35,
    }
}

/// Chord and sweep are fractions of rotor radius. Root and tip chord are
/// fractions of maximum chord. Camber is a fraction of the vertical allowance.
struct BladeStock {
    family: BladeFamily,
    count: std::ops::RangeInclusive<u8>,
    chord: std::ops::Range<f32>,
    root: std::ops::Range<f32>,
    shoulder: std::ops::Range<f32>,
    tip: std::ops::Range<f32>,
    sweep: std::ops::Range<f32>,
    pitch: std::ops::Range<f32>,
    washout: std::ops::Range<f32>,
    camber: std::ops::Range<f32>,
}

const BLADE_STOCK: [BladeStock; 4] = [
    BladeStock {
        family: BladeFamily::Paddle,
        count: 3..=6,
        chord: 0.19..0.33,
        root: 0.35..0.67,
        shoulder: 0.46..0.74,
        tip: 0.49..0.89,
        sweep: -0.06..0.08,
        pitch: 0.13..0.30,
        washout: 0.69..0.98,
        camber: 0.00..0.11,
    },
    BladeStock {
        family: BladeFamily::Carved,
        count: 2..=4,
        chord: 0.24..0.43,
        root: 0.32..0.62,
        shoulder: 0.30..0.62,
        tip: 0.11..0.45,
        sweep: -0.12..0.20,
        pitch: 0.24..0.49,
        washout: 0.32..0.74,
        camber: 0.12..0.24,
    },
    BladeStock {
        family: BladeFamily::Swept,
        count: 3..=5,
        chord: 0.20..0.35,
        root: 0.40..0.76,
        shoulder: 0.32..0.65,
        tip: 0.12..0.46,
        sweep: 0.13..0.32,
        pitch: 0.22..0.46,
        washout: 0.37..0.80,
        camber: 0.11..0.24,
    },
    BladeStock {
        family: BladeFamily::Commercial,
        count: 6..=9,
        chord: 0.14..0.24,
        root: 0.53..0.78,
        shoulder: 0.40..0.77,
        tip: 0.56..0.93,
        sweep: -0.08..0.13,
        pitch: 0.16..0.40,
        washout: 0.55..0.87,
        camber: 0.06..0.18,
    },
];

pub(super) fn parameters(o: &IndoorObject) -> FanProgram {
    use BladeFamily::*;
    let mut r = stream(u64::from(o.variant), u64::from_le_bytes(*b"FanBlade"));
    let style = (o.variant & 3) as u8;
    let stock = &BLADE_STOCK[usize::from(style)];
    let family = stock.family;
    let fit = Fit::new(o.size);
    let radius = o.size.x.min(o.size.z) * r.random_range(0.455..0.485);
    let width = radius * r.random_range(stock.chord.clone());
    let root_pitch: f32 = r.random_range(stock.pitch.clone());
    let pitch = root_pitch.min((1.14 * fit.band / width).min(0.89).asin());
    let mut finish = stream(
        o.finish_seed.unwrap_or(u64::from(o.variant)),
        u64::from_le_bytes(*b"FanStock"),
    );
    let finish_draw: f32 = finish.random();
    let blade_surface = match family {
        Carved if finish_draw < 0.87 => Surface::Wood,
        Paddle if finish_draw < 0.57 => Surface::Wood,
        Swept if finish_draw < 0.52 => Surface::Plastic,
        Commercial if finish_draw < 0.12 => Surface::Chrome,
        _ if finish_draw < 0.88 => Surface::Metal,
        _ => Surface::Plastic,
    };
    let hub_radius = fit.mount * r.random_range(0.66..0.95);
    let root_chord_ratio = r.random_range(stock.root.clone());
    let tip_chord_ratio = r.random_range(stock.tip.clone());
    FanProgram {
        product: Some(BladeProfile {
            family,
            root_chord_ratio,
            shoulder_span: r.random_range(stock.shoulder.clone()),
            tip_chord_ratio,
            tip_pitch_radians: pitch * r.random_range(stock.washout.clone()),
            camber_m: fit.band * r.random_range(stock.camber.clone()),
            tip_rise_m: fit.band * r.random_range(-0.12..0.16),
        }),
        style,
        blades: r.random_range(stock.count.clone()),
        radius_m: radius,
        blade_width_m: width,
        pitch_radians: pitch,
        sweep_m: radius * r.random_range(stock.sweep.clone()),
        taper: 1. - tip_chord_ratio,
        thickness_m: match family {
            Carved => width * r.random_range(0.072..0.13),
            Paddle => width * r.random_range(0.043..0.072),
            _ => width * r.random_range(0.031..0.068),
        }
        .min(0.28 * fit.band),
        rotor_y: fit.y,
        rotor_top: fit.top,
        hub_radius,
        phase: stream(o.seed, u64::from_le_bytes(*b"FanPhase")).random_range(0.0..TAU),
        blade_surface,
        housing_surface: if finish.random_bool(0.10) {
            Surface::Chrome
        } else {
            Surface::Metal
        },
    }
}

pub(super) fn build(
    a: &mut Assembly,
    rotor: &mut Assembly,
    o: &IndoorObject,
    p: &FanProgram,
    section: &BladeProfile,
    moving: bool,
) {
    let label = "ceiling_fan";
    let hub = p.hub_radius;
    let y = p.rotor_y;
    let band = p.rotor_top - y;
    let shoulder = match section.family {
        BladeFamily::Carved => 0.60,
        BladeFamily::Commercial => 0.89,
        _ => 0.74,
    };
    rotor.part(p.housing_surface, label).lathe(
        &[
            (0., y - band * 0.69),
            (hub * 0.49, y - band * 0.69),
            (hub * 0.85, y - band * 0.48),
            (hub, y - band * 0.13),
            (hub, y + band * 0.32),
            (hub * shoulder, p.rotor_top),
            (0., p.rotor_top),
        ],
        24,
        Transform::IDENTITY,
    );
    blade::build(rotor.part(p.blade_surface, label), o, p, section);

    let bottom = p.rotor_top + 0.003;
    let c = canopy(o);
    let mount_y = o.size.y - c.height_m;
    let height = (mount_y - bottom) * if o.size.y < 0.39 { 0.75 } else { 0.42 };
    let motor_top = bottom + height.min(hub * 1.15);
    let rod = (hub * 0.15).clamp(0.010, 0.023);
    // The axle crosses the bearing gap. The motor, downrod and mounting cup stay fixed.
    // Only the coaxial spindle enters the rotating cap. Keep this contact
    // separate for the motion planner; a blade hitting the casing still blocks spin.
    let contact = moving.then(|| format!("{label}#{AXLE_CONTACT}"));
    a.part(p.housing_surface, contact.as_deref().unwrap_or(label))
        .lathe(
            &[(hub * 0.24, p.rotor_top - 0.001), (hub * 0.24, bottom)],
            24,
            Transform::IDENTITY,
        );
    a.part(p.housing_surface, label).lathe(
        &[
            (hub * 0.24, bottom),
            (hub * 0.61, bottom),
            (hub * 0.89, bottom + 0.19 * (motor_top - bottom)),
            (hub * 0.89, bottom + 0.61 * (motor_top - bottom)),
            (hub * 0.57, motor_top),
            (rod * 1.68, motor_top),
            (rod, motor_top + 0.007),
            (rod, mount_y),
            (c.radius_m * 0.39, mount_y),
            (c.radius_m, mount_y + c.height_m * 0.45),
            (c.radius_m, o.size.y - c.pad_height_m),
            (c.pad_radius_m, o.size.y - c.pad_height_m),
            (c.pad_radius_m, o.size.y),
        ],
        24,
        Transform::IDENTITY,
    );
}
