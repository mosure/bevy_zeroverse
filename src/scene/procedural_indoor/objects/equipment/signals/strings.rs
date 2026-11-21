//! A festoon has supported spans, a wired socket at each station and a finite
//! set of bulb finishes. All positions are in the object's local metric frame.
use super::*;
use rand::seq::SliceRandom;

mod bulb;
mod light;
pub(crate) use light::{
    illumination_scale, practical_lights, reflection_apertures, spawn_miniature_lights, BulbLight,
};

const SEGMENTS: usize = 16;
const MAX_PRACTICAL: usize = 4;

/// Dimensions relative to socket radius. The opening and recess set the
/// fully visible and fully occluded angles of the circular diffuser.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Reflector {
    pub lens_radius: f32,
    pub mouth_radius: f32,
    pub recess: f32,
}

impl Reflector {
    const LENS_DEPTH: f32 = 1.24;

    fn beam(self) -> [f32; 2] {
        [
            (self.mouth_radius - self.lens_radius).atan2(self.recess),
            (self.mouth_radius + self.lens_radius).atan2(self.recess),
        ]
    }
}

pub fn suspended(o: &IndoorObject) -> bool {
    o.variant & 1 != 0
}

pub fn practical(o: &IndoorObject) -> bool {
    o.kind == ObjectKind::StringLights && o.variant & 2 != 0
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StringProgram {
    pub suspended: bool,
    pub practical: bool,
    pub bulbs: usize,
    pub bays: usize,
    pub sag: f32,
    pub skew: f32,
    pub bow: f32,
    pub radius_m: f32,
    pub drop_m: f32,
    pub shape: u32,
    pub reflector: Option<Reflector>,
    pub colors_srgb: Vec<[f32; 3]>,
    /// Nominal diffuser luminance in cd/m² before individual lamp tolerances.
    pub luminance: f32,
    pub spacing_jitter: f32,
    pub clipped: bool,
    pub supports: Vec<Vec3>,
    pub span_sags: Vec<f32>,
    pub channel_gain: Vec<f32>,
    ordered: bool,
    lean: f32,
    twist: f32,
    phase: f32,
}

pub fn program(o: &IndoorObject) -> StringProgram {
    let mut r = stream(o.seed, 0x535452494e4753);
    let practical = practical(o);
    let clipped = r.random_bool(if practical { 0.24 } else { 0.56 });
    let irregular: f32 = r.random();
    let radius = if practical {
        r.random_range(0.025..0.052)
    } else {
        r.random_range(0.006..0.020)
    };
    let radius = f32::min(radius, (o.size.z * 0.18).min(o.size.y * 0.075));
    let pitch = if practical {
        r.random_range(0.56..1.25)
    } else {
        r.random_range(0.14..0.40)
    };
    let jitter = 0.025 + 0.26 * irregular;
    let capacity = (0.96 * o.size.x * (1. - 2. * jitter) / (2.26 * radius + 0.008))
        .floor()
        .max(1.) as usize;
    let bulbs = ((o.size.x / pitch).round() as usize)
        .clamp(2, if practical { MAX_PRACTICAL } else { 36 })
        .min(capacity);
    let bays = r.random_range(1..=(o.size.x / 1.1).round().clamp(1., 4.) as usize);
    let skew = r.random_range(-0.16..0.16);
    let margin = (1.14 * radius + 0.004) / o.size.z;
    let bow = r.random_range(0.035..0.34_f32.min((0.5 - margin).max(0.036)));
    let sag = r.random_range(0.085..0.52);
    let drop_m = if clipped {
        radius * r.random_range(0.28..0.85)
    } else {
        o.size.y * r.random_range(0.045..0.22)
    };
    let shape = if practical && r.random_bool(0.36) {
        4
    } else {
        r.random_range(0..4)
    };
    let mut supports = Vec::with_capacity(bays + 1);
    let mut span_sags = Vec::with_capacity(bays);
    let phase: f32 = r.random_range(0.0..TAU);
    for i in 0..=bays {
        let shift = if i == 0 || i == bays {
            0.
        } else {
            r.random_range(-0.33..0.33) * irregular
        };
        let t = (i as f32 + shift) / bays as f32;
        let reserve = (drop_m * 1.32 + radius * 3.8 + 0.005) / o.size.y;
        let y =
            (0.78 + skew * (t - 0.5) + irregular * r.random_range(-0.13..0.12)).max(reserve + 0.02);
        supports.push(Vec3::new(
            (t - 0.5) * 0.96,
            y,
            bow * r.random_range(-1.0..1.0),
        ));
        if i != 0 {
            // Keep clearance for the entire lead, socket and tilted diffuser.
            let cap = (supports[i - 1].y.min(y) - reserve).max(0.005);
            let width = (supports[i].x - supports[i - 1].x) * bays as f32 / 0.96;
            span_sags.push((sag * width * (1. + irregular * r.random_range(-0.60..0.55))).min(cap));
        }
    }
    let mut p = StringProgram {
        suspended: suspended(o),
        practical,
        bulbs,
        bays,
        sag,
        skew,
        bow,
        radius_m: radius,
        drop_m,
        shape,
        reflector: (shape == 4).then(|| Reflector {
            lens_radius: r.random_range(0.52..0.80),
            mouth_radius: r.random_range(0.85..0.95),
            recess: r.random_range(0.28..0.92),
        }),
        colors_srgb: Vec::new(),
        luminance: if practical {
            r.random_range(480_f32.ln()..2400_f32.ln()).exp()
        } else {
            r.random_range(80_f32.ln()..850_f32.ln()).exp()
        },
        spacing_jitter: jitter,
        clipped,
        supports,
        span_sags,
        channel_gain: Vec::new(),
        ordered: r.random_bool(0.30),
        lean: if clipped {
            r.random_range(0.42..1.30)
        } else {
            r.random_range(0.035..0.26)
        },
        twist: r.random_range(0.5..2.8),
        phase,
    };
    if !practical {
        // Miniature LEDs remain low-flux decorative sources even for the
        // largest globes in the palette (4π²r²L for a Lambertian sphere).
        p.luminance = p
            .luminance
            .min(4. / (4. * PI * PI * (1.06 * radius).powi(2) * 1.14));
    }
    p.sample_palette(&mut r);
    p
}

impl StringProgram {
    /// Number of distinct bulb tints, ignoring brightness grades of the same tint.
    pub fn color_count(&self) -> usize {
        self.colors_srgb
            .iter()
            .enumerate()
            .filter(|(i, rgb)| !self.colors_srgb[..*i].contains(rgb))
            .count()
    }

    fn sample_palette(&mut self, r: &mut impl Rng) {
        let style = r.random_range(0..5);
        let n = if style < 2 {
            1
        } else if style == 2 {
            2
        } else {
            r.random_range(3..=6)
        };
        let n = n.min(self.bulbs);
        let hue = r.random_range(0.0..360.);
        for i in 0..n {
            let rgb = if n == 1 {
                crate::scene::procedural_indoor::photometry::tinted_white(
                    r.random_range(1900.0..6200.),
                    r.random_range(-0.025..0.025),
                )
                .to_array()
            } else {
                let c = Color::hsl(
                    (hue + 360. * i as f32 / n as f32 + r.random_range(-15.0..15.))
                        .rem_euclid(360.),
                    if style == 4 {
                        r.random_range(0.36..0.68)
                    } else {
                        r.random_range(0.68..0.96)
                    },
                    if style == 4 {
                        r.random_range(0.65..0.81)
                    } else {
                        r.random_range(0.49..0.63)
                    },
                )
                .to_srgba();
                [c.red, c.green, c.blue]
            };
            self.colors_srgb.push(rgb);
            self.channel_gain.push(r.random_range(0.83..1.14));
        }
        // A few shared brightness grades model output differences even in a
        // single-colour set. Every allocated binding is used at least once.
        let channels = (n * 3).min(self.bulbs).min(8);
        let trim = r.random_range(0.45..0.86);
        for i in n..channels {
            self.colors_srgb.push(self.colors_srgb[i % n]);
            self.channel_gain
                .push(self.channel_gain[i % n] * if i < 2 * n { trim } else { 0.5 + 0.5 * trim });
        }
    }

    /// Bay-local progress preserves exact anchor vertices at i / bays.
    pub fn point(&self, size: Vec3, t: f32) -> Vec3 {
        let f = t.clamp(0., 1.) * self.bays as f32;
        let bay = (f as usize).min(self.bays - 1);
        self.in_bay(size, bay, f - bay as f32)
    }
    fn in_bay(&self, size: Vec3, bay: usize, u: f32) -> Vec3 {
        size * (self.supports[bay].lerp(self.supports[bay + 1], u)
            - Vec3::Y * (4. * u * (1. - u) * self.span_sags[bay]))
    }
    pub fn anchors(&self, size: Vec3) -> impl Iterator<Item = Vec3> + '_ {
        self.supports.iter().map(move |q| size * *q)
    }
    fn line(&self, size: Vec3) -> Vec<Vec3> {
        let mut line = Vec::with_capacity(self.bays * SEGMENTS + 1);
        line.push(size * self.supports[0]);
        for bay in 0..self.bays {
            line.extend((1..=SEGMENTS).map(|i| self.in_bay(size, bay, i as f32 / SEGMENTS as f32)));
        }
        line
    }
    fn fastener(&self, q: Vec3, size: Vec3) -> Vec3 {
        if self.suspended {
            q.with_y(size.y - 0.008)
        } else {
            q.with_z(-0.5 * size.z + 0.008)
        }
    }
    pub fn local_reservations(&self, o: &IndoorObject) -> Vec<(Vec3, Vec3)> {
        let line = self.line(o.size);
        let mut boxes = Vec::with_capacity(line.len() + 2 * (self.bays + 1) + 2 * self.bulbs);
        let segment =
            |a: Vec3, b: Vec3, r: f32| (a.min(b) - Vec3::splat(r), a.max(b) + Vec3::splat(r));
        for pair in line.windows(2) {
            boxes.push(segment(pair[0], pair[1], 0.003));
        }
        for q in self.anchors(o.size) {
            let end = self.fastener(q, o.size);
            boxes.push(segment(q, end, 0.003));
            boxes.push(segment(end, end, 0.008));
        }
        for bulb in self.sockets_on(o, line) {
            boxes.push(segment(bulb.tap, bulb.socket, 0.002));
            boxes.push(bulb.bounds());
        }
        boxes
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = program(o);
    let l = o.kind.class_name();
    let line = p.line(o.size);
    a.part(Surface::Rubber, l).tube(&line, 0.002, 6);
    for q in p.anchors(o.size) {
        let end = p.fastener(q, o.size);
        a.part(Surface::Metal, l).rod(q, end, 0.0025);
        a.box_part(Surface::Metal, l, end, Vec3::splat(0.016), 0.002);
    }
    for bulb in p.sockets_on(o, line) {
        a.part(Surface::Rubber, l).rod(bulb.tap, bulb.socket, 0.002);
        bulb.housing(a, l);
        let tag = if p.practical { "#practical" } else { "" };
        bulb.aperture(a.part(Surface::Light, &format!("{l}#bulb{}{tag}", bulb.channel)));
    }
}

#[cfg(test)]
mod tests;
