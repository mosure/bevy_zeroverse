//! Metric electrical contact inserts shared by installed and portable receptacles.
//! Positions are in the local XY plane, facing +Z.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum OutletFace {
    Polarized,
    RoundWell,
    ThreePin,
}

const MODULAR: u32 = 768;
pub(crate) fn installed_variant(standard: u32) -> u32 {
    MODULAR + standard % 3
}
impl OutletFace {
    pub(crate) fn from_variant(v: u32) -> Option<Self> {
        match v.checked_sub(MODULAR)? {
            0 => Some(Self::Polarized),
            1 => Some(Self::RoundWell),
            2 => Some(Self::ThreePin),
            _ => None,
        }
    }
}

pub(super) struct Contact {
    pub face: OutletFace,
    pub contact_scale: f32,
    pub face_z_m: f32,
    pub well_depth_m: f32,
    pub insert: Surface,
    pub pin_sides: u32,
    pub well_sides: u32,
}

#[derive(Clone, Copy)]
pub(super) struct Rect {
    pub(super) lo: Vec2,
    pub(super) hi: Vec2,
}

impl Rect {
    pub(super) fn new(center: Vec2, size: Vec2) -> Self {
        Self {
            lo: center - size * 0.5,
            hi: center + size * 0.5,
        }
    }
    pub(super) fn corners(self) -> [Vec2; 4] {
        [
            self.lo,
            Vec2::new(self.hi.x, self.lo.y),
            self.hi,
            Vec2::new(self.lo.x, self.hi.y),
        ]
    }
    pub(super) fn face(self, g: &mut Geometry, z: f32) {
        if self.hi.x > self.lo.x + 1e-7 && self.hi.y > self.lo.y + 1e-7 {
            g.metric_quad(self.corners().map(|v| v.extend(z)), Vec3::Z);
        }
    }
}

/// Subdivide only at hole edges. Open interiors carry their own stock and normals.
pub(super) fn pierced(g: &mut Geometry, face: Rect, z: f32, holes: &[Rect]) {
    if holes.is_empty() {
        face.face(g, z);
        return;
    }
    let mut levels = Vec::with_capacity(2 + 2 * holes.len());
    levels.extend([face.lo.y, face.hi.y]);
    for h in holes {
        levels.extend([h.lo.y, h.hi.y]);
    }
    levels.sort_unstable_by(f32::total_cmp);
    levels.dedup();
    // Callers sort openings left-to-right. Several can share a column where
    // their vertical ranges are disjoint.
    for band in levels.windows(2) {
        let y = (band[0] + band[1]) * 0.5;
        let mut start = face.lo.x;
        for h in holes.iter().filter(|h| h.lo.y < y && y < h.hi.y) {
            Rect {
                lo: Vec2::new(start, band[0]),
                hi: Vec2::new(h.lo.x, band[1]),
            }
            .face(g, z);
            start = h.hi.x;
        }
        Rect {
            lo: Vec2::new(start, band[0]),
            hi: Vec2::new(face.hi.x, band[1]),
        }
        .face(g, z);
    }
}

pub(super) fn edge(g: &mut Geometry, a: Rect, za: f32, b: Rect, zb: f32, inward: bool) {
    let aa = a.corners();
    let bb = b.corners();
    for i in 0..4 {
        let j = (i + 1) % 4;
        let d = aa[j] - aa[i];
        let normal = Vec3::new(d.y, -d.x, 0.);
        g.metric_quad(
            [
                aa[i].extend(za),
                aa[j].extend(za),
                bb[j].extend(zb),
                bb[i].extend(zb),
            ],
            if inward { -normal } else { normal },
        );
    }
}

fn slot(a: &mut Assembly, label: &str, hole: Rect, face: f32, depth: f32) {
    let g = a.part(Surface::Ink, label);
    edge(g, hole, face, hole, face - depth, true);
    hole.face(g, face - depth);
}

pub(super) fn directions(n: u32) -> &'static [Vec2] {
    use std::sync::OnceLock;
    fn polygon<const N: usize>(cache: &'static OnceLock<[Vec2; N]>) -> &'static [Vec2] {
        cache.get_or_init(|| {
            std::array::from_fn(|i| {
                let t = TAU * i as f32 / N as f32;
                Vec2::new(t.cos(), t.sin())
            })
        })
    }
    static EIGHT: OnceLock<[Vec2; 8]> = OnceLock::new();
    static SIXTEEN: OnceLock<[Vec2; 16]> = OnceLock::new();
    static WALL: OnceLock<[Vec2; 24]> = OnceLock::new();
    match n {
        8 => polygon(&EIGHT),
        16 => polygon(&SIXTEEN),
        _ => polygon(&WALL),
    }
}

// One closed strip from a square lip to a circular bowl (or the inscribed
// patch inside the bowl). Every octant includes the exact square corner.
fn ring(
    g: &mut Geometry,
    center: Vec2,
    r: f32,
    square: f32,
    depth: [f32; 2],
    outside: bool,
    n: u32,
) {
    let dirs = directions(n);
    for i in 0..dirs.len() {
        let j = (i + 1) % dirs.len();
        let point = |d: Vec2, rect: bool, z: f32| {
            (center
                + d * if rect {
                    square / d.abs().max_element()
                } else {
                    r
                })
            .extend(z)
        };
        let outer = |d| point(d, outside, depth[0]);
        let inner = |d| point(d, !outside, depth[1]);
        g.metric_quad(
            [
                outer(dirs[i]),
                outer(dirs[j]),
                inner(dirs[j]),
                inner(dirs[i]),
            ],
            Vec3::Z,
        );
    }
}

// Ground contact with an arched crown and a flat base. The enclosing patch
// meets the tiled face exactly. Only the bore sees its recessed dark backing.
fn ground_contact(a: &mut Assembly, b: &Contact, label: &str, center: Vec2) {
    let r = 0.00255 * b.contact_scale;
    let half = 0.0032 * b.contact_scale;
    let depth = 0.0017 * b.contact_scale;
    let top = b.face_z_m;
    let dirs = directions(b.pin_sides);
    let contour = |d: Vec2| {
        center
            + if d.y >= 0. {
                d * r
            } else {
                d * (r / d.abs().max_element())
            }
    };
    let outer = |d: Vec2| (center + d * (half / d.abs().max_element())).extend(top);
    for i in 0..dirs.len() {
        let (d, e) = (dirs[i], dirs[(i + 1) % dirs.len()]);
        let (p, q) = (contour(d), contour(e));
        a.part(b.insert, label)
            .metric_quad([outer(d), outer(e), q.extend(top), p.extend(top)], Vec3::Z);
        a.part(Surface::Ink, label).metric_quad(
            [
                p.extend(top),
                q.extend(top),
                q.extend(top - depth),
                p.extend(top - depth),
            ],
            (-d).extend(0.),
        );
    }
    Rect::new(center, Vec2::splat(half * 2.)).face(a.part(Surface::Ink, label), top - depth);
}

fn round_well(a: &mut Assembly, b: &Contact, label: &str, c: Vec2) {
    let k = b.contact_scale;
    let half = 0.021 * k;
    let r = 0.0199 * k;
    let well = b.face_z_m - b.well_depth_m;
    let g = a.part(b.insert, label);
    ring(g, c, r, half, [b.face_z_m, b.face_z_m], true, b.well_sides);
    // Profile begins at the inner back edge and turns out towards the lip.
    // Its wound inward face forms the visible cup.
    let tf =
        Transform::from_translation(c.extend(well)).with_rotation(Quat::from_rotation_x(FRAC_PI_2));
    g.lathe(&[(r, b.well_depth_m), (r * 0.92, 0.)], b.well_sides, tf);
    ring(
        g,
        c,
        r * 0.92,
        0.0128 * k,
        [well, well],
        false,
        b.well_sides,
    );
    let patch = Rect::new(c, Vec2::splat(0.0256 * k));
    let hole_half = 0.0030 * k;
    let centers = [c - Vec2::X * 0.0093 * k, c + Vec2::X * 0.0093 * k];
    let holes = centers.map(|p| Rect::new(p, Vec2::splat(hole_half * 2.)));
    pierced(a.part(b.insert, label), patch, well, &holes);
    for center in centers {
        let radius = 0.00255 * k;
        let depth = 0.0016 * k;
        ring(
            a.part(b.insert, label),
            center,
            radius,
            hole_half,
            [well, well],
            true,
            b.pin_sides,
        );
        let tf = Transform::from_translation(center.extend(well - depth))
            .with_rotation(Quat::from_rotation_x(FRAC_PI_2));
        a.part(Surface::Ink, label).lathe(
            &[(radius, depth), (radius, 0.), (0., 0.)],
            b.pin_sides,
            tf,
        );
    }
    for sign in [-1., 1.] {
        let tab = Vec3::new(c.x, c.y + sign * r * 0.927, well + b.well_depth_m * 0.60);
        a.box_part(
            Surface::Metal,
            label,
            tab,
            Vec3::new(0.005 * k, r * 0.10, b.well_depth_m * 0.42),
            0.00045,
        );
    }
}

/// One uninterrupted panel around the mains contacts and optional USB opening.
pub(super) fn insert(
    a: &mut Assembly,
    b: &Contact,
    label: &str,
    body: Rect,
    centers: &[Vec2],
    usb: Option<Rect>,
) {
    let k = b.contact_scale;
    let mut pins = Vec::with_capacity(6);
    let mut cup = None;
    let mut grounds = [Vec2::ZERO; 2];
    for (&c, ground) in centers.iter().zip(&mut grounds) {
        match b.face {
            OutletFace::RoundWell => cup = Some(c),
            OutletFace::ThreePin => {
                for sign in [-1., 1.] {
                    pins.push(Rect::new(
                        c + Vec2::new(sign * 0.0111, -0.004) * k,
                        Vec2::new(0.0069, 0.0043) * k,
                    ));
                }
                pins.push(Rect::new(
                    c + Vec2::Y * 0.010 * k,
                    Vec2::new(0.0044, 0.0086) * k,
                ));
            }
            OutletFace::Polarized => {
                for (sign, len) in [(-1., 0.0082), (1., 0.0067)] {
                    pins.push(Rect::new(
                        c + Vec2::new(sign * 0.00635, 0.003) * k,
                        Vec2::new(0.0023, len) * k,
                    ));
                }
                *ground = c - Vec2::Y * 0.0090 * k;
            }
        }
    }
    pins.extend(usb);
    let mut cuts = pins.clone();
    if b.face == OutletFace::Polarized {
        cuts.extend(
            grounds[..centers.len()]
                .iter()
                .map(|&p| Rect::new(p, Vec2::splat(0.0064 * k))),
        );
    }
    if let Some(c) = cup {
        cuts.push(Rect::new(c, Vec2::splat(0.042 * k)));
    }
    cuts.sort_unstable_by(|a, b| a.lo.x.total_cmp(&b.lo.x));
    pierced(a.part(b.insert, label), body, b.face_z_m, &cuts);
    for pin in pins {
        slot(a, label, pin, b.face_z_m, 0.0017 * k);
    }
    if b.face == OutletFace::Polarized {
        for &ground in &grounds[..centers.len()] {
            ground_contact(a, b, label, ground);
        }
    }
    if let Some(c) = cup {
        round_well(a, b, label, c);
    }
}
