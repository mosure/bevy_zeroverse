//! The optical surface and its mount share a contour in the eye-bone frame.
//! Local -Z is forward. Wrapping and tilting the lens recess its outer/lower edge.
use super::{EyewearProgram, Rim};
use crate::scene::procedural_indoor::humans::face::projection::Patch;
use bevy::prelude::*;

pub(super) const SAMPLES: usize = 32;
pub(super) struct Lens {
    pub centre: Vec3,
    pub side: f32,
    pub offsets: [Vec2; SAMPLES],
    curvature: f32,
    slope: Vec2,
}

impl Lens {
    fn new(p: &EyewearProgram, eye: Vec3, scale: f32) -> Self {
        let side = eye.x.signum();
        let cx = eye.x + side * p.decentration_m * scale;
        let width = (cx.abs() - p.bridge_width_m * scale * 0.5).clamp(0.017 * scale, 0.034 * scale);
        let half_height = p.lens_height_m * scale * 0.5;
        let offsets = std::array::from_fn(|i| {
            let (s, c) = (i as f32 * std::f32::consts::TAU / SAMPLES as f32).sin_cos();
            let shape = |v: f32| v.signum() * v.abs().powf(p.corner_exponent);
            let u = shape(c);
            let v = shape(s);
            Vec2::new(
                side * width * u * (1. + p.lower_taper * v.min(0.)),
                half_height * v + p.upsweep_m * scale * u.max(0.).powi(2),
            )
        });
        Self {
            centre: Vec3::new(cx, eye.y, 0.),
            side,
            offsets,
            curvature: (p.lens_curve_radius_m * scale).recip(),
            slope: Vec2::new(side * p.wrap, -p.tilt),
        }
    }
    pub fn point(&self, offset: Vec2) -> Vec3 {
        self.centre
            + offset.extend(self.slope.dot(offset) + 0.5 * self.curvature * offset.length_squared())
    }
    pub fn normal(&self, offset: Vec2) -> Vec3 {
        (self.slope + self.curvature * offset)
            .extend(-1.)
            .normalize()
    }
    pub fn mount(&self, i: usize, rim: Rim) -> Vec3 {
        self.point(self.offsets[i] * if rim == Rim::DrillMounted { 0.84 } else { 1. })
    }
}

pub(in super::super) struct Fit<'a> {
    pub(super) program: &'a EyewearProgram,
    pub(super) lenses: [Lens; 2],
    pub(super) scale: f32,
}
impl<'a> Fit<'a> {
    pub(in super::super) fn new(
        p: &'a EyewearProgram,
        mut eyes: [Vec3; 2],
        scale: f32,
        skin: &Patch,
    ) -> Self {
        eyes.sort_by(|a, b| a.x.total_cmp(&b.x));
        let mut lenses = eyes.map(|eye| Lens::new(p, eye, scale));
        let mut datum = f32::INFINITY;
        for (lens, eye) in lenses.iter().zip(eyes) {
            let apex = skin
                .front(eye.truncate())
                .map_or(eye.z - 0.012 * scale, |s| s.0);
            datum = datum.min(
                apex - p.cornea_clearance_m * scale
                    - lens.point(eye.truncate() - lens.centre.truncate()).z,
            );
            // Sample both the edges and the cheeks/nasal surface *inside* the
            // optical area. A single bridge-depth sample cannot fit a wrapped pair.
            for offset in lens.offsets {
                for fraction in [0.46, 0.78, 1.] {
                    let pnt = lens.point(offset * fraction);
                    if let Some((z, _)) = skin.front(pnt.truncate()) {
                        let margin = if fraction == 1. {
                            0.0015 + p.rim_radius_m * (1. + p.brow_weight)
                        } else {
                            0.0015
                        };
                        datum = datum.min(z - margin * scale - pnt.z);
                    }
                }
            }
        }
        for lens in &mut lenses {
            lens.centre.z = datum;
        }
        Self {
            program: p,
            lenses,
            scale,
        }
    }
    pub(super) fn bridge(&self, skin: &Patch) -> [Vec3; 9] {
        let [a, b] = self
            .lenses
            .each_ref()
            .map(|l| l.mount(14, self.program.rim));
        std::array::from_fn(|i| {
            let t = i as f32 / 8.;
            let arch = 4. * t * (1. - t);
            let mut p = a.lerp(b, t)
                + Vec3::new(0., self.program.bridge_rise_m, -0.0022) * self.scale * arch;
            if i != 0 && i != 8 {
                if let Some((z, _)) = skin.front(p.truncate()) {
                    p.z =
                        p.z.min(z - (self.program.rim_radius_m + 0.0012) * self.scale);
                }
            }
            p
        })
    }
}

// Mirrored fullbody_default vertices on the temporal skin, the superior root
// of the pinna and the fold behind the ear. The arm bends behind the pinna;
// using the outermost point of the ear would push the temple away from the head.
const EAR_VERTICES: [[usize; 7]; 2] = [
    [7425, 7559, 12330, 12323, 12316, 7565, 7620],
    [718, 870, 5749, 5742, 5735, 876, 935],
];
const NOMINAL: [[f32; 3]; 7] = [
    [0.065, -0.001, 0.022],
    [0.068, -0.001, 0.043],
    [0.072, 0.001, 0.059],
    [0.072, 0.0015, 0.076],
    [0.073, -0.0045, 0.086],
    [0.070, -0.014, 0.092],
    [0.065, -0.027, 0.095],
];
pub(super) fn temple_path(
    l: &Lens,
    p: &EyewearProgram,
    scale: f32,
    posed: &[Vec3],
    origin: Vec3,
    rotation: Quat,
    temple_skin: &Patch,
) -> [Vec3; 10] {
    let side = l.side;
    let ids = EAR_VERTICES[usize::from(side > 0.)];
    let inv = rotation.inverse();
    let landmarks: [Vec3; 7] = std::array::from_fn(|i| {
        let v = Vec3::from_array(NOMINAL[i]) * Vec3::new(side, 1., 1.) * scale;
        // A differently indexed externally supplied mesh can still use the
        // nominal fit. Never route a temple to a remote or opposite-side vertex.
        posed
            .get(ids[i])
            .map(|p| inv * (*p - origin))
            .filter(|p| p.distance(v) < 0.030 * scale)
            .unwrap_or(v)
    });
    let mount = l.mount(2, p.rim);
    let radial = Vec3::X * side;
    let clearance = p.rim_radius_m + 0.0016;
    let surface =
        |i: usize, lift: f32| landmarks[i] + (radial * clearance + Vec3::Y * lift) * scale;
    let outer = l.point(l.offsets[2]) + radial * (0.003 * scale);
    let hinge = outer + Vec3::Z * (0.002 * scale);
    let cheek = surface(0, 0.0035);
    let mut route = [
        mount,
        hinge,
        hinge.lerp(cheek, 0.36),
        hinge.lerp(cheek, 0.70),
        cheek,
        surface(1, 0.0040),
        surface(2, 0.0020),
        landmarks[3] + (radial * 0.0015 + Vec3::Y * (p.rim_radius_m + 0.0004)) * scale,
        surface(4, 0.001),
        landmarks[5].lerp(landmarks[6], p.ear_hook) + radial * ((p.rim_radius_m + 0.0006) * scale),
    ];
    // The orbital rim can bulge out between the hinge and the ear. Keep the
    // whole anterior arm outside that surface while retaining its drilled mount.
    for i in 1..6 {
        let mut offset = 0.0_f32;
        for t in [0., 0.25, 0.5, 0.75, 1.] {
            let v = route[i].lerp(route[i + 1], t);
            if let Some((depth, _)) = temple_skin.front(v.zy()) {
                offset = offset.max(-depth + clearance * scale - side * v.x);
            }
        }
        route[i] += radial * offset;
        route[i + 1] += radial * offset;
    }
    route
}
