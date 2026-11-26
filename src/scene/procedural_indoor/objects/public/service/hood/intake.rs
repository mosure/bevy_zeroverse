//! Replaceable grease filters and a separate work-light rail. All dimensions
//! are local metres. The canopy underside closes the plenum above the openings.
use super::*;
use crate::scene::procedural_indoor::geometry::{housing_outline, Geometry};

pub(super) const ROOF: f32 = 0.037;
const FASCIA: f32 = 0.008;
const FILTER_FACE: f32 = 0.012;
const LENS_FACE: f32 = 0.003;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
#[serde(tag = "style", rename_all = "snake_case")]
pub enum FilterStyle {
    FoldedBaffle {
        across_x: bool,
        folds: usize,
        pitch_m: f32,
        width_m: f32,
        depth_m: f32,
    },
    PerimeterPanel {
        gap_m: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LampStyle {
    Puck,
    Linear,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct FilterDeck {
    pub center: Vec2,
    pub size: Vec2,
    pub aperture_center: Vec2,
    pub aperture_size: Vec2,
    pub cassettes: usize,
    pub filter: FilterStyle,
    pub rim_m: f32,
    pub lamp: LampStyle,
    pub lamps: usize,
    pub lens_size: Vec2,
    pub lamp_rail_z: f32,
}

impl FilterDeck {
    pub(crate) fn installed(o: &IndoorObject) -> Self {
        Self::fit(o, &hood_parameters(o))
    }

    pub fn fit(o: &IndoorObject, p: &HoodProgram) -> Self {
        let mut cut = stream(o.seed, u64::from_le_bytes(*b"HoodDeck"));
        let (flue_center, flue_size) = p.chimney(o.size);
        // The rectangular cassette bay must fit the opaque motor, including
        // inside an elliptical glass wing or a rounded bell.
        let (center, size) = match p.construction {
            2 => (flue_center.xz(), flue_size * 1.13),
            3 => (Vec2::ZERO, o.size.xz() * Vec2::new(0.55, 0.46)),
            _ => (Vec2::ZERO, o.size.xz() * Vec2::new(0.86, 0.77)),
        };
        let edge = cut
            .random_range(0.009_f32..0.019)
            .min(size.min_element() * 0.065);
        let band = (size.y * cut.random_range(0.17..0.26))
            .clamp(0.035, 0.075)
            .min(size.y * 0.34);
        let aperture_size = size - Vec2::new(2. * edge, edge + band);
        let aperture_center = center - Vec2::Y * ((band - edge) * 0.5);
        let cassettes = p
            .filters
            .min((aperture_size.x / 0.115).floor().max(1.) as usize);
        let bay_width = aperture_size.x / cassettes as f32;
        let rim_m = cut
            .random_range(0.004_f32..0.009)
            .min(bay_width.min(aperture_size.y) * 0.07);
        let inner = Vec2::new(bay_width, aperture_size.y) - Vec2::splat(2. * rim_m);
        let filter = if cut.random_bool(if p.construction == 2 { 0.48 } else { 0.69 }) {
            let across_x = cut.random_bool(0.62);
            let span = if across_x { inner.x } else { inner.y };
            let folds = (span / cut.random_range(0.022..0.040))
                .round()
                .clamp(2., 18.) as usize;
            let pitch_m = span / folds as f32;
            FilterStyle::FoldedBaffle {
                across_x,
                folds,
                pitch_m,
                width_m: pitch_m * cut.random_range(0.51..0.72),
                depth_m: cut.random_range(0.005..0.012),
            }
        } else {
            FilterStyle::PerimeterPanel {
                gap_m: cut
                    .random_range(0.006_f32..0.017)
                    .min(inner.min_element() * 0.12),
            }
        };
        let mut optics = stream(o.seed, u64::from_le_bytes(*b"HoodLens"));
        let lamp = if optics.random_bool(0.45) {
            LampStyle::Linear
        } else {
            LampStyle::Puck
        };
        let lamps = if lamp == LampStyle::Linear {
            p.lights.min(2)
        } else {
            p.lights
        };
        let cell = (size.x - 2. * edge) / lamps as f32;
        let lens_size = match lamp {
            LampStyle::Puck => {
                Vec2::splat((band * optics.random_range(0.44..0.72)).min(cell * 0.65))
            }
            LampStyle::Linear => Vec2::new(
                cell * optics.random_range(0.47..0.85),
                band * optics.random_range(0.24..0.42),
            ),
        };
        Self {
            center,
            size,
            aperture_center,
            aperture_size,
            cassettes,
            filter,
            rim_m,
            lamp,
            lamps,
            lens_size,
            lamp_rail_z: center.y + (size.y - band) * 0.5,
        }
    }

    /// Centre of the exposed lens on its room-facing surface.
    pub fn lens_center(&self, i: usize) -> Vec3 {
        let x = self.center.x + self.aperture_size.x * ((i as f32 + 0.5) / self.lamps as f32 - 0.5);
        Vec3::new(x, LENS_FACE, self.lamp_rail_z)
    }

    fn write_lens(&self, g: &mut Geometry, i: usize) {
        let c = self.lens_center(i);
        match self.lamp {
            LampStyle::Puck => g.lathe(
                &[(0., 0.), (self.lens_size.x * 0.5, 0.)],
                16,
                Transform::from_translation(c),
            ),
            LampStyle::Linear => {
                g.metric_quad(rect(c.xz(), self.lens_size, c.y), Vec3::NEG_Y);
            }
        }
    }

    pub(crate) fn reflection_aperture(&self) -> Geometry {
        let mut g = Geometry::default();
        for i in 0..self.lamps {
            self.write_lens(&mut g, i);
        }
        g
    }

    /// Installed, outward-facing area including every lens on this rail.
    pub fn emitting_area_m2(&self) -> f32 {
        let one = match self.lamp {
            LampStyle::Puck => {
                let r = self.lens_size.x * 0.5;
                8. * (std::f32::consts::TAU / 16.).sin() * r * r
            }
            LampStyle::Linear => self.lens_size.x * self.lens_size.y,
        };
        one * self.lamps as f32
    }

    fn bay(&self, i: usize) -> (Vec2, Vec2) {
        let size = Vec2::new(
            self.aperture_size.x / self.cassettes as f32,
            self.aperture_size.y,
        );
        let center = self.aperture_center
            + Vec2::X * (size.x * (i as f32 + 0.5) - self.aperture_size.x * 0.5);
        (center, size)
    }

    pub(super) fn build(&self, a: &mut Assembly, o: &IndoorObject) {
        let label = o.kind.class_name();
        let g = a.part(Surface::Metal, label);
        let fascia = rect(self.center, self.size, FASCIA);
        let opening = rect(self.aperture_center, self.aperture_size, FASCIA);
        band(g, rect(self.center, self.size, ROOF), fascia);
        band(g, fascia, opening);
        band(
            g,
            opening,
            rect(self.aperture_center, self.aperture_size, ROOF),
        );
        for i in 0..self.cassettes {
            let (center, size) = self.bay(i);
            let inner = size - Vec2::splat(2. * self.rim_m);
            // Neighbouring cassettes meet edge to edge. The outer fold is seated
            // inside the reveal; the inner fold holds the ends of each baffle.
            let lip = rect(center, inner, FILTER_FACE);
            band(g, rect(center, size, FILTER_FACE), lip);
            band(g, lip, rect(center, inner, FILTER_FACE + 0.015));
            match self.filter {
                FilterStyle::FoldedBaffle {
                    across_x,
                    folds,
                    pitch_m,
                    width_m,
                    depth_m,
                } => {
                    let (cross, run) = if across_x {
                        (Vec2::X, Vec2::Y)
                    } else {
                        (Vec2::Y, Vec2::X)
                    };
                    let half_run = inner.dot(run) * 0.5 + self.rim_m * 0.36;
                    for j in 0..folds {
                        let station = (j as f32 + 0.5 - folds as f32 * 0.5) * pitch_m;
                        folded_channel(
                            g,
                            center + cross * station,
                            cross * width_m,
                            run * half_run,
                            depth_m,
                        );
                    }
                }
                FilterStyle::PerimeterPanel { gap_m } => {
                    let panel = inner - Vec2::splat(2. * gap_m);
                    let face = FILTER_FACE + 0.0008;
                    g.cuboid(
                        Vec3::new(panel.x, 0.005, panel.y),
                        0.0015,
                        Transform::from_xyz(center.x, face + 0.0025, center.y),
                    );
                    for sign in [-1., 1.] {
                        let top = face + 0.005;
                        g.cuboid(
                            Vec3::new(panel.x * 0.15, ROOF - top, panel.y * 0.30),
                            0.,
                            Transform::from_xyz(
                                center.x + sign * panel.x * 0.30,
                                (top + ROOF) * 0.5,
                                center.y,
                            ),
                        );
                    }
                }
            }
        }
        for i in 0..self.lamps {
            let c = self.lens_center(i);
            let half = self.lens_size * 0.5;
            let trim = 0.0025_f32.min(half.min_element() * 0.30);
            match self.lamp {
                LampStyle::Puck => {
                    let r = half.x;
                    // Open bezel and a recessed diffuser. Neither has a cap in
                    // front of the lens. Their shared edge needs no depth offset.
                    let tf = Transform::from_translation(c);
                    a.part(Surface::Chrome, label).lathe(
                        &[
                            (r, 0.),
                            (r, -0.0008),
                            (r + trim, -0.0008),
                            (r + trim, FASCIA - LENS_FACE),
                        ],
                        16,
                        tf,
                    );
                }
                LampStyle::Linear => {
                    let center = c.xz();
                    let outer = self.lens_size + Vec2::splat(2. * trim);
                    let bezel = a.part(Surface::Metal, label);
                    band(bezel, rect(center, outer, FASCIA), rect(center, outer, c.y));
                    band(
                        bezel,
                        rect(center, outer, c.y),
                        rect(center, self.lens_size, c.y),
                    );
                }
            }
            self.write_lens(a.part(Surface::Light, "lamp"), i);
        }
    }
}

// The folded channel terminates inside the two frame rails. Its creases and
// intervening air gaps remain visible from below and from oblique viewpoints.
fn folded_channel(g: &mut Geometry, center: Vec2, width: Vec2, half_run: Vec2, depth: f32) {
    let profile = [(-0.50, 1.), (-0.23, 0.), (0.23, 0.), (0.50, 1.)];
    let point = |i: usize, end: f32| {
        let (u, y) = profile[i];
        let p = center + u * width + end * half_run;
        Vec3::new(p.x, FILTER_FACE + 0.001 + y * depth, p.y)
    };
    for i in 0..4 {
        let j = (i + 1) % 4;
        let outward = if i == 3 { Vec3::Y } else { Vec3::NEG_Y };
        g.metric_quad(
            [point(i, -1.), point(j, -1.), point(j, 1.), point(i, 1.)],
            outward,
        );
    }
}

fn rect(c: Vec2, size: Vec2, y: f32) -> [Vec3; 4] {
    [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)]
        .map(|(x, z)| Vec3::new(c.x + size.x * 0.5 * x, y, c.y + size.y * 0.5 * z))
}

fn band<const N: usize>(g: &mut Geometry, a: [Vec3; N], b: [Vec3; N]) {
    for i in 0..N {
        let j = (i + 1) % N;
        let side = [a[i], a[j], b[j], b[i]];
        g.metric_quad(side, (side[1] - side[0]).cross(side[2] - side[0]));
    }
}

pub(super) fn glass_wing(a: &mut Assembly, o: &IndoorObject, center: Vec2, flue: Vec2, base: f32) {
    let g = a.part(Surface::Glass, o.kind.class_name());
    let outline = housing_outline(2.6);
    let perimeter: [Vec2; 16] = std::array::from_fn(|i| outline[i * 2] * o.size.xz() * 0.49);
    // Connect both contours along rays from the offset motor. Matching angular
    // indices of two independently centred outlines can produce crossed faces.
    let hole = perimeter.map(|outer| {
        let ray = outer - center;
        let rectangle_radius = (ray.abs() / (flue * 0.58)).max_element().recip();
        center + ray * rectangle_radius.min(0.96)
    });
    let ring = |points: [Vec2; 16], y| points.map(|p| Vec3::new(p.x, y, p.y));
    let lo = base + 0.009;
    let hi = lo + (o.size.z * 0.018).clamp(0.006, 0.011);
    band(g, ring(perimeter, hi), ring(perimeter, lo));
    band(g, ring(perimeter, lo), ring(hole, lo));
    band(g, ring(hole, lo), ring(hole, hi));
    band(g, ring(hole, hi), ring(perimeter, hi));
}
