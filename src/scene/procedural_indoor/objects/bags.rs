//! Closed, dimensioned personal luggage. Body profiles, closure, carrying straps
//! and pockets vary together; all parts retain the Bag semantic/instance owner.
use super::*;
mod details;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum BagStyle {
    Briefcase,
    Backpack,
    Tote,
    Satchel,
    Duffel,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BagProgram {
    pub style: BagStyle,
    pub body_height: f32,
    pub taper: f32,
    pub roundness: f32,
    pub fullness: f32,
    pub handle_span: f32,
    pub handle_drop: f32,
    pub strap_width: f32,
    pub pocket_width: f32,
    pub pocket_height: f32,
    pub pocket_depth: f32,
    pub flap: bool,
    pub seam: bool,
    pub finish: Surface,
}

pub fn parameters(o: &IndoorObject) -> BagProgram {
    let mut r = stream(o.seed, 7431);
    let style = [
        BagStyle::Briefcase,
        BagStyle::Backpack,
        BagStyle::Tote,
        BagStyle::Satchel,
        BagStyle::Duffel,
    ][r.random_range(0..5)];
    let (body_height, roundness, drop) = match style {
        BagStyle::Briefcase => (
            r.random_range(0.66..0.79),
            r.random_range(5.0..10.0),
            r.random_range(0.08..0.15),
        ),
        BagStyle::Backpack => (
            r.random_range(0.86..0.94),
            r.random_range(2.5..4.8),
            r.random_range(0.03..0.055),
        ),
        BagStyle::Tote => (
            r.random_range(0.52..0.70),
            r.random_range(2.8..5.0),
            r.random_range(0.20..0.28),
        ),
        BagStyle::Satchel => (
            r.random_range(0.55..0.72),
            r.random_range(3.5..7.0),
            r.random_range(0.22..0.26),
        ),
        BagStyle::Duffel => (
            r.random_range(0.52..0.67),
            r.random_range(2.0..3.3),
            r.random_range(0.20..0.27),
        ),
    };
    BagProgram {
        style,
        body_height,
        roundness,
        taper: r.random_range(0.70..0.96),
        fullness: r.random_range(0.72..0.95),
        handle_span: r.random_range(0.24..0.52),
        handle_drop: drop,
        strap_width: r.random_range(0.016..0.033),
        pocket_width: r.random_range(0.40..0.72),
        pocket_height: r.random_range(0.28..0.49),
        pocket_depth: r.random_range(0.04..0.12),
        flap: r.random_bool(0.5),
        seam: r.random_bool(0.65),
        finish: match style {
            BagStyle::Backpack => {
                if r.random_bool(0.80) {
                    Surface::FabricAlt
                } else {
                    Surface::Leather
                }
            }
            BagStyle::Tote | BagStyle::Duffel => {
                if r.random_bool(0.72) {
                    Surface::Fabric
                } else {
                    Surface::Leather
                }
            }
            _ => {
                if r.random_bool(0.72) {
                    Surface::Leather
                } else {
                    Surface::FabricAlt
                }
            }
        },
    }
}

impl BagProgram {
    fn rings(&self, s: Vec3) -> [(f32, Vec2); 6] {
        let w = s.x
            * if self.style == BagStyle::Backpack {
                0.38
            } else {
                0.475
            };
        let d = s.z * 0.37 * self.fullness;
        let shape: [(f32, f32, f32); 6] = match self.style {
            BagStyle::Briefcase => [
                (0., 0.92, 0.90),
                (0.06, 1., 1.),
                (0.40, 1., 1.),
                (0.84, 1., 1.),
                (0.94, 0.99, 0.99),
                (1., 0.91, 0.87),
            ],
            BagStyle::Backpack => [
                (0., 0.78, 0.70),
                (0.08, 0.96, 0.98),
                (0.38, 1., 1.),
                (0.70, 0.95, 0.91),
                (0.90, 0.74, 0.73),
                (1., 0.42, 0.38),
            ],
            BagStyle::Tote => [
                (0., self.taper, 0.75),
                (0.08, self.taper + 0.03, 0.90),
                (0.36, 0.91, 1.),
                (0.70, 0.97, 0.94),
                (0.93, 1., 0.74),
                (1., 0.99, 0.58),
            ],
            BagStyle::Satchel => [
                (0., 0.88, 0.78),
                (0.09, 0.99, 0.98),
                (0.37, 1., 1.),
                (0.74, 0.99, 0.89),
                (0.94, self.taper, 0.79),
                (1., self.taper * 0.97, 0.70),
            ],
            BagStyle::Duffel => [
                (0., 0.82, 0.62),
                (0.10, 0.95, 0.87),
                (0.38, 1., 1.),
                (0.70, 0.97, 0.90),
                (0.92, 0.84, 0.60),
                (1., 0.74, 0.22),
            ],
        };
        shape.map(|(y, x, z)| (y * s.y * self.body_height, Vec2::new(x * w, z * d)))
    }

    /// Analytic cross-section of the same piecewise loft used by the mesh.
    /// Pockets, seams and strap anchors follow this surface rather than float.
    fn front(&self, s: Vec3, x: f32, y: f32) -> f32 {
        let rings = self.rings(s);
        let pair = rings
            .windows(2)
            .find(|v| y <= v[1].0)
            .unwrap_or(&rings[4..]);
        let t = ((y - pair[0].0) / (pair[1].0 - pair[0].0)).clamp(0., 1.);
        let half = pair[0].1.lerp(pair[1].1, t);
        half.y
            * (1. - (x.abs() / half.x).min(0.999).powf(self.roundness))
                .max(0.)
                .powf(1. / self.roundness)
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    let h = s.y * p.body_height;
    a.part(p.finish, l)
        .housing(&p.rings(s), p.roundness, Transform::IDENTITY);
    let trim = if p.finish == Surface::Leather {
        Surface::Leather
    } else {
        Surface::FabricAlt
    };
    match p.style {
        BagStyle::Backpack => {
            // A rounded front pocket and a second optional small zipped pocket.
            details::pocket(
                a,
                &p,
                s,
                Vec2::new(0., h * 0.35),
                Vec2::new(s.x * p.pocket_width * 0.85, h * p.pocket_height),
                s.z * p.pocket_depth,
                trim,
                l,
            );
            if p.flap {
                details::pocket(
                    a,
                    &p,
                    s,
                    Vec2::new(0., h * 0.69),
                    Vec2::new(s.x * 0.40, h * 0.16),
                    s.z * 0.045,
                    p.finish,
                    l,
                );
            }
            // Paired flat padded shoulder straps, attached at both ends, behind
            // the pack. They reserve their sag inside the advertised envelope.
            for side in [-1., 1.] {
                let pts: Vec<_> = (0..=12)
                    .map(|i| {
                        let t = i as f32 / 12.;
                        let y = h * (0.15 + 0.66 * t);
                        let x = side * s.x * (0.20 - 0.08 * t);
                        Vec3::new(
                            x,
                            y,
                            -p.front(s, x, y) - s.z * 0.11 * (PI * t).sin() - 0.001,
                        )
                    })
                    .collect();
                details::strap(
                    a.part(trim, l),
                    &pts,
                    p.strap_width.min(s.x * 0.12) * 1.25,
                    0.003,
                    Vec3::Z,
                );
            }
            details::handle(
                a,
                &p,
                s,
                0.,
                h * 0.965,
                s.x * 0.19,
                s.y * p.handle_drop,
                trim,
                l,
            );
        }
        BagStyle::Briefcase => {
            details::handle(
                a,
                &p,
                s,
                0.,
                h * 0.98,
                s.x * p.handle_span,
                s.y * p.handle_drop,
                trim,
                l,
            );
            if p.flap {
                details::pocket(
                    a,
                    &p,
                    s,
                    Vec2::new(0., h * 0.48),
                    Vec2::new(s.x * p.pocket_width, h * p.pocket_height),
                    s.z * 0.045,
                    p.finish,
                    l,
                );
            }
            // Hinged cases have clasps; soft briefcases have a zipper instead.
            if p.roundness > 7.0 {
                for side in [-1., 1.] {
                    let x = side * s.x * 0.27;
                    let y = h * 0.83;
                    a.box_part(
                        Surface::Metal,
                        l,
                        Vec3::new(x, y, p.front(s, x, y) + 0.002),
                        Vec3::new(s.x * 0.075, h * 0.085, 0.005),
                        0.001,
                    );
                }
            }
        }
        BagStyle::Tote | BagStyle::Duffel => {
            for side in [-1., 1.] {
                let y = h * if p.style == BagStyle::Tote {
                    0.92
                } else {
                    0.72
                };
                let z = side * p.front(s, s.x * p.handle_span * 0.5, y);
                details::handle(
                    a,
                    &p,
                    s,
                    z,
                    y,
                    s.x * p.handle_span,
                    s.y * p.handle_drop,
                    trim,
                    l,
                );
            }
            if p.flap && p.style == BagStyle::Tote {
                details::pocket(
                    a,
                    &p,
                    s,
                    Vec2::new(s.x * 0.04, h * 0.45),
                    Vec2::new(s.x * p.pocket_width, h * p.pocket_height),
                    s.z * p.pocket_depth * 0.5,
                    trim,
                    l,
                );
            }
        }
        BagStyle::Satchel => {
            // A shaped flap overlays the front; shoulder strap closes at the
            // upper side panels instead of dangling disconnected in midair.
            details::pocket(
                a,
                &p,
                s,
                Vec2::new(0., h * 0.72),
                Vec2::new(s.x * p.taper * 0.88, h * 0.43),
                s.z * 0.035,
                trim,
                l,
            );
            details::handle(
                a,
                &p,
                s,
                0.,
                h * 0.95,
                s.x * p.taper * 0.87,
                s.y * p.handle_drop,
                trim,
                l,
            );
            let count = if p.flap { 2 } else { 1 };
            for i in 0..count {
                let x = (i as f32 - (count - 1) as f32 * 0.5) * s.x * 0.42;
                let y = h * 0.55;
                a.box_part(
                    Surface::Metal,
                    l,
                    Vec3::new(x, y, p.front(s, x, y) + s.z * 0.022),
                    Vec3::new(s.x * 0.07, h * 0.065, 0.005),
                    0.001,
                );
            }
        }
    }
    details::closure(a, &p, s, l);
}
