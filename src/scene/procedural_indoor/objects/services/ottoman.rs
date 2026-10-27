//! Upholstered footrests and poufs with fitted load-bearing bases. Dimensions,
//! construction and finish draws are independent; all parts share one OBB/class.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
pub enum Base {
    Feet,
    Plinth,
    Sled,
    Upholstered,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct OttomanProgram {
    pub base: Base,
    pub base_height: f32,
    pub base_span: Vec2,
    pub rail: f32,
    pub roundness: f32,
    pub crown: f32,
    /// Zero means a single upholstered volume, otherwise a separate top cushion.
    pub cap_fraction: f32,
    pub body_inset: f32,
    pub body: Surface,
    pub top: Surface,
    pub frame: Surface,
}

pub fn dimensions(r: &mut impl Rng) -> Vec3 {
    let width = r.random_range(0.40..0.97);
    let elongated = r.random_bool(0.27);
    Vec3::new(
        width,
        r.random_range(0.30..0.50),
        if elongated {
            width * r.random_range(0.55..0.79)
        } else {
            width * r.random_range(0.87..1.12)
        },
    )
}

pub fn parameters(o: &IndoorObject) -> OttomanProgram {
    let mut r = stream(o.seed, 0x4f54_544f_5348_4150);
    let mut finish = stream(o.finish_seed.unwrap_or(o.seed), 0x4f54_544f_4649_4e49);
    let base = [Base::Feet, Base::Plinth, Base::Sled, Base::Upholstered][r.random_range(0..4)];
    let base_height = o.size.y
        * match base {
            Base::Feet | Base::Sled => r.random_range(0.17..0.36),
            Base::Plinth => r.random_range(0.08..0.22),
            Base::Upholstered => r.random_range(0.018..0.04),
        };
    let body = [Surface::Fabric, Surface::FabricAlt, Surface::Leather][finish.random_range(0..3)];
    let cap_fraction = if r.random_bool(0.57) {
        r.random_range(0.22..0.43)
    } else {
        0.
    };
    let top = if cap_fraction > 0. && finish.random_bool(0.32) {
        [Surface::Fabric, Surface::FabricAlt, Surface::Leather][finish.random_range(0..3)]
    } else {
        body
    };
    OttomanProgram {
        base,
        base_height,
        base_span: Vec2::new(r.random_range(0.26..0.34), r.random_range(0.24..0.33)),
        rail: o.size.x.min(o.size.z).min(o.size.y) * r.random_range(0.045..0.080),
        roundness: r.random_range(2.0..8.5),
        crown: r.random_range(2.5..5.8),
        cap_fraction,
        body_inset: if cap_fraction > 0. {
            r.random_range(0.91..0.98)
        } else {
            1.
        },
        body,
        top,
        frame: [
            Surface::WoodEdge,
            Surface::Metal,
            Surface::Chrome,
            Surface::Plastic,
        ][finish.random_range(0..4)],
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let label = o.kind.class_name();
    let span = p.base_span * s.xz();
    let h = p.base_height;
    // Feet/sleds meet a load-spreading subframe, not a floating rounded cushion.
    // The upholstered volume overlaps it by a small, dimensioned contact depth.
    let deck = p.rail * 0.8;
    match p.base {
        Base::Feet | Base::Sled => {
            for x in [-span.x, span.x] {
                if p.base == Base::Sled {
                    a.box_part(
                        p.frame,
                        label,
                        Vec3::new(x, p.rail * 0.5, 0.),
                        Vec3::new(p.rail, p.rail, 2. * span.y + p.rail),
                        p.rail * 0.18,
                    );
                }
                for z in [-span.y, span.y] {
                    a.box_part(
                        p.frame,
                        label,
                        Vec3::new(x, h * 0.5, z),
                        Vec3::new(p.rail, h, p.rail),
                        p.rail * 0.16,
                    );
                }
                a.box_part(
                    p.frame,
                    label,
                    Vec3::new(x, h, 0.),
                    Vec3::new(p.rail, deck, span.y * 2. + p.rail),
                    deck * 0.15,
                );
            }
            a.box_part(
                p.frame,
                label,
                Vec3::new(0., h, 0.),
                Vec3::new(span.x * 2. + p.rail, deck, p.rail),
                deck * 0.15,
            );
        }
        Base::Plinth | Base::Upholstered => {
            // Elliptical plinth follows both independent footprint axes.
            a.part(
                if p.base == Base::Upholstered {
                    Surface::Rubber
                } else {
                    p.frame
                },
                label,
            )
            .cylinder(
                1.,
                h,
                Transform::from_xyz(0., h * 0.5, 0.).with_scale(Vec3::new(span.x, 1., span.y)),
            );
        }
    }
    let bottom = (h - deck * 0.35).max(0.);
    let cap = (s.y - bottom) * p.cap_fraction;
    let body_top = s.y - cap;
    padding(
        a.part(p.body, label),
        Vec3::new(s.x * p.body_inset, body_top - bottom, s.z * p.body_inset),
        bottom,
        &p,
    );
    if cap > 0. {
        // Rounded panels interlock slightly at their central contact surface;
        // no coplanar upholstery faces or unsupported detached cushion.
        let overlap = cap * 0.035;
        padding(
            a.part(p.top, label),
            Vec3::new(s.x, cap + overlap, s.z),
            s.y - cap - overlap,
            &p,
        );
    }
}

/// Rounded boxing with broad planar contact patches. Stacking two ellipsoids
/// would leave a visible air gap around a single point of contact. The flat
/// underside also seats the body on the subframe across a real support area.
fn padding(g: &mut Geometry, s: Vec3, bottom: f32, p: &OttomanProgram) {
    let half = s.xz() * 0.5;
    let edge = (s.y * (6.7 - p.crown) * 0.09)
        .min(s.y * 0.44)
        .min(half.min_element() * 0.45);
    let rings = [
        (0., half - Vec2::splat(edge)),
        (edge * 0.134, half - Vec2::splat(edge * 0.5)),
        (edge * 0.5, half - Vec2::splat(edge * 0.134)),
        (edge, half),
        (s.y - edge, half),
        (s.y - edge * 0.5, half - Vec2::splat(edge * 0.134)),
        (s.y - edge * 0.134, half - Vec2::splat(edge * 0.5)),
        (s.y, half - Vec2::splat(edge)),
    ];
    g.padded_housing(&rings, p.roundness, Transform::from_xyz(0., bottom, 0.));
}
