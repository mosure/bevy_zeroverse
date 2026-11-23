//! Landscaped grade outside the primary room. The near field follows permanent
//! window apertures; each courtyard's soil opening follows its installed trees.
use super::super::super::materials::exterior::label;
use super::*;
mod paving;
use paving::{Frame, Paving, Rect};

pub(super) const FIRST_VERSION: u32 = 173;
const GRADE: f32 = -0.07;
const PATIO: f32 = 0.03;

#[derive(Clone, Copy)]
struct Forecourt {
    rect: Rect,
    paving: Paving,
    seed: u64,
}

fn surrounds(
    scene: &IndoorManifest,
    windows: &[WindowView],
    d: &District,
    mut visit: impl FnMut(Rect, Option<Forecourt>),
) {
    let half = scene.room_size.xz() * 0.5;
    let (left, right, back, front) = (
        -half.x - 0.5,
        half.x + 0.5,
        -half.y - 0.5,
        half.y + scene.adjoining_space().max_depth() + 0.5,
    );
    // Land use is shared around the property. Independent per-side opt-outs
    // otherwise put paving on almost every multi-frontage site.
    let hardscape = scene.generator_version >= FIRST_VERSION
        && stream(scene.seed, 9538).random::<f32>() >= 0.08 + 0.16 * (1. - d.building_fraction);
    let bounds = [
        (
            Vec2::new(left - 18., back - 18.),
            Vec2::new(left, front + 18.),
            Vec2::NEG_X,
        ),
        (
            Vec2::new(right, back - 18.),
            Vec2::new(right + 18., front + 18.),
            Vec2::X,
        ),
        (
            Vec2::new(left, back - 18.),
            Vec2::new(right, back),
            Vec2::NEG_Y,
        ),
        (
            Vec2::new(left, front),
            Vec2::new(right, front + 18.),
            Vec2::Y,
        ),
    ];
    for (side, (lo, hi, outward)) in bounds.into_iter().enumerate() {
        let bounds = Rect { lo, hi };
        let field = hardscape
            .then(|| {
                let mut r = stream(scene.seed, 9541 + side as u64);
                // Weighted reservoir favours wider, lower openings. An installation
                // can also occur outside a narrow clerestory; camera count is irrelevant.
                let mut choice = None;
                let mut sum = 0.;
                for w in windows {
                    let alignment = w.normal.dot(outward);
                    if alignment < 0.58 {
                        continue;
                    }
                    let weight = (alignment * w.width / (0.45 + w.sill.max(0.))).max(0.001);
                    sum += weight;
                    if r.random::<f32>() * sum < weight {
                        choice = Some(w);
                    }
                }
                let w = choice?;
                // Some open gardens deliberately keep uninterrupted soft ground.
                if r.random_bool(0.14) {
                    return None;
                }
                let normal = outward.abs();
                let tangent = normal.yx();
                let span = (w.width * r.random_range(1.1..2.6) + r.random_range(3.0..8.0))
                    .min(bounds.size().dot(tangent) - 0.20);
                let depth = r.random_range(3.1..8.7);
                let along = (w.center.dot(tangent) + r.random_range(-0.22..0.22) * span).clamp(
                    lo.dot(tangent) + span * 0.5 + 0.06,
                    hi.dot(tangent) - span * 0.5 - 0.06,
                );
                let lip = lo.dot(outward).min(hi.dot(outward))
                    + d.paving_width
                    + r.random_range(0.07..0.30);
                let c = tangent * along + outward * (lip + depth * 0.5);
                let extent = (tangent * span + normal * depth) * 0.5;
                let rect = Rect {
                    lo: c - extent,
                    hi: c + extent,
                };
                let seed = r.random();
                Some(Forecourt {
                    rect,
                    seed,
                    paving: Paving::sample(seed, rect),
                })
            })
            .flatten();
        visit(bounds, field);
    }
    if let Some((mut lo, mut hi)) = scene.adjoining_space().recess(scene.room_size) {
        if lo.x < -half.x + 0.01 {
            lo.x = left;
        } else {
            hi.x = right;
        }
        hi.y = front;
        visit(Rect { lo, hi }, None);
    }
}

pub(super) fn grade(
    a: &mut Assembly,
    scene: &IndoorManifest,
    windows: &[WindowView],
    d: &District,
) {
    let surface = if stream(scene.seed, 6389).random_bool(0.45) {
        Surface::Soil
    } else {
        Surface::Concrete
    };
    let ground_label = label(surface, "floor", scene.seed);
    let frame = Frame::new(Vec2::ZERO, 0.);
    surrounds(scene, windows, d, |r, court| {
        if let Some(c) = court {
            paving::fill_apron(
                frame,
                a.part(surface, &ground_label),
                r,
                c.rect,
                GRADE,
                c.paving.relief,
            );
            c.paving.lay(
                frame,
                a.part(c.paving.surface, &label(c.paving.surface, "floor", c.seed)),
                c.rect,
                GRADE,
                None,
            );
        } else {
            a.box_part(
                surface,
                &ground_label,
                Vec3::new((r.lo.x + r.hi.x) * 0.5, -0.17, (r.lo.y + r.hi.y) * 0.5),
                Vec3::new(r.hi.x - r.lo.x, 0.20, r.hi.y - r.lo.y),
                0.,
            );
        }
    });
}

#[derive(Clone, Copy)]
struct Bed {
    outer: Rect,
    soil: Rect,
    elevation: f32,
    coping: f32,
}

impl Bed {
    fn fit(p: &Parcel, frame: Frame, plants: &[IndoorObject]) -> Option<Self> {
        if p.kind != 2 || plants.is_empty() {
            return None;
        }
        let mut r = stream(p.seed, 9557);
        let mut lo = Vec2::splat(f32::INFINITY);
        let mut hi = -lo;
        for plant in plants {
            let center = frame.local(plant.position.xz());
            let flank = (plant.size.x * 0.24).clamp(0.26, 0.82);
            let spread = Vec2::new(
                flank * r.random_range(1.0..1.7),
                flank * r.random_range(1.05..1.9),
            );
            lo = lo.min(center - spread);
            hi = hi.max(center + spread);
        }
        let half = p.size * 0.5;
        let soil = Rect {
            lo: lo.max(-half + 0.12),
            hi: hi.min(Vec2::new(half.x - 0.12, p.size.y * 0.18)),
        };
        Some(Self {
            outer: soil.inset(-r.random_range(0.038..0.082)),
            soil,
            elevation: PATIO - r.random_range(0.045..0.088),
            coping: PATIO + r.random_range(0.009..0.024),
        })
    }
}

/// Raised courtyard / pergola deck. Paver joints and the tree opening both have
/// real returns, so surface normals, depth and RGB agree at their edges.
pub(super) fn patio(a: &mut Assembly, p: &Parcel, plants: &mut [IndoorObject]) {
    let frame = Frame::new(p.center, p.yaw);
    let rect = Rect {
        lo: -p.size * 0.5,
        hi: p.size * 0.5,
    };
    let program = Paving::sample(p.seed, rect);
    let bed = Bed::fit(p, frame, plants);
    let substrate = a.part(
        Surface::Concrete,
        &label(Surface::Concrete, "floor", p.seed),
    );
    frame.base(
        substrate,
        rect,
        -0.08,
        PATIO - program.relief,
        bed.map(|b| b.outer),
    );
    program.lay(
        frame,
        a.part(program.surface, &label(program.surface, "floor", p.seed)),
        rect,
        PATIO,
        bed.map(|b| b.outer.inset(-program.joint * 0.5)),
    );
    if let Some(b) = bed {
        let border = a.part(
            Surface::Concrete,
            &label(Surface::Concrete, "floor", p.seed),
        );
        frame.rim(
            border,
            b.outer,
            b.soil,
            b.elevation,
            PATIO - program.relief,
            b.coping,
        );
        frame.top(
            a.part(Surface::Soil, &label(Surface::Soil, "floor", p.seed)),
            b.soil,
            b.elevation,
            Vec2::ZERO,
        );
        for plant in plants {
            plant.position.y = b.elevation;
        }
    }
}

pub(super) fn measure(
    scene: &IndoorManifest,
    windows: &[WindowView],
    district: &District,
    sites: &[Parcel],
    out: &mut Vec<(&'static str, f32)>,
) {
    if scene.generator_version < FIRST_VERSION {
        return;
    }
    let mut courts = 0;
    surrounds(scene, windows, district, |_, c| {
        if let Some(c) = c {
            courts += 1;
            out.push((
                "exterior_forecourt_area_m2",
                c.rect.size().element_product(),
            ));
            out.push(("exterior_forecourt_slabs", c.paving.panel_count() as f32));
            out.push(("exterior_forecourt_bond", c.paving.bond as u32 as f32));
            paving_metrics(c.paving, c.rect, out);
        }
    });
    out.push(("exterior_forecourts", courts as f32));
    for p in sites.iter().filter(|p| matches!(p.kind, 2 | 3)) {
        let rect = Rect {
            lo: -p.size * 0.5,
            hi: p.size * 0.5,
        };
        paving_metrics(Paving::sample(p.seed, rect), rect, out);
        if p.kind != 2 {
            continue;
        }
        if let Some(b) = Bed::fit(p, Frame::new(p.center, p.yaw), &garden_plants(scene, p)) {
            out.push((
                "exterior_courtyard_soil_area_m2",
                b.soil.size().element_product(),
            ));
            out.push(("exterior_courtyard_soil_recess_m", PATIO - b.elevation));
        }
    }
}
fn paving_metrics(p: Paving, rect: Rect, out: &mut Vec<(&'static str, f32)>) {
    let module = p.module(rect);
    out.extend([
        ("exterior_paving_joint_width_m", p.joint),
        ("exterior_paving_stock_length_m", module.x),
        ("exterior_paving_course_width_m", module.y),
        (
            "exterior_paving_fired_clay",
            f32::from(p.surface == Surface::Terracotta),
        ),
    ]);
}

#[cfg(test)]
mod tests;
