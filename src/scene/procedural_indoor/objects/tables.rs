//! Continuous convex top outlines and independent structural support programs.
use super::*;
mod clearance;
mod conference;
mod outline;
mod structure;
mod supports;
#[cfg(test)]
mod tests;
pub(crate) use clearance::Clearance;
use structure::{PartSink, StructureSink};
/// Shared lecture/dining rows need usable work surface across their full bank.
/// Keep broad catalogue shapes for freestanding tables and desks.
pub(crate) const SHARED_ROW_VARIANT: u32 = 30;
#[derive(Debug, Clone, serde::Serialize)]
pub struct TableProgram {
    pub top_thickness: f32,
    pub edge_bevel: f32,
    pub leg_radius: f32,
    pub leg_inset: f32,
    pub leg_rake: f32,
    /// 2 = ellipse/circle; high exponents approach rounded rectangles/trapezoids.
    pub outline_exponent: f32,
    pub taper: f32,
    pub support: u8,
    pub top_surface: Surface,
    pub pedestal_radius: f32,
    pub pedestal_base_radius: f32,
    pub pedestal_base_aspect: f32,
    pub capsule_blend: f32,
    pub leg_taper: f32,
    pub leg_aspect: f32,
    pub frame_bend_m: f32,
    /// Fraction of the available under-top span used by a splayed trestle.
    pub trestle_spread: f32,
    /// Continuous rake of the spider pedestal spokes toward the top mount.
    pub spider_sweep: f32,
    /// Convex blend with a rounded polygon; zero retains a curved outline.
    pub polygon_blend: f32,
    pub polygon_sides: u32,
    pub polygon_phase: f32,
    pub polygon_rounding: f32,
    pub pedestal_waist: f32,
    pub pedestal_shoulder: f32,
    pub pedestal_facets: u32,
    pub frame_surface: Surface,
    /// Convex boat-shaped shoulders, independent of the capsule/polygon blend.
    pub boat_blend: f32,
    pub boat_end_width: f32,
    pub boat_shoulder: f32,
    /// Repeated structural bays for long conference tops (metres).
    pub span_pitch: f32,
    pub end_overhang: f32,
    pub beam_depth: f32,
    pub bay_width: f32,
    pub bay_rake: f32,
    pub bay_style: u8,
}
pub fn parameters(o: &IndoorObject) -> TableProgram {
    let mut rng = stream(o.seed, 172);
    let mut shape = stream(o.seed, 6420);
    let mut conference = stream(o.seed, 6421);
    let exponent = if o.kind == ObjectKind::Desk {
        rng.random_range(5.0..24.0)
    } else if rng.random_bool(0.4) {
        2.0
    } else {
        rng.random_range(2.2..24.0)
    };
    let mut program = TableProgram {
        top_thickness: rng.random_range(0.026..0.052),
        edge_bevel: rng.random_range(0.003..0.012),
        leg_radius: rng.random_range(0.018..0.036),
        leg_inset: rng.random_range(0.11..0.18),
        leg_rake: rng.random_range(0.0..0.075),
        outline_exponent: exponent,
        taper: if exponent > 5.0 && rng.random_bool(0.45) {
            rng.random_range(-0.42..0.42)
        } else {
            0.0
        },
        support: rng.random_range(0..10),
        top_surface: match rng.random_range(0..10) {
            0 => Surface::Ceramic,
            1 => Surface::Plastic,
            2 => Surface::Concrete,
            _ => Surface::Wood,
        },
        pedestal_radius: rng.random_range(0.045..0.085),
        pedestal_base_radius: rng.random_range(0.24..0.36),
        pedestal_base_aspect: rng.random_range(0.72..1.0),
        capsule_blend: rng.random_range(
            0.0..if o.kind == ObjectKind::Desk {
                0.25
            } else {
                1.0
            },
        ),
        leg_taper: rng.random_range(0.60..1.0),
        leg_aspect: rng.random_range(0.65..1.0),
        frame_bend_m: rng.random_range(0.025..0.065),
        trestle_spread: rng.random_range(0.62..0.98),
        spider_sweep: rng.random_range(0.18..0.48),
        polygon_blend: if shape.random_bool(if o.kind == ObjectKind::Desk {
            0.18
        } else {
            0.55
        }) {
            shape.random_range(0.35..0.90)
        } else {
            0.
        },
        polygon_sides: if o.kind == ObjectKind::Desk {
            4
        } else {
            shape.random_range(3..=8)
        },
        polygon_phase: shape.random_range(0.0..TAU),
        polygon_rounding: shape.random_range(0.12..0.45),
        pedestal_waist: shape.random_range(0.28..0.92),
        pedestal_shoulder: shape.random_range(0.25..0.78),
        pedestal_facets: [6, 8, 12, 32][shape.random_range(0..4)],
        frame_surface: if o.variant == 1 {
            Surface::WoodEdge
        } else {
            [
                Surface::Metal,
                Surface::Metal,
                Surface::Chrome,
                Surface::WoodEdge,
            ][shape.random_range(0..4)]
        },
        boat_blend: if conference.random_bool(if o.kind == ObjectKind::Table {
            0.55
        } else {
            0.18
        }) {
            conference.random_range(0.20..0.78)
        } else {
            0.
        },
        boat_end_width: conference.random_range(0.48..0.92),
        boat_shoulder: conference.random_range(0.28..0.70),
        span_pitch: conference.random_range(1.25..2.10),
        end_overhang: conference.random_range(0.40..0.70),
        beam_depth: conference.random_range(0.035..0.060),
        bay_width: conference.random_range(0.38..0.62),
        bay_rake: conference.random_range(0.0..0.14),
        bay_style: conference.random_range(0..6),
    };
    if o.variant == SHARED_ROW_VARIANT {
        let mut row = stream(o.seed, 0x524f57544f50);
        program.outline_exponent = row.random_range(6.0..24.0);
        program.taper = row.random_range(-0.06..0.06);
        program.capsule_blend = row.random_range(0.0..0.22);
        program.polygon_blend = 0.;
        program.boat_blend = row.random_range(0.0..0.12);
    }
    program
}

/// Number of load-bearing stations. Zero uses the compact-table support program.
pub fn support_bay_count(o: &IndoorObject) -> usize {
    conference::bay_count(o, &parameters(o))
}
pub fn outline(o: &IndoorObject) -> Vec<Vec2> {
    outline::sample(o.size.xz() * 0.5, &parameters(o))
}

pub fn supports(o: &IndoorObject, lo: Vec2, hi: Vec2, margin: f32) -> bool {
    supports_outline(&outline(o), lo, hi, margin)
}
fn supports_outline(outline: &[Vec2], lo: Vec2, hi: Vec2, margin: f32) -> bool {
    [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)]
        .iter()
        .all(|p| {
            (0..outline.len()).all(|i| {
                let a = outline[i];
                let edge = outline[(i + 1) % outline.len()] - a;
                edge.perp_dot(*p - a) >= (margin + 0.012) * edge.length()
            })
        })
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let p = parameters(o);
    let label = o.kind.class_name();
    let outline = outline(o);
    let underside = s.y - p.top_thickness;
    a.part(p.top_surface, label).profile_slab(
        &outline,
        p.top_thickness,
        p.edge_bevel,
        Transform::from_xyz(0.0, s.y - p.top_thickness * 0.5, 0.0),
    );
    if s.x > s.z && p.top_surface == Surface::Wood {
        for uv in &mut a.part(Surface::Wood, label).uvs {
            *uv = [uv[1], -uv[0]];
        }
    }
    let edge: Vec<_> = outline.iter().map(|p| *p * 0.985).collect();
    a.part(
        if p.top_surface == Surface::Wood {
            Surface::WoodEdge
        } else {
            p.top_surface
        },
        label,
    )
    .profile_slab(
        &edge,
        0.012,
        0.003,
        Transform::from_xyz(0.0, underside - 0.006, 0.0),
    );
    build_structure(a, o, &p, &outline);
}

// Rendering and clearance consume the same structural primitive program.
fn build_structure(
    a: &mut impl StructureSink,
    o: &IndoorObject,
    p: &TableProgram,
    outline: &[Vec2],
) {
    let s = o.size;
    let label = o.kind.class_name();
    let underside = s.y - p.top_thickness - 0.012;
    let frame = p.frame_surface;
    if conference::bay_count(o, p) > 0 {
        conference::build(a, o, p, outline, underside);
        return;
    }
    let half = Vec2::new(s.x, s.z) * 0.5;
    let mut anchor = Vec2::new(
        (half.x - p.leg_inset).max(0.12),
        (half.y - p.leg_inset).max(0.12),
    ) * 0.68;
    // Mounts must fit the true convex top, including its tapered narrow end.
    for _ in 0..24 {
        if [-1., 1.].into_iter().all(|x| {
            [-1., 1.].into_iter().all(|z| {
                let mount = anchor * Vec2::new(x, z);
                let pad = Vec2::splat(if p.support == 8 { 0.10 } else { 0.05 });
                supports_outline(outline, mount - pad, mount + pad, 0.)
            })
        }) {
            break;
        }
        anchor *= 0.92;
    }
    match p.support {
        0 | 1 => {
            for x in [-1.0, 1.0] {
                for z in [-1.0, 1.0] {
                    let top = Vec3::new(
                        x * anchor.x,
                        underside,
                        z * anchor.y * if p.support == 1 { 0.38 } else { 1.0 },
                    );
                    let foot = Vec3::new(
                        x * (anchor.x + p.leg_rake),
                        p.leg_radius + 0.008,
                        z * (anchor.y + p.leg_rake),
                    );
                    let delta = top - foot;
                    a.part(frame, label).lathe(
                        &[
                            (0., 0.),
                            (p.leg_radius * p.leg_taper, 0.),
                            (p.leg_radius, delta.length()),
                            (0., delta.length()),
                        ],
                        16,
                        Transform::from_translation(foot)
                            .with_rotation(Quat::from_rotation_arc(Vec3::Y, delta.normalize()))
                            .with_scale(Vec3::new(p.leg_aspect, 1., 1.)),
                    );
                    a.box_part(
                        frame,
                        label,
                        top.with_y(underside - 0.004),
                        Vec3::new(0.09, 0.008, 0.075),
                        0.003,
                    );
                    a.part(Surface::Rubber, label).cylinder(
                        p.leg_radius + 0.003,
                        p.leg_radius + 0.01,
                        Transform::from_translation(foot.with_y((p.leg_radius + 0.01) * 0.5)),
                    );
                }
            }
            for x in [-1.0, 1.0] {
                a.part(frame, label).rod(
                    Vec3::new(x * anchor.x, underside - 0.03, -anchor.y),
                    Vec3::new(x * anchor.x, underside - 0.03, anchor.y),
                    0.024,
                );
                if p.support == 1 {
                    a.part(frame, label).rod(
                        Vec3::new(x * anchor.x, 0.22, -anchor.y * 0.7),
                        Vec3::new(x * anchor.x, 0.22, anchor.y * 0.7),
                        0.018,
                    );
                }
            }
        }
        2 => {
            let long_x = s.x > s.z;
            let ratio = s.x.max(s.z) / s.x.min(s.z);
            let offsets = if ratio > 1.8 {
                vec![-0.25, 0.25]
            } else {
                vec![0.0]
            };
            for offset in offsets {
                let center = if long_x {
                    Vec3::X * s.x * offset
                } else {
                    Vec3::Z * s.z * offset
                };
                let center =
                    supports::fit_center(center, outline, s.x.min(s.z) * p.pedestal_radius + 0.018);
                a.part(frame, label).cylinder(
                    s.x.min(s.z) * p.pedestal_radius,
                    underside - 0.04,
                    Transform::from_translation(center + Vec3::Y * (underside + 0.04) * 0.5),
                );
                let radius = s.x.min(s.z) * p.pedestal_base_radius;
                a.part(frame, label).lathe(
                    &[
                        (0., 0.0),
                        (radius * 0.94, 0.0),
                        (radius, 0.012),
                        (radius * 0.97, 0.022),
                        (radius * 0.65, 0.036),
                        (0., 0.04),
                    ],
                    40,
                    Transform::from_translation(center).with_scale(Vec3::new(
                        1.,
                        1.,
                        p.pedestal_base_aspect,
                    )),
                );
            }
        }
        3 => {
            for x in [-anchor.x, anchor.x] {
                let radius = p.leg_radius.min(anchor.y * 0.24);
                let r = p.frame_bend_m.max(radius * 2.5).min(anchor.y * 0.65);
                let lo = p.leg_radius + 0.007;
                let hi = underside - p.leg_radius;
                let mut path = Vec::new();
                for (cy, cz, phase) in [
                    (lo + r, anchor.y - r, PI),
                    (hi - r, anchor.y - r, FRAC_PI_2),
                    (hi - r, -anchor.y + r, 0.),
                    (lo + r, -anchor.y + r, -FRAC_PI_2),
                ] {
                    for i in 0..=6 {
                        let t = phase - i as f32 * FRAC_PI_2 / 6.;
                        path.push(Vec3::new(x, cy + r * t.cos(), cz + r * t.sin()));
                    }
                }
                path.push(path[0]);
                a.part(frame, label).tube(&path, radius, 10);
            }
        }
        4 => {
            for z in [-anchor.y, anchor.y] {
                for sign in [-1.0, 1.0] {
                    a.part(frame, label).rod(
                        Vec3::new(sign * anchor.x, p.leg_radius + 0.01, z),
                        Vec3::new(-sign * anchor.x, underside, z),
                        p.leg_radius,
                    );
                }
            }
        }
        5 => {
            // End panels leave an open knee bay; width, rake and thickness
            // remain independent of the top outline and finish.
            for x in [-anchor.x, anchor.x] {
                a.part(frame, label).cuboid(
                    Vec3::new(p.leg_radius * 2.0, underside - 0.016, anchor.y * 2.0),
                    0.004,
                    Transform::from_xyz(x, (underside + 0.016) * 0.5, 0.0),
                );
            }
        }
        6 => {
            // Two true trestles: the narrow upper mount leaves knee clearance,
            // while splayed legs carry loads to separated, grounded feet.
            // The rail mechanically joins the trestles below the tabletop.
            let spread = anchor.y * p.trestle_spread;
            let mount = spread * 0.30;
            for x in [-anchor.x, anchor.x] {
                for sign in [-1., 1.] {
                    let top = Vec3::new(x, underside - 0.018, sign * mount);
                    let foot = Vec3::new(x, p.leg_radius + 0.008, sign * spread);
                    a.part(frame, label).rod(foot, top, p.leg_radius);
                    a.part(Surface::Rubber, label).cylinder(
                        p.leg_radius + 0.005,
                        p.leg_radius + 0.010,
                        Transform::from_translation(foot.with_y((p.leg_radius + 0.010) * 0.5)),
                    );
                }
                a.box_part(
                    frame,
                    label,
                    Vec3::new(x, underside - 0.017, 0.),
                    Vec3::new(p.leg_radius * 2.8, 0.034, spread * 2. + p.leg_radius * 2.),
                    0.006,
                );
                let rail_y = underside * 0.42;
                let fraction =
                    (rail_y - p.leg_radius - 0.008) / (underside - 0.018 - p.leg_radius - 0.008);
                let rail_z = spread + (mount - spread) * fraction;
                a.part(frame, label).rod(
                    Vec3::new(x, rail_y, -rail_z),
                    Vec3::new(x, rail_y, rail_z),
                    p.leg_radius * 0.65,
                );
            }
            a.part(frame, label).rod(
                Vec3::new(-anchor.x, underside * 0.42, 0.),
                Vec3::new(anchor.x, underside * 0.42, 0.),
                p.leg_radius * 0.80,
            );
        }
        7 => {
            // Sculptural spider bases have grounded spokes and an actual
            // central stem. Long conference tops use two stems for support.
            let long_x = s.x > s.z;
            let count = if s.x.max(s.z) / s.x.min(s.z) > 1.8 {
                2
            } else {
                1
            };
            let radius = s.x.min(s.z) * p.pedestal_base_radius;
            for i in 0..count {
                let offset = if count == 1 {
                    0.
                } else {
                    (i as f32 - 0.5) * s.x.max(s.z) * 0.5
                };
                let center = if long_x {
                    Vec3::X * offset
                } else {
                    Vec3::Z * offset
                };
                let center =
                    supports::fit_center(center, outline, s.x.min(s.z) * p.pedestal_radius + 0.018);
                a.part(frame, label).cylinder(
                    s.x.min(s.z) * p.pedestal_radius,
                    underside - 0.13,
                    Transform::from_translation(center + Vec3::Y * (underside + 0.13) * 0.5),
                );
                for x in [-1., 1.] {
                    for z in [-1., 1.] {
                        let direction = Vec3::new(x, 0., z).normalize();
                        let foot = center + direction * radius + Vec3::Y * (p.leg_radius + 0.008);
                        let root = center
                            + direction * radius * p.spider_sweep
                            + Vec3::Y * (underside * 0.44).max(0.16);
                        let path = [foot, foot.lerp(root, 0.40) + Vec3::Y * p.frame_bend_m, root];
                        a.part(frame, label).tube(&path, p.leg_radius, 10);
                        // Each spoke is attached to the stem, rather than
                        // terminating in open space beside it.
                        a.part(frame, label)
                            .rod(root, center.with_y(root.y), p.leg_radius);
                        a.part(Surface::Rubber, label).cylinder(
                            p.leg_radius + 0.005,
                            p.leg_radius + 0.010,
                            Transform::from_translation(foot.with_y((p.leg_radius + 0.010) * 0.5)),
                        );
                    }
                }
            }
        }
        8 => supports::hairpin(a, p, anchor, underside, frame, label),
        _ => supports::drums(a, o, p, outline, underside, frame, label),
    }
    if (3..=5).contains(&p.support) {
        for x in [-anchor.x, anchor.x] {
            for z in [-anchor.y, anchor.y] {
                a.part(Surface::Rubber, label).cylinder(
                    p.leg_radius + 0.005,
                    0.016,
                    Transform::from_xyz(x, 0.008, z),
                );
            }
        }
    }
    a.part(frame, label).rod(
        Vec3::new(-anchor.x, underside - 0.035, 0.0),
        Vec3::new(anchor.x, underside - 0.035, 0.0),
        0.024,
    );
    if o.kind == ObjectKind::Desk || (o.kind == ObjectKind::Table && p.outline_exponent > 4.0) {
        a.box_part(
            Surface::Plastic,
            label,
            Vec3::new(0.0, underside - 0.07, 0.0),
            Vec3::new((s.x * 0.35).min(anchor.x * 2.), 0.045, 0.16),
            0.005,
        );
        let center = supports::fit_center(Vec3::new(0., 0., -s.z * 0.20), outline, 0.09);
        let pad = Vec2::new(0.084, 0.032);
        if supports_outline(outline, center.xz() - pad, center.xz() + pad, 0.008) {
            a.box_part(
                Surface::Metal,
                label,
                center.with_y(s.y + 0.001),
                Vec3::new(0.16, 0.002, 0.055),
                0.0,
            );
        }
    }
}
