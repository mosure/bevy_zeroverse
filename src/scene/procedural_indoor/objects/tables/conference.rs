//! Span-aware frames. Rendering and seated-person/chair clearance share every part.
use super::*;
use crate::scene::procedural_indoor::envelope::polygon;

pub(super) fn boardroom_finish(o: &IndoorObject, p: &mut TableProgram) {
    let mut r = stream(o.seed, 0x424f_4152_4454_4f50);
    // A tapered boat or barrel can still seat a person at each end. A strongly
    // blended triangle loses the available width along most of a long board.
    let broad = ((o.size.x.min(o.size.z) - 3.5) * 0.5).clamp(0., 1.);
    p.outline_exponent = r.random_range((2.4 + 8. * broad)..24.);
    p.polygon_blend *= 0.12 * (1. - broad * 0.85);
    p.taper *= 0.20 * (1. - broad * 0.75);
    p.capsule_blend *= 1. - 0.80 * broad;
    p.boat_blend *= 1. - 0.48 * broad;
    p.boat_end_width = r.random_range(0.68..0.96);
    p.boat_shoulder = r.random_range(0.42..0.82);
    p.top_thickness = r.random_range(0.035..0.065);
    p.edge_bevel = r.random_range(0.0015..0.0045);
    p.top_surface = [
        Surface::Wood,
        Surface::Wood,
        Surface::Wood,
        Surface::Plastic,
        Surface::Ceramic,
    ][r.random_range(0..5)];
}

/// Manufacture a long tabletop as joined leaves on its continuous backing and
/// multi-bay frame. Clipping the shared outline keeps all outer edges aligned.
pub(super) fn joined_top(a: &mut Assembly, o: &IndoorObject, p: &TableProgram, outline: &[Vec2]) {
    let mut r = stream(o.seed, 0x5441_424c_454c_4546);
    let span = o.size.xz().max_element();
    let axis = if o.size.x > o.size.z {
        Vec2::X
    } else {
        Vec2::Y
    };
    let leaves = (span / r.random_range(1.55..2.65)).round().max(2.) as usize;
    let half_gap = r.random_range(0.0007..0.0016);
    let half = span * 0.5;
    let pitch = span / leaves as f32;
    let mut start = -half;
    let slab = a.part(p.top_surface, o.kind.class_name());
    for i in 1..=leaves {
        let end = if i == leaves {
            half
        } else {
            -half + pitch * (i as f32 + r.random_range(-0.13..0.13))
        };
        let mut leaf = polygon::clip(outline, axis, start);
        if i < leaves {
            leaf = polygon::clip(&leaf, -axis, -(end - half_gap));
        }
        // profile_slab uses a centre fan and metric UVs. Each convex leaf
        // needs its own interior origin, including the two curved end panels.
        let center = leaf.iter().sum::<Vec2>() / leaf.len() as f32;
        for point in &mut leaf {
            *point -= center;
        }
        let first_vertex = slab.positions.len();
        slab.profile_slab(
            &leaf,
            p.top_thickness,
            p.edge_bevel,
            Transform::from_xyz(center.x, o.size.y - p.top_thickness * 0.5, center.y),
        );
        // The centre fan is local to each leaf. The veneer and stone face
        // charts span the assembled top; the bevel keeps its edge unwrap.
        for vertex in first_vertex..slab.positions.len() {
            if slab.normals[vertex][1].abs() > 0.99 {
                let [x, _, z] = slab.positions[vertex];
                slab.uvs[vertex] = [x, z];
            }
        }
        start = end + half_gap;
    }
}

pub(super) fn bay_count(o: &IndoorObject, p: &TableProgram) -> usize {
    let size = o.size.xz();
    if o.kind != ObjectKind::Table
        || size.max_element() <= 3.0
        || size.max_element() / size.min_element() < 1.35
    {
        return 0;
    }
    (((size.max_element() - 2. * p.end_overhang) / p.span_pitch).ceil() as usize + 1).clamp(2, 12)
}

pub(super) fn build(
    a: &mut impl StructureSink,
    o: &IndoorObject,
    p: &TableProgram,
    outline: &[Vec2],
    underside: f32,
) {
    let count = bay_count(o, p);
    let size = o.size.xz();
    let long_x = size.x > size.y;
    let (length, width) = (size.max_element(), size.min_element());
    let axis = if long_x { Vec3::X } else { Vec3::Z };
    let across = if long_x { Vec3::Z } else { Vec3::X };
    let rotation = if long_x {
        Quat::from_rotation_y(FRAC_PI_2)
    } else {
        Quat::IDENTITY
    };
    let label = o.kind.class_name();
    let frame = p.frame_surface;
    let stem_radius = (p.leg_radius * 1.4).clamp(0.028, 0.052);
    let beam_y = underside - p.beam_depth * 0.5;
    let mut centers = Vec::with_capacity(count);
    for i in 0..count {
        let offset = (i as f32 / (count - 1) as f32 - 0.5) * (length - 2. * p.end_overhang);
        // Both mount and crossbar must fit tapered/rounded polygon ends.
        let center = supports::fit_center(axis * offset, outline, 0.12);
        let mut half_span = width * p.bay_width * 0.5;
        for _ in 0..24 {
            let pad = across.xz() * half_span + axis.xz().abs() * 0.065;
            if supports_outline(outline, center.xz() - pad, center.xz() + pad, 0.02) {
                break;
            }
            half_span *= 0.92;
        }
        // A crossbar at each station transfers the slab load into its support.
        a.part(frame, label).cuboid(
            Vec3::new(half_span * 2. + 0.025, p.beam_depth, 0.065),
            0.005,
            Transform::from_translation(center.with_y(beam_y)).with_rotation(rotation),
        );
        let height = underside - p.beam_depth;
        let foot_y = 0.026;
        let foot_span = (half_span + p.bay_rake).min(width * 0.40);
        match p.bay_style {
            0 | 1 => {
                // T-foot or splayed twin-column base; both leave a central knee bay.
                let spread = if p.bay_style == 0 {
                    0.
                } else {
                    half_span * 0.70
                };
                let columns = if p.bay_style == 0 { 1 } else { 2 };
                for j in 0..columns {
                    let sign = if j == 0 { -1. } else { 1. };
                    a.part(frame, label).rod(
                        center + across * spread * sign + Vec3::Y * foot_y,
                        center + across * spread * 0.75 * sign + Vec3::Y * height,
                        stem_radius,
                    );
                }
                a.part(frame, label).cuboid(
                    Vec3::new(foot_span * 2., 0.04, 0.12),
                    0.012,
                    Transform::from_translation(center.with_y(foot_y)).with_rotation(rotation),
                );
            }
            2 | 3 => {
                // Repeated A trestles or portal legs with individually sampled rake.
                let mount = if p.bay_style == 2 {
                    half_span * 0.25
                } else {
                    half_span
                };
                for sign in [-1., 1.] {
                    a.part(frame, label).rod(
                        center + across * foot_span * sign + Vec3::Y * (stem_radius + 0.008),
                        center + across * mount * sign + Vec3::Y * height,
                        stem_radius,
                    );
                }
            }
            4 => {
                // Panel pedestals under the spine, not a full-width wall at each seat.
                a.part(frame, label).cuboid(
                    Vec3::new(half_span * 1.35, height - 0.012, 0.065 + p.leg_radius),
                    0.008,
                    Transform::from_translation(center.with_y((height + 0.012) * 0.5))
                        .with_rotation(rotation),
                );
            }
            _ => {
                let r = half_span.min(width * 0.22).min(0.38);
                a.part(frame, label).lathe(
                    &[
                        (0., 0.),
                        (r, 0.),
                        (r, 0.02),
                        (r * p.pedestal_waist, height * p.pedestal_shoulder),
                        (r * 0.72, height),
                        (0., height),
                    ],
                    p.pedestal_facets,
                    Transform::from_translation(center),
                );
            }
        }
        if p.bay_style < 5 {
            let pad_span = if p.bay_style == 4 {
                half_span * 0.58
            } else {
                foot_span * 0.95
            };
            for sign in [-1., 1.] {
                a.part(Surface::Rubber, label).cylinder(
                    stem_radius + 0.009,
                    0.020,
                    Transform::from_translation(center + across * pad_span * sign + Vec3::Y * 0.01),
                );
            }
        }
        centers.push(center);
    }
    // Adjacent crossbars are mechanically joined; no unsupported multi-metre slab.
    for pair in centers.windows(2) {
        let delta = pair[1] - pair[0];
        a.part(frame, label).cuboid(
            Vec3::new(0.075, p.beam_depth, delta.length() + 0.045),
            0.005,
            Transform::from_translation(((pair[0] + pair[1]) * 0.5).with_y(beam_y))
                .with_rotation(Quat::from_rotation_arc(Vec3::Z, delta.normalize())),
        );
    }
}
