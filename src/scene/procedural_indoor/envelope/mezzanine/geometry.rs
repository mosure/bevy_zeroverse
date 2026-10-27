use super::*;
use crate::scene::procedural_indoor::objects::Assembly;

fn box_bounds(a: &mut Assembly, surface: Surface, label: &str, lo: Vec3, hi: Vec3) {
    a.box_part(surface, label, (lo + hi) * 0.5, hi - lo, 0.);
}

pub(in super::super) fn build(a: &mut Assembly, m: &Mezzanine) {
    let Some(c) = &m.construction else {
        legacy::build(a, m);
        return;
    };
    let lo = Vec3::new(m.deck.min.x, m.deck.height - m.thickness, m.deck.min.y);
    let hi = Vec3::new(m.deck.max.x, m.deck.height, m.deck.max.y);
    let cap = m.thickness.min(0.035);
    box_bounds(a, c.frame_surface, "floor", lo, hi.with_y(hi.y - cap));
    box_bounds(a, c.deck_surface, "floor", lo.with_y(hi.y - cap), hi);
    for (lo, hi) in m.columns() {
        support_column(a, c, lo, hi);
    }
    if let Some(frame) = m.landing_frame() {
        for (lo, hi) in frame.columns {
            support_column(a, c, lo, hi);
        }
        box_bounds(
            a,
            c.frame_surface,
            "other_structure",
            frame.beam.0,
            frame.beam.1,
        );
    }
    for (lo, hi) in m.beams() {
        box_bounds(a, c.frame_surface, "other_structure", lo, hi);
    }
    for (i, (lo, hi)) in m.treads().enumerate() {
        let bottom = hi.y - c.tread_thickness;
        if c.stair_structure == StairStructure::Solid {
            box_bounds(a, c.frame_surface, "floor", lo, hi.with_y(bottom));
        }
        // Inlaid nosing is split out of the cap, not a coplanar overlay. Its
        // upper surface agrees exactly with the recorded support elevation.
        let axis = if m.stair_axis == 0 { 0 } else { 2 };
        let mut cap_lo = lo.with_y(bottom);
        let mut cap_hi = hi;
        let mut nose_lo = cap_lo;
        let mut nose_hi = cap_hi;
        if m.ascending_positive() {
            nose_hi[axis] = nose_lo[axis] + c.nosing_depth;
            cap_lo[axis] = nose_hi[axis];
        } else {
            nose_lo[axis] = nose_hi[axis] - c.nosing_depth;
            cap_hi[axis] = nose_lo[axis];
        }
        box_bounds(a, c.tread_surface, "floor", cap_lo, cap_hi);
        if c.nosing_depth > 0. {
            box_bounds(
                a,
                c.tread_surface,
                "floor",
                nose_lo,
                nose_hi.with_y(hi.y - 0.004),
            );
            box_bounds(
                a,
                Surface::Rubber,
                "floor",
                nose_lo.with_y(hi.y - 0.004),
                nose_hi,
            );
        }
        if c.stair_structure == StairStructure::ClosedStringer {
            let previous = m.deck.height * i as f32 / m.steps as f32;
            let mut p = lo.with_y(previous);
            let mut q = hi.with_y(bottom);
            if m.ascending_positive() {
                q[axis] = p[axis] + 0.025;
            } else {
                p[axis] = q[axis] - 0.025;
            }
            box_bounds(a, c.tread_surface, "floor", p, q);
        }
        if c.stair_structure != StairStructure::Solid {
            let cross = 2 - axis;
            for side in [lo[cross] + 0.10, hi[cross] - 0.10] {
                let mut p = lo.with_y((bottom - m.riser_height() * 0.5).max(0.));
                let mut q = hi.with_y(bottom);
                p[cross] = side - 0.035;
                q[cross] = side + 0.035;
                box_bounds(a, c.frame_surface, "other_structure", p, q);
            }
        }
    }
    if c.stair_structure != StairStructure::Solid {
        let cross = 1 - m.stair_axis;
        let middle = (m.stair_min[cross] + m.stair_max[cross]) * 0.5;
        for span in m.rail_spans().into_iter().filter(|s| s.stair) {
            let mut p = span.start;
            let mut q = span.end;
            let component = if cross == 0 { 0 } else { 2 };
            let sign = if p[component] < middle { 1. } else { -1. };
            p[component] += sign * 0.10;
            q[component] += sign * 0.10;
            if p.y < c.tread_thickness + 0.025 && q.y > p.y {
                p = p.lerp(q, (c.tread_thickness + 0.025 - p.y) / (q.y - p.y));
            }
            let top_p = p.with_y((p.y - c.tread_thickness).max(0.025));
            let top_q = q.with_y((q.y - c.tread_thickness).max(0.025));
            let bottom_p = top_p.with_y((top_p.y - c.beam_depth).max(0.));
            let bottom_q = top_q.with_y((top_q.y - c.beam_depth).max(0.));
            let mut extrusion = Vec3::ZERO;
            extrusion[component] = 0.065;
            super::super::construction::panel(
                a.part(c.frame_surface, "other_structure"),
                &[
                    bottom_p + extrusion * 0.5,
                    bottom_q + extrusion * 0.5,
                    top_q + extrusion * 0.5,
                    top_p + extrusion * 0.5,
                ],
                extrusion,
            );
        }
    }
    let mut posts = Vec::new();
    for span in m.rail_spans() {
        super::super::guardrail::build(
            a,
            &c.guard,
            super::super::guardrail::GuardRun {
                start: span.start,
                end: span.end,
                height: m.rail_height,
                stair_rise: if span.stair && (span.start.y - span.end.y).abs() > 1e-5 {
                    m.riser_height()
                } else {
                    0.
                },
            },
            |p| if span.stair { m.stair_support(p) } else { p.y },
            &mut posts,
        );
    }
}

fn support_column(a: &mut Assembly, c: &Construction, lo: Vec3, hi: Vec3) {
    let size = hi - lo;
    let center = (lo + hi) * 0.5;
    match c.support_section {
        SupportSection::Square => box_bounds(a, c.frame_surface, "other_structure", lo, hi),
        SupportSection::Round => {
            // Elliptical columns vary depth independently; the same box
            // remains conservative for placement and camera clearance.
            a.part(c.frame_surface, "other_structure").cylinder(
                0.5,
                size.y,
                Transform::from_translation(center).with_scale(Vec3::new(size.x, 1., size.z)),
            );
        }
        SupportSection::ISection => {
            let flange = size.z * 0.13;
            box_bounds(
                a,
                c.frame_surface,
                "other_structure",
                lo,
                hi.with_z(lo.z + flange),
            );
            box_bounds(
                a,
                c.frame_surface,
                "other_structure",
                lo.with_z(hi.z - flange),
                hi,
            );
            box_bounds(
                a,
                c.frame_surface,
                "other_structure",
                Vec3::new(center.x - size.x * 0.07, lo.y, lo.z + flange),
                Vec3::new(center.x + size.x * 0.07, hi.y, hi.z - flange),
            );
        }
    }
}
