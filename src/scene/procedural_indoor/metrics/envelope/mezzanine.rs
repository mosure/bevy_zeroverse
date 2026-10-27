use super::*;
use crate::scene::procedural_indoor::envelope::{guardrail::GuardSection, Mezzanine};

pub(super) fn record(
    m: &Mezzanine,
    report: &mut CoverageReport,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    for (key, value) in [
        ("mezzanine_slab_thickness_m", m.thickness),
        ("mezzanine_deck_width_m", m.deck.max.x - m.deck.min.x),
        ("mezzanine_deck_depth_m", m.deck.max.y - m.deck.min.y),
        ("mezzanine_guard_height_m", m.rail_height),
        ("mezzanine_support_count", m.columns().len() as f32),
    ] {
        numeric.push(key, value as f64)?;
    }
    category(
        report,
        "mezzanine_stair_axis",
        if m.stair_axis == 0 { "X" } else { "Z" },
    );
    category(
        report,
        "mezzanine_stair_direction",
        if m.ascending_positive() {
            "Positive"
        } else {
            "Negative"
        },
    );
    if let Some(c) = &m.construction {
        numeric.push(
            "mezzanine_landing_support_count",
            if c.landing.is_some()
                && c.stair_structure
                    != crate::scene::procedural_indoor::envelope::mezzanine::StairStructure::Solid
            {
                2.
            } else {
                0.
            },
        )?;
        for (key, value) in [
            ("mezzanine_tread_thickness_m", c.tread_thickness),
            ("mezzanine_nosing_depth_m", c.nosing_depth),
            ("mezzanine_support_width_m", c.support_width),
            ("mezzanine_support_depth_m", c.support_depth),
            ("mezzanine_support_pitch_m", c.support_pitch),
            ("mezzanine_beam_depth_m", c.beam_depth),
            (
                "mezzanine_landing_extra_run_m",
                c.landing.as_ref().map_or(0., |l| l.extra_run),
            ),
        ] {
            numeric.push(key, value as f64)?;
        }
        for (key, value) in [
            (
                "mezzanine_stair_structure",
                format!("{:?}", c.stair_structure),
            ),
            (
                "mezzanine_support_section",
                format!("{:?}", c.support_section),
            ),
            ("mezzanine_frame_material", format!("{:?}", c.frame_surface)),
            ("mezzanine_deck_material", format!("{:?}", c.deck_surface)),
            ("mezzanine_tread_material", format!("{:?}", c.tread_surface)),
            ("mezzanine_landing", c.landing.is_some().to_string()),
        ] {
            category(report, key, &value);
        }
        guard("mezzanine_guard", &c.guard, report, numeric)?;
    }
    Ok(())
}

fn category(report: &mut CoverageReport, key: &str, value: &str) {
    *report
        .categories
        .entry(key.into())
        .or_default()
        .entry(value.into())
        .or_default() += 1;
}

pub(super) fn guard(
    prefix: &str,
    g: &GuardSection,
    report: &mut CoverageReport,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    for (key, value) in [
        ("post_width_m", g.post_width),
        ("post_pitch_m", g.post_pitch),
        ("bay_jitter", g.bay_jitter),
        ("infill_width_m", g.infill_width),
        ("infill_gap_m", g.infill_gap),
        ("panel_thickness_m", g.panel_thickness),
        ("handrail_width_m", g.handrail_width),
        ("handrail_depth_m", g.handrail_depth),
    ] {
        numeric.push(&format!("{prefix}_{key}"), value as f64)?;
    }
    for (key, value) in [
        ("infill", format!("{:?}", g.infill)),
        ("frame_material", format!("{:?}", g.frame)),
        ("handrail_material", format!("{:?}", g.handrail)),
        ("panel_material", format!("{:?}", g.panel)),
        ("round_posts", g.round_posts.to_string()),
        ("round_handrail", g.round_handrail.to_string()),
    ] {
        category(report, &format!("{prefix}_{key}"), &value);
    }
    Ok(())
}
