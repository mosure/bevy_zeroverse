use super::*;
use crate::scene::procedural_indoor::architecture::ceiling::CeilingModule;

/// Stock categories are counted per installation. Branch reach and separation
/// describe the actual mounted positions, including saddle clearance.
pub(super) fn record(
    m: &CeilingModule,
    version: u32,
    report: &mut CoverageReport,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    let Some(run) = m.sprinkler_run(version) else {
        return Ok(());
    };
    for (key, value) in [
        (
            "ceiling_sprinkler_branch_layout",
            format!("{:?}", run.layout),
        ),
        ("ceiling_sprinkler_frame_finish", format!("{:?}", run.frame)),
        ("ceiling_sprinkler_bulb_tint", format!("{:?}", run.bulb)),
    ] {
        *report
            .categories
            .entry(key.into())
            .or_default()
            .entry(value)
            .or_default() += 1;
    }
    for (key, value) in [
        ("ceiling_sprinkler_heads_per_run", f32::from(run.count)),
        ("ceiling_sprinkler_deflector_slots", f32::from(run.slots)),
        ("ceiling_sprinkler_deflector_diameter_m", 2. * run.radius),
        ("ceiling_sprinkler_frame_span_m", run.span),
    ] {
        numeric.push(key, f64::from(value))?;
    }
    let mut previous = None;
    for i in 0..run.count {
        let site = run.site(m, i);
        numeric.push("ceiling_sprinkler_lateral_offset_m", f64::from(site.z))?;
        if let Some(x) = previous {
            numeric.push("ceiling_sprinkler_station_spacing_m", f64::from(site.x - x))?;
        }
        previous = Some(site.x);
    }
    Ok(())
}
