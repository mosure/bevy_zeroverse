use super::{category, io, Categories, IndoorManifest, IndoorObject, NumericCollector};
use crate::scene::procedural_indoor::{
    envelope::polygon,
    materials::{rug_design::RugDesign, rugs, Surface},
    objects::floor::rug,
};
use bevy::math::Vec3Swizzles;

pub(super) fn record(
    scene: &IndoorManifest,
    o: &IndoorObject,
    categories: &mut Categories,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    let Some(cut) = rug::parameters(o) else {
        return Ok(());
    };
    category(categories, "rug_cut", format!("{:?}", cut.cut));
    category(categories, "rug_binding", format!("{:?}", cut.binding));
    numeric.push("rug_binding_width_m", f64::from(cut.binding_width_m))?;
    let area = polygon::area(&cut.outline(o.size.xz()));
    numeric.push(
        "rug_footprint_fill_fraction",
        f64::from(area / (o.size.x * o.size.z)),
    )?;

    let Some(program) = &scene.program else {
        return Ok(());
    };
    let slot = rugs::slot(o.finish_seed.unwrap_or(o.seed));
    let seed = rugs::dye_seed(program.materials[Surface::FabricAlt as usize].seed, slot);
    let design = RugDesign::sample_at_version(seed, scene.generator_version);
    let Some(w) = design.weave else { return Ok(()) };
    let count = if design.center > 0. { w.count } else { 0 };
    let guards = w
        .guard_half_widths
        .iter()
        .filter(|&&width| width > 0.)
        .count();
    category(categories, "rug_medallion_count", count);
    category(categories, "rug_guard_stripes", guards);
    let raised = w.pile_contrast > 0.
        && (count > 0 || guards > 0)
        && program.materials[Surface::FabricAlt as usize].relief_m > 0.;
    category(
        categories,
        "rug_pattern_pile",
        if raised { "high_low" } else { "level" },
    );
    if count > 1 {
        let longitudinal = rug::warp_along_z(o) == (o.size.z > o.size.x);
        category(
            categories,
            "rug_medallion_row_axis",
            if longitudinal {
                "longitudinal"
            } else {
                "transverse"
            },
        );
    }
    if count > 0 {
        numeric.push(
            "rug_medallion_row_diameter_uv",
            f64::from(2. / w.inverse_radii[0]),
        )?;
        numeric.push(
            "rug_medallion_cross_diameter_uv",
            f64::from(2. / w.inverse_radii[1]),
        )?;
    }
    Ok(())
}
