//! Measurements of the built envelope, including zero counts for absent features.
use super::{CoverageReport, IndoorManifest};
use crate::scene::procedural_indoor::{envelope::polygon, metrics_sort::NumericCollector};
use bevy::math::Vec3Swizzles;
use std::io::{self, Write};

pub(super) fn record(
    scene: &IndoorManifest,
    report: &mut CoverageReport,
    numeric: &mut NumericCollector,
    writer: &mut impl Write,
) -> io::Result<()> {
    let Some(e) = &scene.envelope else {
        return Ok(());
    };
    serde_json::to_writer(
        &mut *writer,
        &serde_json::json!({"seed": scene.seed, "room_size": scene.room_size, "envelope": e, "partitions": scene.program.as_ref().map(|p| &p.partitions)}),
    )?;
    writeln!(writer)?;
    let area = polygon::area(&e.footprint);
    let mut reflex = 0;
    let mut oblique = 0;
    for i in 0..e.footprint.len() {
        let a = e.footprint[i];
        let b = e.footprint[(i + 1) % e.footprint.len()];
        let c = e.footprint[(i + 2) % e.footprint.len()];
        reflex += usize::from((b - a).perp_dot(c - b) < -0.0001);
        let d = b - a;
        oblique += usize::from(d.x.abs().min(d.y.abs()) > 0.001);
    }
    let arches = scene.program.as_ref().map_or(0, |p| {
        p.partitions.iter().filter(|p| p.arch_rise > 0.).count()
    });
    let roof_min = e
        .footprint
        .iter()
        .map(|&p| e.ceiling_height(scene.room_size, p))
        .fold(f32::INFINITY, f32::min);
    let roof_max = e
        .footprint
        .iter()
        .map(|&p| e.ceiling_height(scene.room_size, p))
        .fold(f32::NEG_INFINITY, f32::max);
    for (key, value) in [
        ("envelope_footprint_area_m2", area),
        (
            "envelope_footprint_fraction",
            area / (scene.room_size.x * scene.room_size.z),
        ),
        (
            "envelope_perimeter_m",
            polygon::edges(&e.footprint)
                .map(|(a, b)| a.distance(b))
                .sum(),
        ),
        ("envelope_vertices", e.footprint.len() as f32),
        ("envelope_reflex_corners", reflex as f32),
        ("envelope_oblique_edges", oblique as f32),
        ("envelope_ceiling_min_m", roof_min),
        ("envelope_ceiling_max_m", roof_max),
        (
            "envelope_ceiling_slope_degrees",
            (e.ceiling_drop / scene.room_size.xz())
                .length()
                .atan()
                .to_degrees(),
        ),
        ("envelope_floor_min_m", e.minimum_floor()),
        (
            "envelope_floor_max_m",
            e.floor_patches.iter().map(|p| p.height).fold(0., f32::max),
        ),
        ("envelope_floor_patches", e.floor_patches.len() as f32),
        ("envelope_pillars", e.pillars.len() as f32),
        ("envelope_archways", arches as f32),
        (
            "envelope_mezzanine_area_m2",
            e.mezzanine
                .as_ref()
                .map_or(0., |m| (m.deck.max - m.deck.min).element_product()),
        ),
    ] {
        numeric.push(key, value as f64)?;
    }
    for (key, present) in [
        ("envelope_concave", reflex > 0),
        ("envelope_oblique", oblique > 0),
        ("envelope_sloped_ceiling", e.ceiling_drop.length() > 0.001),
        (
            "envelope_raised_floor",
            e.floor_patches.iter().any(|p| p.height > 0.001),
        ),
        ("envelope_depressed_floor", e.minimum_floor() < -0.001),
        ("envelope_mezzanine", e.mezzanine.is_some()),
        ("envelope_pillars", !e.pillars.is_empty()),
        ("envelope_archways", arches > 0),
    ] {
        *report
            .categories
            .entry(key.into())
            .or_default()
            .entry(present.to_string())
            .or_default() += 1;
    }
    if let Some(m) = &e.mezzanine {
        for (key, value) in [
            ("mezzanine_height_m", m.deck.height),
            ("mezzanine_stair_rise_m", m.deck.height / m.steps as f32),
            (
                "mezzanine_stair_tread_m",
                (m.stair_max[m.stair_axis] - m.stair_min[m.stair_axis]) / m.steps as f32,
            ),
            ("mezzanine_stair_steps", m.steps as f32),
            (
                "mezzanine_stair_width_m",
                m.stair_max[1 - m.stair_axis] - m.stair_min[1 - m.stair_axis],
            ),
            (
                "mezzanine_furniture",
                scene
                    .objects
                    .iter()
                    .filter(|o| {
                        o.solid && !o.neighbor && (o.position.y - m.deck.height).abs() < 0.002
                    })
                    .count() as f32,
            ),
        ] {
            numeric.push(key, value as f64)?;
        }
    }
    Ok(())
}
