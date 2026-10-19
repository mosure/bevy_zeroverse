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
    let doors = scene.door_installations();
    let fixture = super::super::architecture::fixtures::parameters(scene);
    *report
        .categories
        .entry("luminaire_construction".into())
        .or_default()
        .entry(format!("{:?}", fixture.shape))
        .or_default() += 1;
    for position in super::super::architecture::fixture_positions(scene) {
        let size = super::super::architecture::fixtures::size(scene, position);
        numeric.push("luminaire_body_width_m", size.x as f64)?;
        numeric.push("luminaire_body_depth_m", size.y as f64)?;
        numeric.push("luminaire_body_length_m", size.z as f64)?;
        numeric.push(
            "luminaire_emitting_area_m2",
            super::super::architecture::fixtures::emitting_area_at(scene, position) as f64,
        )?;
    }
    numeric.push("door_count", doors.len() as f64)?;
    for d in &doors {
        for (key, value) in [
            ("door_mechanism", format!("{:?}", d.program.mechanism)),
            ("door_panel", format!("{:?}", d.program.panel)),
            ("door_handle", format!("{:?}", d.program.handle)),
            ("door_finish", format!("{:?}", d.program.finish)),
        ] {
            *report
                .categories
                .entry(key.into())
                .or_default()
                .entry(value)
                .or_default() += 1;
        }
        for (key, value) in [
            ("door_opening", d.program.opening),
            ("door_width_m", d.width),
            ("door_height_m", d.height),
            ("door_rotor_phase", d.program.rotor_phase),
            ("door_x_m", d.center.x),
            ("door_z_m", d.center.z),
        ] {
            numeric.push(key, value as f64)?;
        }
    }
    let topology = scene.room_topology();
    numeric.push("room_topology_regions", topology.regions.len() as f64)?;
    numeric.push("room_topology_portals", topology.portals.len() as f64)?;
    numeric.push(
        "room_topology_open_connections",
        topology.open_connections.len() as f64,
    )?;
    numeric.push(
        "room_topology_boundary_partitions",
        topology.boundary_partitions.len() as f64,
    )?;
    for r in &topology.regions {
        numeric.push(
            if r.primary {
                "primary_region_area_m2"
            } else {
                "other_region_area_m2"
            },
            r.area_m2 as f64,
        )?;
    }
    serde_json::to_writer(
        &mut *writer,
        &serde_json::json!({"seed": scene.seed, "room_size": scene.room_size, "envelope": e, "ceiling_installation":scene.ceiling_installation, "luminaire_program":fixture, "topology": topology, "functional_programs":scene.functional_programs(), "support_regions":super::super::layout::functional::support_regions(scene), "doors":doors,"partitions": scene.program.as_ref().map(|p| &p.partitions)}),
    )?;
    writeln!(writer)?;
    if let Some(ceiling) = &scene.ceiling_installation {
        numeric.push("ceiling_installation_modules", ceiling.modules.len() as f64)?;
        numeric.push(
            "ceiling_installation_footprint_m2",
            ceiling
                .modules
                .iter()
                .map(|m| (m.size.x * m.size.z) as f64)
                .sum(),
        )?;
        *report
            .categories
            .entry("ceiling_exposed_slab".into())
            .or_default()
            .entry(ceiling.exposed_slab.to_string())
            .or_default() += 1;
        for m in &ceiling.modules {
            *report
                .categories
                .entry("ceiling_system".into())
                .or_default()
                .entry(format!("{:?}", m.system))
                .or_default() += 1;
            numeric.push("ceiling_installation_headroom_m", m.headroom(scene) as f64)?;
            for (key, value) in [
                ("ceiling_body_depth_m", m.size.y),
                (
                    "ceiling_plenum_gap_m",
                    m.support_top_m - (m.center.y + m.size.y * 0.5),
                ),
                ("ceiling_body_bottom_m", m.center.y - m.size.y * 0.5),
                ("ceiling_run_length_m", m.size.x.max(m.size.z)),
                ("ceiling_run_width_m", m.size.x.min(m.size.z)),
            ] {
                numeric.push(key, value as f64)?;
            }
            use crate::scene::procedural_indoor::architecture::ceiling::CeilingShape;
            if let Some(detail) = &m.detail {
                match detail.shape {
                    CeilingShape::Duct {
                        section,
                        end_scale,
                        bend,
                        rise,
                        branches,
                        seam_pitch_m,
                    } => {
                        for (key, value) in [
                            ("ceiling_duct_section", section),
                            ("ceiling_duct_end_scale", end_scale),
                            ("ceiling_duct_bend", bend),
                            ("ceiling_duct_rise", rise),
                            ("ceiling_duct_branches", branches as f32),
                            ("ceiling_duct_seam_pitch_m", seam_pitch_m),
                        ] {
                            numeric.push(key, value as f64)?;
                        }
                    }
                    CeilingShape::Grid { aspect, recess_m } => {
                        numeric.push("ceiling_grid_aspect", aspect as f64)?;
                        numeric.push("ceiling_grid_pitch_m", m.pitch_m as f64)?;
                        if m.system == crate::scene::procedural_indoor::architecture::ceiling::CeilingSystem::SuspendedTiles {
                            numeric.push("ceiling_tile_recess_m",recess_m as f64)?;
                        }
                    }
                    CeilingShape::Conduit {
                        count,
                        radius_m,
                        bend,
                        rise,
                        ..
                    } => {
                        for (key, value) in [
                            ("ceiling_conduit_count", count as f32),
                            (
                                "ceiling_conduit_radius_m",
                                radius_m
                                    .min(m.size.x.min(m.size.z) * 0.56 / count as f32 * 0.32)
                                    .min(m.size.y * 0.12),
                            ),
                            ("ceiling_conduit_bend", bend),
                            ("ceiling_conduit_rise", rise),
                        ] {
                            numeric.push(key, value as f64)?;
                        }
                    }
                    CeilingShape::Sprinkler {
                        head_pitch_m,
                        branch_reach,
                        upright,
                        ..
                    } => {
                        numeric.push("ceiling_sprinkler_pitch_m", head_pitch_m as f64)?;
                        numeric.push("ceiling_sprinkler_branch_reach", branch_reach as f64)?;
                        *report
                            .categories
                            .entry("ceiling_sprinkler_orientation".into())
                            .or_default()
                            .entry(if upright { "upright" } else { "pendent" }.into())
                            .or_default() += 1;
                    }
                    CeilingShape::Truss { web, .. } => {
                        *report
                            .categories
                            .entry("ceiling_truss_web".into())
                            .or_default()
                            .entry(format!("{web:?}"))
                            .or_default() += 1;
                    }
                    _ => {}
                }
            }
        }
    }
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
        ("envelope_ceiling_patches", e.soffits.len() as f32),
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
    for p in &e.pillars {
        if let Some(profile) = &p.profile {
            for (key, value) in [
                ("column_aspect", profile.aspect),
                ("column_roundness", profile.roundness),
                ("column_taper", profile.taper),
                ("column_rotation_radians", profile.rotation),
                ("column_collar_height_m", profile.collar_height),
            ] {
                numeric.push(key, value as f64)?;
            }
        }
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
