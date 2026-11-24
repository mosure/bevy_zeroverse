//! Measurements of the built envelope, including zero counts for absent features.
mod mezzanine;
mod sprinklers;
use super::{CoverageReport, IndoorManifest};
use crate::scene::procedural_indoor::{
    architecture::doors::VisionShape, envelope::polygon, metrics_sort::NumericCollector,
};
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
    numeric.push("exterior_wall_thickness_m", e.wall_thickness() as f64)?;
    for stock in e
        .pillars
        .iter()
        .filter_map(|p| p.profile.as_ref()?.construction)
    {
        *report
            .categories
            .entry("column_stock".into())
            .or_default()
            .entry(format!("{:?}", stock.stock))
            .or_default() += 1;
        numeric.push("column_base_projection_m", stock.base_projection_m as f64)?;
        if let Some(pitch) = stock.sleeve_pitch_m {
            numeric.push("column_sleeve_pitch_m", pitch as f64)?;
        }
    }
    let mut lined_walls = 0;
    for (edge, lining) in e.walls.iter().filter_map(|w| w.lining.map(|s| (w.edge, s))) {
        lined_walls += 1;
        *report
            .categories
            .entry("wall_lining_stock".into())
            .or_default()
            .entry(format!("{:?}", lining.stock))
            .or_default() += 1;
        let cap = |i| e.ceiling_height(scene.room_size, e.footprint[i]);
        numeric.push(
            "wall_lining_height_m",
            f64::from(lining.installed_height(cap(edge), cap((edge + 1) % e.footprint.len()))),
        )?;
        numeric.push("wall_lining_module_width_m", lining.module_width_m as f64)?;
        numeric.push("wall_lining_joint_depth_m", lining.joint_depth_m as f64)?;
        numeric.push("wall_lining_joint_width_m", lining.joint_width_m as f64)?;
        let arrangement = match lining.courses {
            None => "Upright",
            Some(c) => {
                numeric.push("wall_lining_course_height_m", f64::from(c.height_m))?;
                numeric.push("wall_lining_bond_offset", f64::from(c.bond_offset))?;
                if c.phase_count(lining.stock) == 3 {
                    if c.bond_offset > 0. {
                        "ThirdLapRight"
                    } else {
                        "ThirdLapLeft"
                    }
                } else if c.bond_offset == 0. {
                    "Stacked"
                } else {
                    "Staggered"
                }
            }
        };
        *report
            .categories
            .entry("wall_lining_arrangement".into())
            .or_default()
            .entry(arrangement.into())
            .or_default() += 1;
    }
    numeric.push("wall_lining_wall_count", lined_walls as f64)?;
    let adjoining = scene.adjoining_space();
    let adjoining_outline = adjoining.footprint(scene.room_size);
    let height = super::super::architecture::interior_glazing::partition_height(scene);
    let glazing = super::super::architecture::interior_glazing::shared_facade(scene, height);
    let installed_finish = match glazing.openings.first() {
        None => "SolidWall",
        Some(o) => match &o.glazing {
            Some(glass) => {
                numeric.push("shared_glazing_thickness_m", f64::from(glass.thickness_m))?;
                glass.treatment_name()
            }
            None => "LegacyUniform",
        },
    };
    *report
        .categories
        .entry("shared_glazing_treatment".into())
        .or_default()
        .entry(installed_finish.into())
        .or_default() += 1;
    for o in &glazing.openings {
        numeric.push("shared_glazing_bay_width_m", f64::from(o.max.x - o.min.x))?;
    }
    let opening_area: f32 = glazing
        .openings
        .iter()
        .map(|o| (o.max - o.min).element_product())
        .sum();
    let windows = scene
        .adjoining_elevation
        .as_ref()
        .and_then(|p| p.windows.as_ref());
    let apertures = windows.map_or(&[][..], |w| w.facade.openings.as_slice());
    numeric.push("adjoining_exterior_window_count", apertures.len() as f64)?;
    numeric.push(
        "adjoining_exterior_window_area_m2",
        apertures.iter().map(|o| f64::from(o.area())).sum(),
    )?;
    for opening in apertures {
        numeric.push("adjoining_exterior_sill_m", f64::from(opening.min.y))?;
    }
    for (key, value) in [
        ("adjoining_full_width_depth_m", adjoining.depth_m),
        ("adjoining_max_depth_m", adjoining.max_depth()),
        ("adjoining_area_m2", polygon::area(&adjoining_outline)),
        ("shared_glazing_rough_area_m2", opening_area),
        (
            "shared_glazing_fraction_of_clear_wall",
            opening_area / (height * scene.room_size.x),
        ),
        ("shared_glazing_openings", glazing.openings.len() as f32),
    ] {
        numeric.push(key, value as f64)?;
    }
    for (key, value) in [
        (
            "adjoining_plan",
            if adjoining.side_return.is_some() {
                "Stepped"
            } else {
                "Rectangular"
            }
            .to_owned(),
        ),
        ("adjoining_floor", format!("{:?}", adjoining.floor)),
    ] {
        *report
            .categories
            .entry(key.into())
            .or_default()
            .entry(value)
            .or_default() += 1;
    }
    let doors = scene.door_installations();
    let finishes = scene.program.as_ref().and_then(|p| p.finishes.as_ref());
    if let Some(trim) = finishes.and_then(|f| f.trim.as_ref()) {
        for (name, section) in [("base_trim", &trim.base), ("crown_trim", &trim.crown)] {
            *report
                .categories
                .entry(format!("{name}_profile"))
                .or_default()
                .entry(
                    section
                        .as_ref()
                        .map_or_else(|| "Absent".into(), |s| format!("{:?}", s.profile)),
                )
                .or_default() += 1;
            if let Some(s) = section {
                for (dimension, value) in [
                    ("height_m", s.height_m),
                    ("depth_m", s.depth_m),
                    ("shoulder", s.shoulder),
                    ("tip_fraction", s.tip_fraction),
                ] {
                    numeric.push(&format!("{name}_{dimension}"), value as f64)?;
                }
                *report
                    .categories
                    .entry(format!("{name}_material"))
                    .or_default()
                    .entry(format!("{:?}", s.surface))
                    .or_default() += 1;
            }
        }
    }
    if let Some(p) = finishes.and_then(|f| f.interior_glazing.as_ref()) {
        for (key, value) in [
            ("interior_glazing_sill_m", p.sill_m),
            ("interior_glazing_header_m", p.header_m),
            ("interior_glazing_pier_fraction", p.pier_fraction),
            ("interior_frame_width_m", p.frame_width_m),
            ("interior_frame_depth_m", p.frame_depth_m),
            ("interior_mullion_pitch_m", p.mullion_pitch_m),
        ] {
            numeric.push(key, value as f64)?;
        }
    }
    let fixture = super::super::architecture::fixtures::parameters(scene);
    use super::super::architecture::fixtures::{optical_insert, OpticalInsert};
    let insert = optical_insert(scene, &fixture);
    let optic = match insert {
        OpticalInsert::Basket { width, .. } => {
            numeric.push("center_basket_diffuser_width_fraction", width as f64)?;
            "CenterBasket"
        }
        OpticalInsert::Cup { throat, .. } => {
            numeric.push("downlight_throat_to_rim_ratio", throat as f64)?;
            "RecessedCup"
        }
        OpticalInsert::Plain => "Standard",
    };
    *report
        .categories
        .entry("luminaire_optical_insert".into())
        .or_default()
        .entry(optic.into())
        .or_default() += 1;
    *report
        .categories
        .entry("luminaire_construction".into())
        .or_default()
        .entry(format!("{:?}", fixture.shape))
        .or_default() += 1;
    let positions = super::super::architecture::fixture_positions(scene);
    numeric.push(
        "adjoining_ceiling_fixtures",
        (positions.len() - super::super::architecture::primary_fixture_count(scene, &positions))
            as f64,
    )?;
    for position in positions {
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
        for (hole, radius) in d.vision_apertures() {
            let shape = match d.program.vision_light.map(|v| v.shape) {
                None => "Rectangular",
                Some(VisionShape::Porthole { .. }) => "Porthole",
                Some(VisionShape::Slot { .. }) => "Slot",
                Some(VisionShape::Rounded { .. }) => "Rounded",
            };
            *report
                .categories
                .entry("door_vision_shape".into())
                .or_default()
                .entry(shape.into())
                .or_default() += 1;
            for (key, value) in [
                ("door_vision_width_m", hole.width()),
                ("door_vision_height_m", hole.height()),
                ("door_vision_center_height_m", hole.center().y),
                ("door_vision_corner_radius_m", radius),
                (
                    "door_vision_cutout_area_m2",
                    hole.width() * hole.height() - (4. - std::f32::consts::PI) * radius.powi(2),
                ),
            ] {
                numeric.push(key, value as f64)?;
            }
        }
        if let Some(h) = d.program.swing_hanging {
            for (key, value) in [
                ("swing_door_face_setback_m", h.reveal_m),
                ("swing_door_stop_projection_m", h.stop_projection_m),
                ("swing_door_hinges_per_leaf", h.hinges_per_leaf(d) as f32),
                ("swing_door_hinge_length_m", h.hinge_length_m),
            ] {
                numeric.push(key, value as f64)?;
            }
        }
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
    let boardroom = scene.executive_boardroom();
    if let Some(b) = boardroom {
        numeric.push("boardroom_perimeter_seats", b.seat_count as f64)?;
        numeric.push(
            "boardroom_tabletop_floor_fraction",
            (b.tabletop_area_m2 / b.room_floor_area_m2) as f64,
        )?;
        numeric.push(
            "boardroom_seating_envelope_floor_fraction",
            (b.seating_envelope_area_m2 / b.room_floor_area_m2) as f64,
        )?;
    }
    serde_json::to_writer(
        &mut *writer,
        &serde_json::json!({"seed": scene.seed, "room_size": scene.room_size, "executive_boardroom": boardroom, "envelope": e, "adjoining": adjoining, "adjoining_elevation": scene.adjoining_elevation, "adjoining_footprint": adjoining_outline,
        "shared_glazing_openings": glazing.openings, "shared_glazing_fraction_of_clear_wall": opening_area/(height*scene.room_size.x), "finishes": finishes, "ceiling_installation":scene.ceiling_installation, "luminaire_program":fixture, "luminaire_optical_insert":insert, "topology": topology, "functional_programs":scene.functional_programs(), "support_regions":super::super::layout::functional::support_regions(scene), "doors":doors,"partitions": scene.program.as_ref().map(|p| &p.partitions)}),
    )?;
    writeln!(writer)?;
    if let Some(ceiling) = &scene.ceiling_installation {
        let coverage = if let Some(grid) = &ceiling.room_grid {
            for (name, value) in [
                ("ceiling_grid_area_m2", polygon::area(&e.footprint)),
                ("ceiling_grid_pitch_x_m", grid.pitch_m.x),
                ("ceiling_grid_pitch_z_m", grid.pitch_m.y),
                ("ceiling_grid_recessed_lights", grid.troffers.len() as f32),
                (
                    "ceiling_grid_air_terminals",
                    grid.air_terminals.len() as f32,
                ),
            ] {
                numeric.push(name, value as f64)?;
            }
            for air in &grid.air_terminals {
                *report
                    .categories
                    .entry("ceiling_air_terminal".into())
                    .or_default()
                    .entry(format!("{:?}", air.pattern))
                    .or_default() += 1;
            }
            "WallToWall"
        } else if ceiling.exposed_slab {
            "Exposed"
        } else if ceiling.modules.is_empty() {
            "Plain"
        } else {
            "SuspendedModules"
        };
        *report
            .categories
            .entry("ceiling_coverage".into())
            .or_default()
            .entry(coverage.into())
            .or_default() += 1;
        if let Some(frame) = &ceiling.framing {
            numeric.push("ceiling_frame_members", frame.members.len() as f64)?;
            for m in &frame.members {
                for (key, value) in [
                    ("ceiling_frame_depth_m", m.depth),
                    ("ceiling_frame_span_m", m.top[0].distance(m.top[1])),
                    ("ceiling_frame_width_m", m.top[0].distance(m.top[3])),
                ] {
                    numeric.push(key, value as f64)?;
                }
                for (key, value) in [
                    ("ceiling_frame_role", format!("{:?}", m.role)),
                    ("ceiling_frame_section", format!("{:?}", m.section)),
                    ("ceiling_frame_material", format!("{:?}", m.material)),
                ] {
                    *report
                        .categories
                        .entry(key.into())
                        .or_default()
                        .entry(value)
                        .or_default() += 1;
                }
            }
        }
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
            numeric.push(
                "ceiling_hangers",
                m.suspension.as_ref().map_or(0, Vec::len) as f64,
            )?;
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
                        if let Some(stock) = m.duct_register_stock(scene.generator_version) {
                            for i in 0..branches.max(1) {
                                let site = stock.site(m, branches, i);
                                *report
                                    .categories
                                    .entry("ceiling_duct_register_pattern".into())
                                    .or_default()
                                    .entry(format!("{:?}", stock.pattern))
                                    .or_default() += 1;
                                *report
                                    .categories
                                    .entry("ceiling_duct_register_finish".into())
                                    .or_default()
                                    .entry(
                                        if stock.coated {
                                            "factory_coated"
                                        } else {
                                            "duct_metal"
                                        }
                                        .into(),
                                    )
                                    .or_default() += 1;
                                for (key, value) in [
                                    ("ceiling_duct_register_length_m", site.size.x),
                                    ("ceiling_duct_register_width_m", site.size.y),
                                    ("ceiling_duct_register_station", site.station),
                                ] {
                                    numeric.push(key, f64::from(value))?;
                                }
                            }
                        }
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
                    CeilingShape::LayIn {
                        aspect,
                        runner_base_m,
                        tile,
                    } => {
                        for (key, value) in [
                            ("ceiling_grid_aspect", aspect),
                            ("ceiling_grid_pitch_m", m.pitch_m),
                            ("ceiling_tile_recess_m", tile.drop_m),
                            ("ceiling_tile_thickness_m", tile.thickness_m),
                            ("ceiling_runner_base_m", runner_base_m),
                            ("ceiling_runner_flange_m", tile.flange_width_m),
                        ] {
                            numeric.push(key, f64::from(value))?;
                        }
                        for (key, value) in [
                            ("ceiling_tile_edge", format!("{:?}", tile.edge)),
                            ("ceiling_tile_facing", format!("{:?}", tile.facing)),
                        ] {
                            *report
                                .categories
                                .entry(key.into())
                                .or_default()
                                .entry(value)
                                .or_default() += 1;
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
                        sprinklers::record(m, scene.generator_version, report, numeric)?;
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
        mezzanine::record(m, report, numeric)?;
        for (key, value) in [
            ("mezzanine_height_m", m.deck.height),
            ("mezzanine_stair_rise_m", m.deck.height / m.steps as f32),
            ("mezzanine_stair_tread_m", m.tread_depth()),
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
    if let Some(g) = &e.tier_guard {
        if !e.tier_guards().is_empty() {
            numeric.push("tier_guard_height_m", g.height as f64)?;
            mezzanine::guard("tier_guard", &g.section, report, numeric)?;
        }
    }
    Ok(())
}
