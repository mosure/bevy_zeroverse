//! Built furnishing/content diagnostics, with the same finish seeds and physical
//! emitters used by the renderer. Equipment fields exclude dormant parameters.
use super::super::{
    cameras::viewpoint::ViewpointProgram,
    layout::{IndoorManifest, IndoorObject, ObjectKind},
    materials::{self, program::MaterialRecipe, Surface},
    metrics_sort::NumericCollector,
    objects::equipment,
};
use bevy::math::Vec3Swizzles;
use std::{collections::BTreeMap, io};

mod kitchen_fittings;
mod service_equipment;
mod service_furniture;

type Categories = BTreeMap<String, BTreeMap<String, usize>>;

fn category(categories: &mut Categories, key: &str, value: impl ToString) {
    *categories
        .entry(key.into())
        .or_default()
        .entry(value.to_string())
        .or_default() += 1;
}

pub(super) fn record_scene(
    scene: &IndoorManifest,
    numeric: &mut NumericCollector,
    categories: &mut Categories,
) -> io::Result<()> {
    let installation = super::super::layout::electrical_parameters(scene);
    category(categories, "electrical_standard", installation.standard);
    for (key, value) in [
        (
            "electrical_outlet_height_prior_m",
            installation.outlet_height,
        ),
        (
            "electrical_switch_height_prior_m",
            installation.switch_height,
        ),
        ("electrical_plate_depth_prior_m", installation.plate_depth),
        (
            "electrical_worktop_probability",
            installation.worktop_probability,
        ),
    ] {
        numeric.push(key, value as f64)?;
    }
    for object in scene
        .objects
        .iter()
        .filter(|o| super::super::photometry::is_lamp(o.kind))
    {
        let source = super::super::photometry::lamp(scene, object);
        for (key, value) in [
            ("kelvin", source.kelvin),
            ("tint", source.tint),
            ("dimmer", source.dimmer),
            ("lumens", source.lumens),
        ] {
            numeric.push(&format!("source_{:?}_{key}", object.kind), value as f64)?;
        }
    }
    let groups = &scene.assembly_diagnostics;
    numeric.push(
        "feature_wall_edges",
        materials::wall::feature_edges(scene).len() as f64,
    )?;
    numeric.push("functional_programs", groups.programs.len() as f64)?;
    for p in &groups.programs {
        category(
            categories,
            "functional_activity",
            format!("{:?}", p.activity),
        );
        category(categories, "functional_role", format!("{:?}", p.role));
        category(
            categories,
            "functional_capacity_unit",
            format!("{:?}", p.capacity_unit),
        );
        for (name, value) in [
            ("functional_requested_capacity", p.requested_capacity as f64),
            ("functional_accepted_capacity", p.accepted_capacity as f64),
            (
                "functional_rejected_candidates",
                p.rejected_candidates as f64,
            ),
            ("functional_aisle_width_m", p.parameters.aisle_metres as f64),
            (
                "functional_row_pitch_m",
                p.parameters.row_pitch_metres as f64,
            ),
            (
                "functional_seat_pitch_m",
                p.parameters.seat_pitch_metres as f64,
            ),
            (
                "functional_tier_rise_m",
                p.parameters.tier_rise_metres as f64,
            ),
            (
                "functional_group_rotation_rad",
                p.parameters.group_rotation_radians as f64,
            ),
            (
                "functional_reserved_area_m2",
                p.reservations
                    .iter()
                    .map(|r| ((r.max - r.min).x * (r.max - r.min).y) as f64)
                    .sum(),
            ),
        ] {
            numeric.push(name, value)?;
        }
    }
    numeric.push("kitchen_accepted_groups", groups.kitchens.len() as f64)?;
    for group in &groups.kitchens {
        category(categories, "kitchen_configuration", &group.configuration);
        category(categories, "kitchen_fitted_to_wall", group.fitted_to_wall);
        numeric.push(
            "kitchen_worktop_height_m",
            group.worktop_height_metres as f64,
        )?;
        if let Some(p) = &group.sampling {
            numeric.push("kitchen_unit_gap_m", p.unit_gap_m as f64)?;
            numeric.push("kitchen_run_offset_m", p.run_offset_m as f64)?;
            numeric.push("kitchen_appliance_coherence", p.appliance_coherence as f64)?;
        }
        if let Some(o) = group.units.first().and_then(|&id| scene.objects.get(id)) {
            category(
                categories,
                "kitchen_run_first_unit",
                format!("{:?}", o.kind),
            );
        }
        for kind in &group.requested {
            category(categories, "kitchen_requested_units", format!("{kind:?}"));
        }
        numeric.push("kitchen_accepted_units", group.units.len() as f64)?;
        numeric.push("kitchen_accepted_props", group.props.len() as f64)?;
        numeric.push("kitchen_rejected_optional", group.rejected_optional as f64)?;
    }
    for (name, n) in [
        (
            "assembly_proposed_workstations",
            groups.proposed_workstations,
        ),
        ("assembly_proposed_meetings", groups.proposed_meetings),
        (
            "assembly_rejected_surface_attempts",
            groups.rejected_surfaces,
        ),
        (
            "assembly_rejected_seating_attempts",
            groups.rejected_seating,
        ),
        ("assembly_rejected_meetings", groups.rejected_meetings),
        ("assembly_accepted_groups", groups.accepted.len()),
    ] {
        numeric.push(name, n as f64)?;
    }
    for group in &groups.accepted {
        let p = scene.objects[group.surface].position;
        let region = if scene.in_primary_room(p, 0.) {
            "primary"
        } else {
            "main_other"
        };
        for (name, n) in [
            ("requested_capacity", group.requested_capacity),
            ("accepted_capacity", group.seats.len()),
            ("attempts", group.attempts),
        ] {
            numeric.push(
                &format!("assembly_{region}_{:?}_{name}", group.role),
                n as f64,
            )?;
        }
    }
    for (kind, count) in ["building", "garden", "courtyard", "pergola"]
        .into_iter()
        .zip(super::super::architecture::outdoors::counts(scene))
    {
        numeric.push(&format!("exterior_{kind}_parcels"), count as f64)?;
    }
    if let Some(fraction) = super::super::architecture::outdoors::window_context_fraction(scene) {
        numeric.push("exterior_window_context_fraction", fraction as f64)?;
    }
    for (name, value) in super::super::architecture::outdoors::measurements(scene) {
        numeric.push(name, value as f64)?;
    }
    numeric.push(
        "floor_power_connections",
        scene
            .objects
            .iter()
            .filter(|o| {
                o.kind == ObjectKind::Cable
                    && o.cable_target
                        .and_then(|id| scene.objects.get(id))
                        .is_some_and(|target| target.kind == ObjectKind::FloorOutlet)
            })
            .count() as f64,
    )?;
    let p = ViewpointProgram::sample(scene.seed);
    let wall = super::super::layout::wall_composition::WallComposition::sample(scene);
    for (name, value) in [
        ("wall_composition_prior_gallery_weight", wall.weights[0]),
        (
            "wall_composition_prior_presentation_weight",
            wall.weights[1],
        ),
        ("wall_composition_prior_mixed_weight", wall.weights[2]),
        ("wall_composition_prior_span_fraction", wall.coverage),
        ("wall_composition_prior_picture_aspect", wall.picture_aspect),
        ("wall_composition_prior_spacing_m", wall.spacing_m),
        ("wall_composition_prior_eye_height_m", wall.eye_height_m),
        ("wall_composition_prior_asymmetry", wall.asymmetry),
        ("wall_composition_prior_size_variation", wall.size_variation),
        (
            "wall_composition_prior_height_variation_m",
            wall.height_variation_m,
        ),
        (
            "wall_composition_prior_aligned_fraction",
            wall.aligned_fraction,
        ),
        (
            "wall_composition_prior_stacked_fraction",
            wall.stacked_fraction,
        ),
    ] {
        numeric.push(name, value as f64)?;
    }
    for wall in super::super::layout::wall_composition::mounted_art_walls(scene) {
        let stats = super::super::layout::wall_composition::ArtWallStats::measure(&wall);
        numeric.push("wall_art_count_per_wall", stats.count as f64)?;
        // Single-picture walls have no alignment/size comparison denominator.
        if stats.pairs > 0 {
            for (name, value) in [
                (
                    "wall_art_center_height_span_m",
                    stats.center_height_span_m as f64,
                ),
                ("wall_art_area_ratio", stats.area_ratio as f64),
                (
                    "wall_art_aligned_pair_fraction",
                    stats.aligned_pairs as f64 / stats.pairs as f64,
                ),
                (
                    "wall_art_matching_size_pair_fraction",
                    stats.matching_size_pairs as f64 / stats.pairs as f64,
                ),
                (
                    "wall_art_stacked_pair_fraction",
                    stats.stacked_pairs as f64 / stats.pairs as f64,
                ),
            ] {
                numeric.push(name, value)?;
            }
        }
    }
    for (i, name) in ["low", "seated", "standing", "elevated"]
        .into_iter()
        .enumerate()
    {
        for (parameter, value) in [
            ("weight", p.weights[i]),
            ("center_m", p.centers_metres[i]),
            ("scale_m", p.scales_metres[i]),
        ] {
            numeric.push(
                &format!("camera_viewpoint_prior_{name}_{parameter}"),
                value as f64,
            )?;
        }
    }
    numeric.push(
        "camera_level_gaze_prior_fraction",
        p.level_gaze_fraction as f64,
    )?;
    numeric.push(
        "camera_level_gaze_prior_pitch_span_degrees",
        p.level_pitch_span_degrees as f64,
    )?;
    Ok(())
}

pub(super) fn record_material(
    material: &MaterialRecipe,
    numeric: &mut NumericCollector,
    categories: &mut Categories,
) -> io::Result<()> {
    if let Some(f) = &material.flooring {
        category(categories, "floor_installation", format!("{:?}", f.layout));
        for (key, value) in [
            ("installation_cells_u", f.cells[0] as f32),
            ("installation_cells_v", f.cells[1] as f32),
            ("piece_tone", f.piece_tone),
            ("piece_roughness", f.piece_roughness),
        ] {
            numeric.push(&format!("material_Floor_{key}"), value as f64)?;
        }
    }
    if let Some(w) = material.coating.as_ref().and_then(|c| c.wall.as_ref()) {
        for (key, value) in [
            ("wash", w.weights[0]),
            ("fiber", w.weights[1]),
            ("weave", w.weights[2]),
            ("print", w.weights[3]),
            ("pigment", w.pigment),
            ("relief_m", w.relief_m),
            ("roughness_variation", w.roughness_variation),
            ("burnish", w.burnish),
        ] {
            numeric.push(
                &format!("material_{:?}_covering_{key}", material.surface),
                value as f64,
            )?;
        }
    }
    if let Some(m) = &material.manufactured {
        for (key, value) in [
            ("cells_u", m.cells[0] as f32),
            ("cells_v", m.cells[1] as f32),
            ("structure", m.structure),
            ("peening", m.peening),
        ] {
            numeric.push(
                &format!("material_{:?}_manufactured_{key}", material.surface),
                value as f64,
            )?;
        }
    }
    if let Some(t) = &material.textile {
        if let Some(d) = &t.dye_program {
            for axis in 0..2 {
                for (key, value) in [
                    ("runs", d.runs[axis] as f32),
                    ("duty", d.duty[axis]),
                    ("phase", d.phase[axis]),
                    ("strength", d.strength[axis]),
                ] {
                    numeric.push(
                        &format!("material_{:?}_yarn_dye_{axis}_{key}", material.surface),
                        value as f64,
                    )?;
                }
            }
        }
    }
    if let Some(c) = &material.cuticle {
        for (key, value) in [
            ("chlorophyll_variation", c.chlorophyll_variation),
            ("wax", c.wax),
            ("bloom", c.bloom),
            ("pore_cells_u", c.pore_cells[0] as f32),
            ("pore_cells_v", c.pore_cells[1] as f32),
            ("pore_contrast", c.pore_contrast),
            ("transmission", c.transmission),
        ] {
            numeric.push(
                &format!("material_{:?}_{key}", material.surface),
                value as f64,
            )?;
        }
    }
    Ok(())
}

fn finish_seed(scene: &IndoorManifest, o: &IndoorObject) -> u64 {
    let surface = match o.kind {
        ObjectKind::WallArt => Surface::Art,
        ObjectKind::Whiteboard => Surface::Whiteboard,
        ObjectKind::Phone => Surface::PhoneScreen,
        ObjectKind::Display => Surface::Television,
        ObjectKind::Monitor | ObjectKind::Laptop => Surface::Screen,
        _ => unreachable!("content finish requested for a non-display object"),
    };
    materials::variants::screen_seed(
        scene.material_seed(),
        materials::variants::object_slot(surface, o.seed, o.finish_seed),
    )
}

fn record_equipment(
    scene: &IndoorManifest,
    o: &IndoorObject,
    categories: &mut Categories,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    if o.kind == ObjectKind::FloorOutlet {
        let p = super::super::objects::floor::parameters(o);
        category(
            categories,
            "floor_outlet_finish",
            format!("{:?}", p.outlet_finish),
        );
        numeric.push("floor_outlet_yaw_rad", o.yaw as f64)?;
        return Ok(());
    }
    if o.kind == ObjectKind::ExitSign {
        let p = equipment::signals::exit::parameters(o);
        category(categories, "exit_legend", format!("{:?}", p.legend));
        category(categories, "exit_mount", format!("{:?}", p.mount));
        category(categories, "exit_double_sided", p.double_sided);
        category(categories, "exit_rail_housing", p.rail_mounted);
        for (key, value) in [
            ("exit_panel_width_m", p.body.x),
            ("exit_panel_height_m", p.body.y),
            ("exit_panel_depth_m", p.body.z),
            (
                "exit_drop_m",
                if p.double_sided {
                    o.size.y - p.body.y
                } else {
                    0.
                },
            ),
            ("exit_emission_scale", p.luminance),
        ] {
            numeric.push(key, value as f64)?;
        }
        return Ok(());
    }
    if o.kind == ObjectKind::CeilingFan {
        let p = super::super::objects::fans::parameters(o);
        category(categories, "fan_style", p.style);
        category(categories, "fan_blade_count", p.blades);
        category(
            categories,
            "fan_blade_material",
            format!("{:?}", p.blade_surface),
        );
        for (key, value) in [
            ("fan_radius_m", p.radius_m),
            ("fan_drop_m", o.size.y),
            ("fan_blade_width_m", p.blade_width_m),
            ("fan_blade_pitch_rad", p.pitch_radians),
            ("fan_blade_sweep_m", p.sweep_m),
        ] {
            numeric.push(key, value as f64)?;
        }
        return Ok(());
    }
    if !matches!(
        o.kind,
        ObjectKind::Projector
            | ObjectKind::ProjectionScreen
            | ObjectKind::PowerStrip
            | ObjectKind::WallShelf
            | ObjectKind::NoticeBoard
            | ObjectKind::WallSconce
            | ObjectKind::CeilingPendant
    ) {
        return Ok(());
    }
    let p = equipment::parameters(o);
    match o.kind {
        ObjectKind::Projector => {
            let p = equipment::projector::parameters(o);
            category(categories, "projector_housing", format!("{:?}", p.housing));
            for (name, value) in [
                ("projector_roundness", p.roundness),
                (
                    "projector_lens_fraction",
                    p.lens_radius / p.body.y.min(p.body.x),
                ),
                ("projector_lens_offset", p.lens_x / p.body.x),
                ("projector_lens_radius_m", p.lens_radius),
                ("projector_body_width_m", p.body.x),
                ("projector_body_height_m", p.body.y),
                ("projector_body_depth_m", p.body.z),
                ("equipment_vent_count", p.vents as f32),
            ] {
                numeric.push(name, value as f64)?;
            }
        }
        ObjectKind::PowerStrip => {
            let strip = equipment::power_strip::parameters(o);
            category(
                categories,
                "powerstrip_housing",
                format!("{:?}", strip.housing),
            );
            category(categories, "powerstrip_socket_rows", strip.rows);
            category(categories, "powerstrip_plate", format!("{:?}", strip.plate));
            category(categories, "powerstrip_end_caps", strip.end_caps);
            numeric.push("powerstrip_socket_yaw_rad", strip.socket_yaw as f64)?;
            numeric.push(
                "powerstrip_height_m",
                (o.size.y * strip.height_fraction) as f64,
            )?;
            if let Some(support) = o
                .support
                .and_then(|id| scene.objects.iter().find(|s| s.id == id))
            {
                let local = support
                    .transform()
                    .compute_affine()
                    .inverse()
                    .transform_point3(o.position);
                numeric.push(
                    "powerstrip_support_x_fraction",
                    (local.x / support.size.x) as f64,
                )?;
                numeric.push(
                    "powerstrip_support_z_fraction",
                    (local.z / support.size.z) as f64,
                )?;
                numeric.push("powerstrip_support_yaw_rad", (o.yaw - support.yaw) as f64)?;
            }
            // These describe the generated pin geometry, rather than asserting
            // certification to a national electrical standard.
            category(
                categories,
                "powerstrip_electrical_region",
                [
                    "round_two_pin",
                    "parallel_flat_two_pin",
                    "wide_flat_three_pin",
                    "flat_three_pin",
                ][(o.variant % 4) as usize],
            );
            numeric.push(
                "equipment_corner_m",
                (o.size.y * strip.height_fraction * strip.bevel_fraction) as f64,
            )?;
            numeric.push("equipment_socket_count", strip.socket_count as f64)?;
        }
        ObjectKind::WallShelf => {
            let shelf = equipment::wall_shelf::parameters(o);
            category(
                categories,
                "wall_shelf_surface",
                format!("{:?}", shelf.board),
            );
            numeric.push("wall_shelf_brackets", shelf.bracket_x_m.len() as f64)?;
            numeric.push("wall_shelf_thickness_m", shelf.thickness_m as f64)?;
            category(categories, "wall_shelf_support_style", shelf.style);
            for &x in &shelf.bracket_x_m {
                numeric.push("wall_shelf_support_x_fraction", (x / o.size.x) as f64)?;
            }
        }
        ObjectKind::ProjectionScreen => {
            let screen = super::super::objects::presentation::projection::parameters(o);
            category(categories, "projection_screen_framed", screen.framed);
            category(
                categories,
                "projection_screen_housing",
                format!("{:?}", screen.housing),
            );
            for (name, value) in [
                ("projection_screen_drop", screen.drop_fraction),
                ("projection_screen_image_width_m", screen.image_width_m()),
                (
                    "projection_screen_image_height_m",
                    screen.surface_hi.y - screen.surface_lo.y,
                ),
                ("projection_screen_offset_x_m", screen.sheet_center_x_m),
            ] {
                numeric.push(name, value as f64)?;
            }
        }
        ObjectKind::NoticeBoard => {
            category(
                categories,
                "noticeboard_frame",
                format!(
                    "{:?}",
                    if p.housing == Surface::Metal {
                        Surface::Metal
                    } else {
                        Surface::WoodEdge
                    }
                ),
            );
            let columns = if o.size.x > o.size.y { 4 } else { 3 };
            numeric.push(
                "noticeboard_note_count",
                p.note_count.min(columns * 4) as f64,
            )?;
        }
        ObjectKind::WallSconce | ObjectKind::CeilingPendant => {
            category(
                categories,
                "equipment_fixture_housing",
                format!("{:?}/{:?}", o.kind, p.housing),
            );
            numeric.push("equipment_diffuser_fraction", p.diffuser_fraction as f64)?;
            if o.kind == ObjectKind::CeilingPendant {
                numeric.push("equipment_shade_fraction", p.shade_fraction as f64)?;
                numeric.push(
                    "equipment_shade_height_m",
                    (o.size.y * p.shade_fraction) as f64,
                )?;
                numeric.push(
                    "equipment_diffuser_radius_m",
                    (o.size.x.min(o.size.z) * 0.46 * p.diffuser_fraction) as f64,
                )?;
            } else {
                let fixture = equipment::sconce::parameters(o);
                numeric.push(
                    "equipment_diffuser_width_m",
                    (fixture.diffuser_radius * 2.) as f64,
                )?;
                category(categories, "sconce_shade_shape", fixture.shade.shape);
                category(categories, "sconce_arm_style", fixture.arm_style);
                category(categories, "sconce_plate_round", fixture.plate_round);
                category(
                    categories,
                    "sconce_trim",
                    format!("{:?}", fixture.shade.trim),
                );
                for (name, value) in [
                    ("sconce_head_width_m", fixture.head_span.x),
                    ("sconce_head_depth_m", fixture.head_span.y),
                    ("sconce_head_height_m", fixture.head_height),
                    ("sconce_head_offset_x_m", fixture.head_bottom.x),
                    ("sconce_plate_width_m", fixture.plate_span.x),
                    ("sconce_plate_height_m", fixture.plate_span.y),
                    ("sconce_arm_radius_m", fixture.arm_radius),
                ] {
                    numeric.push(name, value as f64)?;
                }
            }
            let (position, color, flux, inner, outer) =
                super::super::architecture::object_fixture_photometry(scene, o);
            for (name, value) in [
                ("equipment_fixture_lumens", flux),
                ("equipment_fixture_inner_angle", inner),
                ("equipment_fixture_outer_angle", outer),
                ("equipment_emitter_room_x_m", position.x),
                ("equipment_emitter_room_y_m", position.y),
                ("equipment_emitter_room_z_m", position.z),
                ("equipment_emitter_red", color.x),
                ("equipment_emitter_green", color.y),
                ("equipment_emitter_blue", color.z),
            ] {
                numeric.push(name, value as f64)?;
            }
        }
        _ => unreachable!("equipment kinds were filtered"),
    }
    Ok(())
}

pub(super) fn record_object(
    scene: &IndoorManifest,
    o: &IndoorObject,
    categories: &mut Categories,
    numeric: &mut NumericCollector,
) -> Result<(), Box<dyn std::error::Error>> {
    record_equipment(scene, o, categories, numeric)?;
    service_equipment::record(o, categories, numeric)?;
    service_furniture::record(o, categories, numeric)?;
    kitchen_fittings::record(o, categories, numeric)?;
    if o.kind == ObjectKind::Cabinet {
        let p = super::super::objects::storage::parameters(o);
        category(categories, "cabinet_front", p.front);
        for (name, surface) in [
            ("frame", p.frame),
            ("shelf", p.shelves),
            ("front", p.fronts),
        ] {
            category(
                categories,
                &format!("cabinet_{name}_substrate"),
                format!("{surface:?}"),
            );
        }
        numeric.push("cabinet_bottom_m", p.bottom_m as f64)?;
        numeric.push("cabinet_body_depth_m", p.body_depth_m as f64)?;
        numeric.push("cabinet_panel_m", p.panel_m as f64)?;
        for b in &p.bays {
            numeric.push("cabinet_bay_width_m", (b.right - b.left) as f64)?;
            for v in b.levels.windows(2) {
                numeric.push(
                    "cabinet_clear_bay_height_m",
                    (v[1] - v[0] - p.panel_m) as f64,
                )?;
            }
        }
    }
    if o.kind == ObjectKind::Clock {
        let p = super::super::objects::clocks::parameters(o);
        category(categories, "clock_digital", p.digital);
        category(
            categories,
            "clock_casing_substrate",
            format!("{:?}", p.casing),
        );
        numeric.push("clock_aspect", (o.size.x / o.size.y) as f64)?;
        numeric.push(
            "clock_case_depth_m",
            super::super::objects::clocks::case_depth(o) as f64,
        )?;
        numeric.push("clock_roundness", p.roundness as f64)?;
    }
    if o.kind.is_kitchen() {
        if let Some(fields) = super::super::objects::kitchen::metadata(o).as_object() {
            for (key, value) in fields {
                let name = format!("kitchen_{:?}_{key}", o.kind);
                match value {
                    serde_json::Value::Number(n) => {
                        if let Some(v) = n.as_f64() {
                            numeric.push(&name, v)?;
                        }
                    }
                    serde_json::Value::String(v) => category(categories, &name, v),
                    serde_json::Value::Bool(v) => category(categories, &name, v),
                    _ => (),
                }
            }
        }
    }
    if matches!(
        o.kind,
        ObjectKind::KitchenCounter
            | ObjectKind::KitchenCupboard
            | ObjectKind::Pantry
            | ObjectKind::KitchenSink
    ) {
        let kitchen = super::super::objects::kitchen::parameters(o);
        let p = kitchen.front;
        if kitchen.door_style == 1 {
            category(categories, "cabinet_glazing_requested_rows", p.glazing_rows);
            category(
                categories,
                "cabinet_glazing_requested_columns",
                p.glazing_columns,
            );
        }
        if kitchen.front_profile != 0 || kitchen.door_style == 1 {
            numeric.push("cabinet_stile_fraction", p.frame_fraction.x as f64)?;
            numeric.push("cabinet_rail_fraction", p.frame_fraction.y as f64)?;
        }
        if kitchen.front_profile == 3 && kitchen.door_style != 1 {
            category(categories, "cabinet_reeds_horizontal", p.horizontal);
            numeric.push("cabinet_reed_pitch_m", p.flute_pitch_m as f64)?;
            numeric.push("cabinet_reed_fill", p.flute_fill as f64)?;
        }
    }
    match o.kind {
        ObjectKind::Notepad => {
            let p = super::super::clutter::tabletop::pad::parameters(o);
            category(categories, "pad_binding", p.binding);
            numeric.push("pad_body_height_m", p.height as f64)?;
            numeric.push("pad_ring_pitch_m", p.ring_pitch as f64)?;
            let slot =
                materials::variants::object_slot(Surface::PrintedPaper, o.seed, o.finish_seed);
            let paper = materials::paper::parameters(materials::variants::screen_seed(
                scene.material_seed(),
                slot,
            ));
            category(categories, "paper_layout", paper.layout);
            numeric.push("paper_ruling_spacing_texels", paper.spacing as f64)?;
            numeric.push("paper_writing_fraction", paper.writing as f64)?;
            numeric.push("paper_roughness", paper.roughness as f64)?;
        }
        ObjectKind::Pencil => {
            let p = super::super::clutter::tabletop::writing::parameters(o);
            category(categories, "writing_mechanism", p.mechanism);
            category(categories, "writing_clip", p.clip);
            numeric.push("writing_length_m", p.length as f64)?;
            numeric.push("writing_radius_m", p.radius as f64)?;
        }
        ObjectKind::ExhibitPlinth => {
            let p = super::super::objects::public::casework::plinth::parameters(o);
            category(categories, "plinth_structure", p.structure);
            category(
                categories,
                "plinth_cap_finish",
                format!("{:?}", p.cap_finish),
            );
            for (key, value) in [
                ("plinth_cap_height_m", p.cap_height),
                ("plinth_base_height_m", p.base_height),
                ("plinth_waist_fraction", p.waist_height),
                ("plinth_roundness", p.roundness),
            ] {
                numeric.push(key, value as f64)?;
            }
        }
        ObjectKind::TrashCan => {
            let p = super::super::objects::containers::bin_parameters(o);
            category(categories, "bin_lid", p.lid);
            category(categories, "bin_finish", format!("{:?}", p.finish));
            numeric.push("bin_squareness", p.squareness as f64)?;
            numeric.push("bin_taper", p.taper as f64)?;
            numeric.push("bin_rim_width_fraction", p.rim_width as f64)?;
        }
        ObjectKind::TissueBox => {
            let p = super::super::objects::containers::tissue_parameters(o);
            category(categories, "tissue_box_finish", format!("{:?}", p.finish));
            numeric.push("tissue_box_height_fraction", p.box_height as f64)?;
            numeric.push("tissue_sheet_fold", p.fold as f64)?;
            numeric.push("tissue_sheet_lean", p.lean as f64)?;
        }
        ObjectKind::SodaCan => {
            let p = super::super::clutter::beverages::can::parameters(o);
            category(categories, "can_opened", p.opened);
            category(categories, "can_bare", p.bare);
            for (key, value) in [
                ("can_radius_fraction", p.radius_fraction),
                ("can_neck_fraction", p.neck),
                ("can_shoulder_fraction", p.shoulder),
                ("can_label_coverage", p.label_top - p.label_bottom),
                ("can_lid_yaw_rad", p.lid_yaw),
            ] {
                numeric.push(key, value as f64)?;
            }
        }
        ObjectKind::Sponge => {
            let p = super::super::objects::kitchen::supplies::sponge::parameters(o);
            category(categories, "sponge_scouring", p.scouring);
            category(categories, "sponge_color", p.color);
            numeric.push("sponge_roundness", p.roundness as f64)?;
            numeric.push("sponge_waist", p.waist as f64)?;
        }
        ObjectKind::CuttingBoard => {
            let p = super::super::objects::kitchen::supplies::board_parameters(o);
            category(categories, "cutting_board_handle", p.handle);
            category(categories, "cutting_board_feet", p.feet_fraction > 0.);
            numeric.push("cutting_board_handle_offset", p.handle_offset as f64)?;
        }
        ObjectKind::WallOutlet | ObjectKind::LightSwitch => {
            let p = super::super::objects::wall::parameters(o);
            let name = if o.kind == ObjectKind::WallOutlet {
                "outlet"
            } else {
                "switch"
            };
            for (key, value) in [
                ("case_width_fraction", p.case_fraction.x),
                ("case_height_fraction", p.case_fraction.y),
                ("control_width_fraction", p.control_fraction.x),
                ("control_height_fraction", p.control_fraction.y),
                (
                    "height_above_floor_m",
                    o.position.y - scene.floor_height(o.position.xz()),
                ),
                ("depth_m", o.size.z),
            ] {
                numeric.push(&format!("{name}_{key}"), value as f64)?;
            }
            category(categories, &format!("{name}_construction"), p.style);
            category(categories, &format!("{name}_gangs"), p.gangs);
            category(
                categories,
                &format!("{name}_finish"),
                format!("{:?}", p.finish),
            );
        }
        _ => {}
    }
    if matches!(o.kind, ObjectKind::Monitor | ObjectKind::Display) {
        let p = super::super::objects::computers::display_parameters(o);
        let d = super::super::objects::computers::display_geometry(o);
        numeric.push("display_aspect", (d.width_m / d.height_m) as f64)?;
        numeric.push("display_width_m", d.width_m as f64)?;
        numeric.push("display_height_m", d.height_m as f64)?;
        numeric.push("display_bezel_m", d.bezel_m as f64)?;
        numeric.push("display_rear_width_fraction", p.back_fraction.x as f64)?;
        numeric.push("display_rear_height_fraction", p.back_fraction.y as f64)?;
        if o.kind == ObjectKind::Monitor {
            numeric.push("display_curvature_radius_m", d.curvature_radius_m as f64)?;
            numeric.push("display_tilt_radians", d.tilt_radians as f64)?;
            category(categories, "display_stand", p.stand);
            category(categories, "display_riser", p.mount.riser_style);
            category(
                categories,
                "display_support_finish",
                format!("{:?}", p.mount.finish),
            );
            numeric.push("display_base_height_m", p.mount.base_height as f64)?;
            numeric.push(
                "display_riser_radius_requested_m",
                p.mount.riser_radius as f64,
            )?;
        }
    }
    if o.kind == ObjectKind::PaperTowel {
        let p = super::super::objects::kitchen::supplies::paper_towel::parameters(o);
        category(categories, "roll_horizontal", p.horizontal);
        category(categories, "roll_base_style", p.base_style);
        category(categories, "roll_holder_finish", format!("{:?}", p.finish));
        for (key, value) in [
            ("roll_radius_m", p.radius),
            ("roll_fullness_radius_ratio", p.radius / p.full_radius),
            ("roll_core_radius_m", p.core_radius),
            ("roll_length_m", p.length),
            ("roll_tail_drop_m", p.tail_drop),
            ("roll_base_height_m", p.base_height),
        ] {
            numeric.push(key, value as f64)?;
        }
    }
    if matches!(o.kind, ObjectKind::Desk | ObjectKind::CoffeeTable) {
        let p = super::super::layout::surfaces::workspace_parameters(o);
        for (key, value) in [
            ("workspace_screen_affinity", p.screen_affinity),
            ("workspace_monitor_affinity", p.monitor_affinity),
            ("workspace_companion_affinity", p.companion_affinity),
            ("workspace_docking_affinity", p.docking_affinity),
            ("workspace_paper_affinity", p.paper_affinity),
            ("workspace_display_span_fraction", p.display_span_fraction),
            ("workspace_rear_fraction", p.rear_fraction),
            ("workspace_asymmetry_fraction", p.asymmetry_fraction),
            ("workspace_accessory_affinity", p.accessory_affinity),
            ("workspace_monitor_gap_m", p.monitor_gap_m),
            (
                "workspace_monitor_depth_scatter_m",
                p.monitor_depth_scatter_m,
            ),
            ("workspace_monitor_size_scatter", p.monitor_size_scatter),
            ("workspace_heading_radians", p.workstation_heading),
            ("workspace_laptop_depth_fraction", p.laptop_depth_fraction),
            ("workspace_docking_side", p.docking_side),
        ] {
            numeric.push(key, value as f64)?;
        }
        let monitors = scene
            .objects
            .iter()
            .filter(|x| x.support == Some(o.id) && x.kind == ObjectKind::Monitor)
            .count();
        let laptops = scene
            .objects
            .iter()
            .filter(|x| x.support == Some(o.id) && x.kind == ObjectKind::Laptop)
            .count();
        let arrangement = if monitors >= 2 {
            "multiple_monitors"
        } else if monitors == 1 && laptops > 0 {
            "docked_laptop"
        } else if monitors == 1 {
            "single_monitor"
        } else if laptops > 0 {
            "laptop"
        } else {
            "screen_free"
        };
        category(categories, "accepted_workspace_arrangement", arrangement);
        numeric.push("workspace_monitor_count", monitors as f64)?;
        numeric.push("workspace_laptop_count", laptops as f64)?;
    }
    if o.kind == ObjectKind::Cable {
        let path = o.cable_path.as_ref().ok_or("cable missing route")?;
        numeric.push(
            "cable_length_m",
            path.windows(2).map(|w| w[0].distance(w[1]) as f64).sum(),
        )?;
        numeric.push("cable_radius_m", equipment::wires::radius(o) as f64)?;
        numeric.push("cable_route_points", path.len() as f64)?;
    }
    if o.kind == ObjectKind::WallArt {
        let p = materials::art::parameters(finish_seed(scene, o));
        let frame = materials::art::frame_parameters(o);
        category(categories, "art_subject", p.subject);
        category(categories, "art_frame", frame.style);
        for (key, value) in [
            ("art_width_m", o.size.x),
            ("art_height_m", o.size.y),
            ("art_aspect", o.size.x / o.size.y),
            ("art_lower_edge_m", o.position.y),
            ("art_center_height_m", o.position.y + o.size.y * 0.5),
            ("art_horizon", p.horizon),
            ("art_relief", p.relief),
            ("art_viewpoint", p.viewpoint),
            ("art_hue", p.hue),
            ("art_saturation", p.saturation),
            ("art_density", p.density),
            ("art_brush", p.brush),
            ("art_border_m", frame.border_width),
            ("art_mat_m", frame.mat_width),
            ("art_mat_aspect", frame.mat_scale.x / frame.mat_scale.y),
            ("art_picture_bias_x", frame.picture_bias.x),
            ("art_picture_bias_y", frame.picture_bias.y),
            ("art_shadow_gap_m", frame.shadow_gap),
        ] {
            numeric.push(key, value as f64)?;
        }
    }
    if o.kind == ObjectKind::Phone {
        let p = super::super::clutter::tabletop::phone::parameters(o);
        category(categories, "phone_case", p.case);
        category(categories, "phone_shell", format!("{:?}", p.shell));
        category(categories, "phone_face_up", p.face_up);
        for (name, value) in [
            ("phone_body_width_m", p.span.x),
            ("phone_body_length_m", p.span.y),
            ("phone_body_height_m", p.body_height),
            ("phone_roundness", p.roundness),
            ("phone_button_length_m", p.button_length),
            ("phone_button_z_m", p.button_z),
        ] {
            numeric.push(name, value as f64)?;
        }
        if p.face_up {
            numeric.push("phone_screen_width_m", p.screen_span.x as f64)?;
            numeric.push("phone_screen_length_m", p.screen_span.y as f64)?;
        } else {
            category(categories, "phone_camera_layout", p.camera_layout);
            category(categories, "phone_lens_count", p.lens_count);
            category(categories, "phone_back", format!("{:?}", p.back));
            numeric.push("phone_lens_radius_m", p.lens_radius as f64)?;
            numeric.push("phone_camera_island_width_m", p.island_span.x as f64)?;
        }
    }
    if o.kind == ObjectKind::Whiteboard {
        let built = super::super::objects::presentation::parameters(o);
        numeric.push("whiteboard_marker_count", built.marker_count as f64)?;
        numeric.push("whiteboard_magnet_count", built.magnets.len() as f64)?;
        numeric.push("whiteboard_depth_m", built.thickness_m as f64)?;
        for item in &built.contents {
            numeric.push("whiteboard_tray_item_yaw_rad", item.yaw as f64)?;
        }
        let p = materials::boards::parameters(finish_seed(scene, o));
        category(categories, "whiteboard_content", p.layout);
        for (key, value) in [
            ("whiteboard_occupancy", p.occupancy),
            ("whiteboard_erasure", p.erasure),
            ("whiteboard_color_count", p.color_count as f32),
            ("whiteboard_margin", p.margin),
            ("whiteboard_slope", p.slope),
            ("whiteboard_curvature", p.curvature),
            ("whiteboard_roughness", p.roughness),
        ] {
            numeric.push(key, value as f64)?;
        }
    }
    if matches!(
        o.kind,
        ObjectKind::Laptop | ObjectKind::Display | ObjectKind::Monitor | ObjectKind::Phone
    ) && (o.kind != ObjectKind::Phone
        || super::super::clutter::tabletop::parameters(o).phone_face_up)
    {
        let kind = match o.kind {
            ObjectKind::Phone => materials::screens::DisplayKind::Phone,
            ObjectKind::Display => materials::screens::DisplayKind::Television,
            _ => materials::screens::DisplayKind::Desktop,
        };
        let p = materials::screens::parameters_for(finish_seed(scene, o), kind);
        if o.kind == ObjectKind::Display {
            category(categories, "tv_content", p.layout);
        }
        numeric.push("screen_time_minutes", p.minutes as f64)?;
        numeric.push(
            "screen_luminance",
            if p.dormant { 0. } else { p.luminance as f64 },
        )?;
        for (key, value) in [
            ("screen_white_kelvin", p.white_kelvin),
            ("screen_white_tint", p.white_tint),
            ("screen_sidebar_fraction", p.sidebar),
            ("screen_content_split", p.content_split),
            ("screen_content_density", p.density),
            ("screen_panels", p.panels as f32),
            ("screen_accent_hue", p.accent_hue),
            ("screen_chart_slope", p.chart_slope),
            ("screen_chart_curvature", p.chart_curvature),
            ("screen_framing", p.framing),
        ] {
            numeric.push(key, value as f64)?;
        }
        category(categories, "screen_dormant", p.dormant);
        category(categories, "screen_content", p.layout);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn presentation_metrics_use_constructed_supports_accessories_and_sheet() {
        use super::super::super::objects::{audit, presentation};
        let directory = scratch();
        let scene = IndoorManifest::generate_with_humans(
            21,
            super::super::IndoorLayout::Mixed,
            0.65,
            0,
            0.,
        )
        .unwrap();
        let mut categories = Categories::new();
        let mut numeric = NumericCollector::new(&directory).unwrap();
        let board = audit::fixture(ObjectKind::Whiteboard, 8);
        let shelf = audit::fixture(ObjectKind::WallShelf, 9);
        let screen = audit::fixture(ObjectKind::ProjectionScreen, 10);
        for o in [&board, &shelf, &screen] {
            record_object(&scene, o, &mut categories, &mut numeric).unwrap();
        }
        let values = numeric.finish().unwrap();
        let b = presentation::parameters(&board);
        let s = equipment::wall_shelf::parameters(&shelf);
        let p = presentation::projection::parameters(&screen);
        assert_eq!(
            values["whiteboard_marker_count"].mean,
            b.marker_count as f64
        );
        assert_eq!(
            values["whiteboard_magnet_count"].mean,
            b.magnets.len() as f64
        );
        assert_eq!(
            values["wall_shelf_brackets"].mean,
            s.bracket_x_m.len() as f64
        );
        assert_eq!(values["wall_shelf_thickness_m"].mean, s.thickness_m as f64);
        assert_eq!(
            values["projection_screen_image_width_m"].mean,
            p.image_width_m() as f64
        );
        assert_eq!(
            values["projection_screen_image_height_m"].mean,
            (p.surface_hi.y - p.surface_lo.y) as f64
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn reported_content_seed_matches_the_constructed_material_slot() {
        let scene =
            IndoorManifest::generate_with_humans(0, super::super::IndoorLayout::Mixed, 0.65, 0, 0.)
                .unwrap();
        for (kind, surface) in [
            (ObjectKind::WallArt, Surface::Art),
            (ObjectKind::Whiteboard, Surface::Whiteboard),
            (ObjectKind::Phone, Surface::PhoneScreen),
            (ObjectKind::Display, Surface::Television),
            (ObjectKind::Monitor, Surface::Screen),
            (ObjectKind::Laptop, Surface::Screen),
        ] {
            let mut checked = 0;
            for seed in 0..16 {
                let mut o = super::super::super::objects::audit::fixture(kind, seed);
                o.finish_seed = Some(17);
                let a = super::super::super::objects::build_object(&o);
                for ((s, label), _) in a.parts.iter().filter(|((s, _), _)| *s == surface) {
                    checked += 1;
                    let slot: usize = label
                        .split('#')
                        .find_map(|tag| tag.strip_prefix("finish"))
                        .unwrap()
                        .parse()
                        .unwrap();
                    assert_eq!(
                        finish_seed(&scene, &o),
                        materials::variants::screen_seed(scene.material_seed(), slot),
                        "{kind:?}/{seed}/{s:?}"
                    );
                }
            }
            assert!(
                checked >= 8,
                "missing constructed content faces for {kind:?}: {checked}"
            );
        }
    }
    fn object(kind: ObjectKind) -> IndoorObject {
        IndoorObject {
            finish_seed: None,
            id: 0,
            kind,
            position: Vec3::new(1., 2., 3.),
            size: Vec3::new(0.18, 0.03, 0.05),
            yaw: 0.3,
            variant: 2,
            seed: 83,
            solid: false,
            support: None,
            neighbor: false,
            interaction_target: None,
            cable_path: None,
            cable_target: None,
        }
    }
    use bevy::prelude::*;
    fn scratch() -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "zeroverse-content-metrics-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&path).unwrap();
        path
    }
    #[test]
    fn metrics_record_built_equipment_fields_and_actual_clamped_counts() {
        let directory = scratch();
        let mut scene =
            IndoorManifest::generate_with_humans(2, super::super::IndoorLayout::Mixed, 0.65, 0, 0.)
                .unwrap();
        scene
            .apply_appearance(super::super::super::appearance::AppearanceSettings {
                illumination_scale: 0.25,
                ..Default::default()
            })
            .unwrap();
        for kind in [
            ObjectKind::Projector,
            ObjectKind::PowerStrip,
            ObjectKind::WallShelf,
            ObjectKind::NoticeBoard,
            ObjectKind::WallSconce,
            ObjectKind::CeilingPendant,
        ] {
            let o = object(kind);
            let mut numeric = NumericCollector::new(&directory).unwrap();
            let mut categories = Categories::new();
            record_equipment(&scene, &o, &mut categories, &mut numeric).unwrap();
            let values = numeric.finish().unwrap();
            assert_eq!(
                values.contains_key("projector_lens_fraction"),
                kind == ObjectKind::Projector
            );
            assert_eq!(
                values.contains_key("equipment_socket_count"),
                kind == ObjectKind::PowerStrip
            );
            assert_eq!(
                values.contains_key("equipment_shade_fraction"),
                kind == ObjectKind::CeilingPendant
            );
            assert_eq!(
                values.contains_key("equipment_diffuser_fraction"),
                matches!(kind, ObjectKind::WallSconce | ObjectKind::CeilingPendant)
            );
            if kind == ObjectKind::Projector {
                let a = super::super::super::objects::build_object(&o);
                let mut lo = f32::INFINITY;
                let mut hi = f32::NEG_INFINITY;
                for ((surface, _), g) in &a.parts {
                    if *surface == Surface::ContainerGlass {
                        for v in &g.positions {
                            lo = lo.min(v[0]);
                            hi = hi.max(v[0]);
                        }
                    }
                }
                // Glass uses 68% of the surrounding optical rim radius.
                assert!(
                    (values["projector_lens_radius_m"].mean as f32 - (hi - lo) / 1.36).abs() < 1e-5
                );
            }
            if kind == ObjectKind::PowerStrip {
                let p = equipment::parameters(&o);
                assert_eq!(values["equipment_socket_count"].mean, p.socket_count as f64);
            }
            if matches!(kind, ObjectKind::WallSconce | ObjectKind::CeilingPendant) {
                let (position, _, lumens, _, _) =
                    super::super::super::architecture::object_fixture_photometry(&scene, &o);
                assert_eq!(values["equipment_fixture_lumens"].mean, lumens as f64);
                assert_eq!(values["equipment_emitter_room_x_m"].mean, position.x as f64);
            }
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn content_metrics_replay_renderer_material_seed_and_camera_priors() {
        let directory = scratch();
        let mut scene = IndoorManifest::generate_with_humans(
            26,
            super::super::IndoorLayout::Mixed,
            0.65,
            0,
            0.,
        )
        .unwrap();
        scene
            .apply_appearance(super::super::super::appearance::AppearanceSettings {
                material_seed: Some(991),
                ..Default::default()
            })
            .unwrap();
        let mut categories = Categories::new();
        let mut numeric = NumericCollector::new(&directory).unwrap();
        record_scene(&scene, &mut numeric, &mut Categories::new()).unwrap();
        for kind in [
            ObjectKind::WallArt,
            ObjectKind::Whiteboard,
            ObjectKind::Display,
        ] {
            record_object(&scene, &object(kind), &mut categories, &mut numeric).unwrap();
        }
        let actual_seed = |surface| {
            materials::variants::screen_seed(
                991,
                materials::variants::object_slot(surface, 83, None),
            )
        };
        let art = materials::art::parameters(actual_seed(Surface::Art));
        let board = materials::boards::parameters(actual_seed(Surface::Whiteboard));
        let screen = materials::screens::parameters_for(
            actual_seed(Surface::Television),
            materials::screens::DisplayKind::Television,
        );
        let values = numeric.finish().unwrap();
        let wall_art = object(ObjectKind::WallArt);
        assert_eq!(
            values["art_center_height_m"].mean,
            (wall_art.position.y + wall_art.size.y * 0.5) as f64
        );
        let walls = super::super::super::layout::wall_composition::mounted_art_walls(&scene);
        let multi_walls = walls.iter().filter(|wall| wall.len() > 1).count();
        assert_eq!(
            values.get("wall_art_count_per_wall").map_or(0, |d| d.count),
            walls.len()
        );
        for key in [
            "wall_art_center_height_span_m",
            "wall_art_area_ratio",
            "wall_art_aligned_pair_fraction",
            "wall_art_matching_size_pair_fraction",
            "wall_art_stacked_pair_fraction",
        ] {
            assert_eq!(values.get(key).map_or(0, |d| d.count), multi_walls);
        }
        assert_eq!(values["art_horizon"].mean, art.horizon as f64);
        assert_eq!(values["whiteboard_occupancy"].mean, board.occupancy as f64);
        assert_eq!(values["screen_time_minutes"].mean, screen.minutes as f64);
        assert_eq!(categories["art_subject"][&art.subject.to_string()], 1);
        assert_eq!(
            categories["whiteboard_content"][&board.layout.to_string()],
            1
        );
        assert_eq!(categories["tv_content"][&screen.layout.to_string()], 1);
        assert_eq!(
            values["camera_viewpoint_prior_elevated_center_m"].mean,
            ViewpointProgram::sample(scene.seed).centers_metres[3] as f64
        );
        assert_eq!(
            values
                .keys()
                .filter(|key| key.starts_with("camera_viewpoint_prior_"))
                .count(),
            12
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
