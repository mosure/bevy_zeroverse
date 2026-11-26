//! Built furnishing/content diagnostics, with the same finish seeds and physical
//! emitters used by the renderer. Equipment fields exclude dormant parameters.
use super::super::{
    cameras::viewpoint::ViewpointProgram,
    layout::{IndoorManifest, IndoorObject, ObjectKind},
    materials::{self, program::MaterialRecipe, Surface},
    metrics_sort::NumericCollector,
    objects::{equipment, seating},
};
use bevy::math::Vec3Swizzles;
use std::{collections::BTreeMap, io};

mod drinkware;
mod kitchen_fittings;
mod rug;
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

pub(super) fn record_swivel(
    stock: &super::super::objects::chairs::SwivelPedestal,
    numeric: &mut NumericCollector,
    categories: &mut Categories,
) -> io::Result<()> {
    use super::super::objects::chairs::Grip;
    numeric.push("chair_swivel_hub_height_m", stock.hub_height_m.into())?;
    category(categories, "chair_swivel_table_order", stock.table_order);
    let mechanism = if let Some(lift) = stock.gas_lift {
        category(
            categories,
            "chair_swivel_control_side",
            if lift.lever.left { "left" } else { "right" },
        );
        numeric.push("chair_lift_exposed_fraction", lift.exposed_fraction.into())?;
        match lift.lever.grip {
            Grip::Paddle => "paddle",
            Grip::Sleeve => "sleeve",
        }
    } else {
        "fixed_height"
    };
    category(categories, "chair_swivel_mechanism", mechanism);
    if let Some(blade) = stock.cast_spoke {
        numeric.push("chair_cast_spoke_depth_m", blade.depth_m.into())?;
    }
    Ok(())
}

pub(super) fn record_scene(
    scene: &IndoorManifest,
    numeric: &mut NumericCollector,
    categories: &mut Categories,
) -> io::Result<()> {
    if let Some(frame) = materials::flooring::laying::mapping(scene) {
        numeric.push(
            "floor_laying_room_heading_degrees",
            frame
                .matrix2
                .x_axis
                .to_angle()
                .rem_euclid(std::f32::consts::PI)
                .to_degrees() as f64,
        )?;
    }
    let installation = super::super::layout::electrical_parameters(scene);
    category(categories, "electrical_standard", installation.standard);
    if let Some(door) = &scene.entry_door {
        let half = (door.transform().rotation * bevy::prelude::Vec3::X).xz() * (0.5 * door.width);
        let jambs = [door.center.xz() - half, door.center.xz() + half];
        let nearest = scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::LightSwitch && !o.neighbor)
            .flat_map(|o| jambs.map(|j| j.distance(o.position.xz())))
            .min_by(f32::total_cmp);
        category(
            categories,
            "entry_light_switch",
            match nearest {
                Some(d) if d <= 1.5 => "within_1_5m_of_jamb",
                Some(_) => "elsewhere",
                None => "absent",
            },
        );
        if let Some(d) = nearest {
            numeric.push("nearest_light_switch_to_entry_jamb_m", f64::from(d))?;
        }
    }
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
        .filter(|o| super::super::photometry::is_lamp_at(o.kind, scene.generator_version))
    {
        let source = super::super::photometry::lamp(scene, object);
        if object.kind == ObjectKind::DeskLamp {
            let beam = super::super::objects::accessories::lamps::beam(object);
            numeric.push(
                "desk_lamp_tilt_degrees",
                beam.tilt_from_down_degrees() as f64,
            )?;
            numeric.push(
                "desk_lamp_workplane_offset_m",
                ((beam.position.y - object.position.y) * beam.direction.xz().length()
                    / -beam.direction.y) as f64,
            )?;
        }
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
    for mount in &groups.mounts {
        let (Some(object), Some(host)) = (
            scene.objects.get(mount.object),
            scene.objects.get(mount.host),
        ) else {
            continue;
        };
        if object.kind == ObjectKind::CeilingPendant {
            numeric.push(
                "exhibit_pendant_offset_m",
                object.position.xz().distance(host.position.xz()) as f64,
            )?;
        } else if object.kind == ObjectKind::ExtractionHood {
            let plate = super::super::objects::kitchen::appliances::cooking_surface(
                host,
                &super::super::objects::kitchen::parameters(host),
            )
            .0;
            numeric.push(
                "hood_to_cooktop_clearance_m",
                (object.position.y - plate.y) as f64,
            )?;
        }
    }
    numeric.push(
        "feature_wall_edges",
        materials::wall::feature_edges(scene).len() as f64,
    )?;
    numeric.push("functional_programs", groups.programs.len() as f64)?;
    for p in &groups.programs {
        service_furniture::record_dining_tables(scene, p, numeric)?;
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
        kitchen_fittings::record_sink_mats(scene, group, numeric)?;
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
        use super::super::layout::assemblies::AssemblyRole;
        if scene.objects[group.surface].kind == ObjectKind::Desk
            && group.role != AssemblyRole::Meeting
        {
            category(
                categories,
                "desk_operator_positions",
                match group.role {
                    AssemblyRole::StandingWorkstation => "standing",
                    _ if group.seats.len() == 2 => "opposing_pair",
                    _ => "seated",
                },
            );
        }
        let region = if scene.in_primary_room(p, 0.) {
            "primary"
        } else {
            "main_other"
        };
        for (name, n) in [
            ("requested_capacity", group.requested_capacity),
            (
                "accepted_capacity",
                group.seats.len() + usize::from(group.role == AssemblyRole::StandingWorkstation),
            ),
            ("attempts", group.attempts),
        ] {
            numeric.push(
                &format!("assembly_{region}_{:?}_{name}", group.role),
                n as f64,
            )?;
        }
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
    if let Some(f) = material.wood.as_ref().and_then(|w| w.figure) {
        for (key, value) in [
            ("ring_pitch_mm", f.ring_pitch_m(material) * 1000.),
            ("crown_deflection", f.crown.abs()),
            ("vessels", f.vessel_strength),
            ("ring_porosity", f.ring_porosity),
            ("ray_fleck", f.ray_strength),
            ("knot", f.knot),
        ] {
            numeric.push(
                &format!("material_{:?}_timber_{key}", material.surface),
                value as f64,
            )?;
        }
    }
    if let Some(f) = &material.flooring {
        category(categories, "floor_installation", format!("{:?}", f.layout));
        if material.carpet.is_some() {
            let installation = if let Some(tiles) = f.carpet_tiles {
                numeric.push(
                    "material_Floor_carpet_registration_fraction",
                    tiles.registration as f64,
                )?;
                format!("{:?}", tiles.direction)
            } else if f.tile_definition == 0. {
                "Broadloom".into()
            } else {
                "ContinuousTilePattern".into()
            };
            category(categories, "carpet_installation", installation);
        }
        if let Some(laying) = f.laying {
            numeric.push(
                "material_Floor_turn_from_wall_degrees",
                laying.turn_radians.to_degrees() as f64,
            )?;
            numeric.push("material_Floor_phase_u", laying.phase.x as f64)?;
            numeric.push("material_Floor_phase_v", laying.phase.y as f64)?;
        }
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
    materials::variants::content_seed(
        scene.material_seed(),
        surface,
        materials::variants::object_slot(surface, o.seed, o.finish_seed),
    )
}

fn record_equipment(
    scene: &IndoorManifest,
    o: &IndoorObject,
    categories: &mut Categories,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    if o.kind == ObjectKind::Speaker {
        let p = super::super::objects::accessories::speaker::parameters(o);
        category(categories, "speaker_mount", format!("{:?}", p.mount.kind));
        category(categories, "speaker_face", p.face_name());
        numeric.push(
            "speaker_downward_pitch_deg",
            f64::from(p.mount.pitch_radians.to_degrees()),
        )?;
        return Ok(());
    }
    if o.kind == ObjectKind::Rug {
        return rug::record(scene, o, categories, numeric);
    }
    if o.kind == ObjectKind::FloorOutlet {
        use super::super::objects::floor;
        let finish = if let Some(p) = floor::service::parameters(o) {
            category(
                categories,
                "floor_outlet_housing",
                format!("{:?}", p.housing),
            );
            category(
                categories,
                "floor_outlet_hatch_finish",
                format!("{:?}", p.hatch_finish),
            );
            category(
                categories,
                "floor_outlet_state",
                if p.open { "open" } else { "closed" },
            );
            numeric.push("floor_outlet_hatch_width_m", f64::from(p.hatch_width_m))?;
            numeric.push("floor_outlet_hatch_depth_m", f64::from(p.hatch_depth_m))?;
            if p.open {
                category(
                    categories,
                    "floor_outlet_exposed_face",
                    format!("{:?}", p.face),
                );
                numeric.push("floor_outlet_exposed_socket_count", p.socket_count as f64)?;
                numeric.push(
                    "floor_outlet_lid_angle_deg",
                    f64::from(p.open_angle_rad.to_degrees()),
                )?;
            }
            p.outlet_finish
        } else {
            floor::parameters(o).outlet_finish
        };
        category(categories, "floor_outlet_finish", format!("{finish:?}"));
        numeric.push("floor_outlet_yaw_rad", o.yaw as f64)?;
        return Ok(());
    }
    if o.kind == ObjectKind::FireDetector {
        let p = equipment::signals::fire::detector::parameters(o);
        category(categories, "detector_face", format!("{:?}", p.face));
        category(categories, "detector_inlet_count", p.inlets);
        for (name, value) in [
            ("detector_diameter_m", p.radius_m * 2.),
            ("detector_opening_fraction", p.opening_fraction),
            (
                "detector_chamber_setback_m",
                p.lower_lip_radius_m - p.chamber_radius_m,
            ),
        ] {
            numeric.push(name, f64::from(value))?;
        }
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
        if let Some(blade) = &p.product {
            category(
                categories,
                "fan_blade_family",
                format!("{:?}", blade.family),
            );
            for (key, value) in [
                ("fan_root_chord_ratio", blade.root_chord_ratio),
                ("fan_tip_chord_ratio", blade.tip_chord_ratio),
                ("fan_tip_pitch_rad", blade.tip_pitch_radians),
                ("fan_camber_m", blade.camber_m),
                ("fan_tip_rise_m", blade.tip_rise_m),
            ] {
                numeric.push(key, value as f64)?;
            }
        }
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
            if let Some(deck) = strip.deck {
                category(
                    categories,
                    "powerstrip_electrical_region",
                    format!("{:?}", deck.face),
                );
                category(
                    categories,
                    "powerstrip_contact_insert",
                    format!("{:?}", deck.insert),
                );
                if strip.socket_count / strip.rows > 1 {
                    numeric.push("powerstrip_column_pitch_m", deck.column_pitch_m as f64)?;
                }
                if strip.rows == 2 {
                    numeric.push("powerstrip_row_pitch_m", deck.row_pitch_m as f64)?;
                }
                numeric.push("powerstrip_contact_scale", deck.contact_scale as f64)?;
                numeric.push("powerstrip_well_depth_m", deck.well_depth_m as f64)?;
                numeric.push("powerstrip_insert_width_m", deck.insert_width_m as f64)?;
            } else {
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
            }
            numeric.push(
                "equipment_corner_m",
                (o.size.y * strip.height_fraction * strip.bevel_fraction) as f64,
            )?;
            numeric.push("equipment_socket_count", strip.socket_count as f64)?;
        }
        ObjectKind::WallShelf => {
            let shelf = equipment::wall_shelf::parameters_at(o, scene.generator_version);
            if let Some(deck) = &shelf.deck {
                numeric.push(
                    "wall_shelf_top_above_floor_m",
                    f64::from(o.position.y + o.size.y - scene.floor_height(o.position.xz())),
                )?;
                category(categories, "wall_shelf_stock", format!("{:?}", deck.stock));
                for end in deck.ends {
                    category(categories, "wall_shelf_end_cut", format!("{:?}", end.cut));
                    numeric.push("wall_shelf_end_setback_m", f64::from(end.setback_m))?;
                }
                if deck.stock == equipment::wall_shelf::ShelfStock::FoldedSteel {
                    numeric.push("wall_shelf_sheet_gauge_m", f64::from(deck.gauge_m))?;
                }
            }
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
            if let Some(i) = &screen.installation {
                category(
                    categories,
                    "projection_screen_format",
                    format!("{:?}", i.format),
                );
                category(
                    categories,
                    "projection_screen_mechanism",
                    format!("{:?}", i.mechanism),
                );
                category(
                    categories,
                    "projection_screen_deployment",
                    format!("{:?}", i.deployment),
                );
                if screen.image_width_m() > 0. {
                    category(
                        categories,
                        "projection_screen_coating",
                        format!("{:?}", i.coating),
                    );
                }
                numeric.push(
                    "projection_screen_full_image_height_m",
                    f64::from(i.full_image_height_m),
                )?;
            }
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
            let p = equipment::parameters_at(o, scene.generator_version);
            let frame = equipment::board_section::parameters(o).frame_surface;
            let facing = equipment::board_section::Facing::for_board(o, scene.generator_version);
            category(categories, "noticeboard_frame", format!("{frame:?}"));
            category(categories, "noticeboard_facing", format!("{facing:?}"));
            let notes = equipment::pinups::placements_at(o, &p, scene.generator_version);
            numeric.push("noticeboard_note_count", notes.len() as f64)?;
            let inset = o.size.x.min(o.size.y) * p.board_frame_fraction;
            let field_area = (o.size.x - 2. * inset) * (o.size.y - 2. * inset);
            let posted_area: f32 = notes.iter().map(|n| n.size.element_product()).sum();
            numeric.push(
                "noticeboard_posted_area_fraction",
                f64::from(posted_area / field_area),
            )?;
        }
        ObjectKind::WallSconce | ObjectKind::CeilingPendant => {
            let housing = if o.kind == ObjectKind::CeilingPendant {
                let fixture = equipment::pendant::parameters(o);
                category(categories, "pendant_optic", fixture.lens.optic.name());
                category(
                    categories,
                    "pendant_suspension",
                    format!("{:?}", fixture.suspension),
                );
                for (name, value) in [
                    ("pendant_shade_diameter_m", 2. * fixture.shade_radius_m),
                    ("pendant_installed_drop_m", o.size.y),
                    (
                        "equipment_shade_fraction",
                        fixture.shade_height_m / o.size.y,
                    ),
                    ("equipment_shade_height_m", fixture.shade_height_m),
                    ("pendant_emitting_area_m2", fixture.lens.emitting_area_m2()),
                    ("pendant_luminous_radius_m", fixture.lens.radius_m),
                    ("pendant_canopy_diameter_m", 2. * fixture.canopy.radius_m),
                ] {
                    numeric.push(name, value as f64)?;
                }
                if let equipment::pendant::Optic::OpenBulb(bulb) = fixture.lens.optic {
                    category(categories, "pendant_bulb_envelope", format!("{bulb:?}"));
                }
                if let Some(fraction) = fixture.diffuser_fraction() {
                    numeric.push("equipment_diffuser_fraction", fraction as f64)?;
                    numeric.push("equipment_diffuser_radius_m", fixture.lens.radius_m as f64)?;
                }
                fixture.shade.finish
            } else {
                let fixture = equipment::sconce::parameters(o);
                let width = fixture.diffuser_radius * 2.;
                numeric.push("equipment_diffuser_width_m", width as f64)?;
                numeric.push(
                    "equipment_diffuser_fraction",
                    (width / fixture.head_span.x) as f64,
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
                fixture.shade.finish
            };
            category(
                categories,
                "equipment_fixture_housing",
                format!("{:?}/{housing:?}", o.kind),
            );
            let (beam, power) = super::super::architecture::object_fixture_photometry(scene, o);
            let position = beam.position;
            let color = power.color;
            for (name, value) in [
                ("equipment_fixture_lumens", power.lumens),
                ("equipment_fixture_inner_angle", beam.inner),
                ("equipment_fixture_outer_angle", beam.outer),
                ("equipment_beam_vertical_component", beam.direction.y),
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
    use super::super::objects::desktop::tower;
    if o.kind == ObjectKind::Bookcase {
        let p = super::super::objects::storage::bookcase::parameters_at(o, scene.generator_version);
        let (rear, front) = p.deck_limits(o.size.z);
        numeric.push("bookshelf_fill", f64::from(p.fill))?;
        numeric.push("bookshelf_columns", p.bays.len() as f64)?;
        // Each bay has the same number of clear compartments at independently sampled heights.
        numeric.push("bookshelf_levels", (p.bays[0].levels.len() - 1) as f64)?;
        numeric.push("bookshelf_usable_depth_m", f64::from(front - rear))?;
        if let Some(rack) = &p.rack {
            category(
                categories,
                "bookshelf_rear_bracing",
                format!("{:?}", rack.bracing),
            );
            numeric.push("bookshelf_post_width_m", f64::from(rack.post_width_m))?;
            numeric.push("bookshelf_deck_bearing_m", f64::from(rack.bearing_m))?;
        }
    }
    if o.kind == ObjectKind::Sofa && scene.generator_version >= seating::pillows::INTRODUCED {
        let p = seating::parameters(o);
        let throws = seating::pillows::metadata(o, &p, &seating::module_widths_for(o, &p));
        numeric.push("sofa_throw_count", throws.len() as f64)?;
        for item in throws {
            category(
                categories,
                "sofa_throw_support",
                if item.propped {
                    "seat_and_back"
                } else {
                    "seat"
                },
            );
            category(categories, "sofa_throw_edge", item.edge_name());
            numeric.push("sofa_throw_width_m", f64::from(item.width_m()))?;
            if item.propped {
                if let Some(d) = item.back_gap_m {
                    numeric.push("propped_sofa_throw_back_gap_m", f64::from(d))?;
                }
            }
        }
    }
    use super::super::objects::public::elevator;
    if o.kind == ObjectKind::PcTower
        && scene.generator_version >= tower::INTRODUCED
        && tower::installed(o)
    {
        let f = tower::Fascia::sample(o, &tower::parameters(o));
        category(categories, "pc_control_bank", format!("{:?}", f.bank));
        category(categories, "pc_intake", format!("{:?}", f.intake));
        category(
            categories,
            "pc_installation",
            if o.support.is_some() {
                "desktop"
            } else {
                "floor"
            },
        );
        numeric.push("pc_front_controls", f.controls.len() as f64)?;
        numeric.push("pc_front_intake_recess_m", f64::from(f.intake_recess_m))?;
        if let Some(desk) = o
            .interaction_target
            .and_then(|i| scene.objects.get(i))
            .and_then(|s| s.support)
            .and_then(|i| scene.objects.get(i))
        {
            let bearing = o.yaw - desk.yaw;
            numeric.push(
                "pc_axis_offset_from_desk_rad",
                f64::from(bearing.sin().abs().atan2(bearing.cos().abs())),
            )?;
        }
    }
    if o.kind == ObjectKind::Elevator && scene.generator_version >= elevator::LANDING_VERSION {
        let (station, sill) = elevator::installation(o);
        category(
            categories,
            "elevator_sill_stock",
            format!("{:?}", sill.stock),
        );
        category(
            categories,
            "elevator_readout_mount",
            format!("{:?}", station.mount),
        );
        category(
            categories,
            "elevator_readout_film",
            format!("{:?}", station.display_film),
        );
        for direction in station.call_directions.iter().flatten() {
            category(
                categories,
                "elevator_call_direction",
                format!("{direction:?}"),
            );
        }
        for (name, value) in [
            ("elevator_landing_floor", f64::from(station.levels.floor)),
            (
                "elevator_displayed_floor",
                f64::from(station.displayed_floor),
            ),
            (
                "elevator_button_diameter_m",
                f64::from(station.button_diameter_m),
            ),
            ("elevator_track_count", sill.guide_centers_m.len() as f64),
            ("elevator_track_depth_m", f64::from(sill.guide_depth_m)),
            (
                "elevator_car_landing_joint_m",
                f64::from(sill.joint_width_m),
            ),
        ] {
            numeric.push(name, value)?;
        }
    }
    record_equipment(scene, o, categories, numeric)?;
    service_equipment::record(o, scene.generator_version, categories, numeric)?;
    service_furniture::record(o, scene.generator_version, categories, numeric)?;
    use super::super::objects::accessories::organizers;
    if scene.generator_version >= organizers::INTRODUCED
        && matches!(o.kind, ObjectKind::DocumentTray | ObjectKind::DeskOrganizer)
    {
        let p = organizers::parameters(o);
        for bay in organizers::compartments(o, &p) {
            let kind = match bay {
                organizers::Contents::Empty => "empty",
                organizers::Contents::Tools { tools, .. } => {
                    numeric.push("organizer_writing_tools", tools.len() as f64)?;
                    for pen in tools {
                        numeric.push(
                            "organizer_tool_lean_deg",
                            f64::from(pen.inclination_rad().to_degrees()),
                        )?;
                    }
                    "tools"
                }
                organizers::Contents::Paper(stack) => {
                    category(categories, "organizer_paper_media", stack.media);
                    category(
                        categories,
                        "organizer_printed_face_up",
                        stack.printed_face_up,
                    );
                    numeric.push("organizer_sheet_count", f64::from(stack.sheets))?;
                    numeric.push("organizer_leaf_curl_m", f64::from(stack.curl_m))?;
                    "paper"
                }
            };
            category(categories, "organizer_compartment", kind);
        }
    }
    use super::super::objects::desktop::{self, pointer};
    if o.kind == ObjectKind::Mouse && scene.generator_version >= pointer::INTRODUCED {
        let p = pointer::PointerProgram::sample(o, &desktop::parameters(o));
        let control = match p.control {
            pointer::Control::ScrollWheel { ribs, .. } => {
                numeric.push("mouse_wheel_flutes", f64::from(ribs))?;
                "scroll_wheel"
            }
            pointer::Control::FingerTrackball { .. } => "finger_trackball",
            pointer::Control::ThumbTrackball { .. } => "thumb_trackball",
        };
        category(categories, "mouse_control", control);
        category(categories, "mouse_grip", format!("{:?}", p.grip));
        category(
            categories,
            "mouse_keyboard_side_known",
            p.keyboard_side_known,
        );
        if let Some(finish) = p.control.ball() {
            category(categories, "mouse_ball_finish", format!("{finish:?}"));
        }
        numeric.push("mouse_control_diameter_m", f64::from(2. * p.radius_m))?;
        numeric.push(
            "mouse_control_exposure_fraction",
            f64::from(p.exposure_fraction),
        )?;
    }
    if o.kind == ObjectKind::Chair && o.variant >= 10 {
        let chair = super::super::objects::chairs::TeachingSeat::sample(o);
        category(categories, "teaching_seat_family", chair.family);
        numeric.push("teaching_seat_depth_m", chair.seat_depth as f64)?;
        if let Some(tablet) = chair.tablet {
            let opening = chair.carry_grip_size();
            category(categories, "teaching_back_carry_opening", opening.is_some());
            if let Some(size) = opening {
                numeric.push("teaching_back_grip_width_m", f64::from(size.x))?;
                numeric.push("teaching_back_grip_height_m", f64::from(size.y))?;
            }
            category(
                categories,
                "writing_tablet_outline",
                format!("{:?}", tablet.outline),
            );
            category(
                categories,
                "writing_tablet_face",
                format!("{:?}", tablet.face_stock),
            );
            category(
                categories,
                "writing_tablet_edge",
                format!("{:?}", tablet.edge_stock),
            );
            numeric.push(
                "writing_tablet_yaw_degrees",
                f64::from(tablet.yaw.to_degrees()),
            )?;
            category(
                categories,
                "writing_tablet_side",
                if tablet.left_handed { "left" } else { "right" },
            );
        } else {
            category(categories, "auditorium_seat_folded", chair.folded);
            if chair.folded {
                numeric.push(
                    "auditorium_fold_degrees",
                    chair.fold_angle.to_degrees() as f64,
                )?;
            }
        }
    }
    kitchen_fittings::record(o, scene.generator_version, categories, numeric)?;
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
        if scene.generator_version >= super::super::objects::storage::hinged::INTRODUCED {
            if let Some(bank) = super::super::objects::storage::hinged::Bank::new(o, &p) {
                for leaf in (0..p.bays.len()).filter_map(|i| bank.leaf(o, &p, i)) {
                    category(categories, "cabinet_hinge_side", format!("{:?}", leaf.hand));
                    category(
                        categories,
                        "cabinet_hinged_panel",
                        format!("{:?}", leaf.face),
                    );
                    category(
                        categories,
                        "cabinet_hinged_pull",
                        format!("{:?}", leaf.pull),
                    );
                    numeric.push(
                        "cabinet_leaf_rest_degrees",
                        f64::from(leaf.rest_radians.to_degrees()),
                    )?;
                }
            }
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
            "clock_display",
            if p.digital { "digital" } else { "analog" },
        );
        numeric.push("clock_time_seconds", p.seconds as f64)?;
        numeric.push("clock_bezel_fraction", p.bezel_fraction as f64)?;
        if let Some(dial) = p.printed {
            category(categories, "clock_dial_style", format!("{:?}", dial.style));
            if p.digital {
                category(categories, "clock_24_hour", dial.hour_24);
            } else {
                category(categories, "clock_minute_rail", dial.minute_rail);
            }
        }
        category(
            categories,
            "clock_casing_substrate",
            format!("{:?}", p.casing),
        );
        numeric.push("clock_aspect", (o.size.x / o.size.y) as f64)?;
        let (back, front) = super::super::objects::clocks::case_planes(o);
        numeric.push("clock_case_depth_m", (front - back) as f64)?;
        numeric.push("clock_roundness", p.roundness as f64)?;
    }
    if o.kind.is_kitchen() {
        if let Some(fields) =
            super::super::objects::kitchen::metadata_for_version(o, scene.generator_version)
                .as_object()
        {
            if let Some(fruit) = fields.get("fruit").and_then(|v| v.as_array()) {
                if let Some(kind) = fields["vessel"]["kind"].as_str() {
                    category(categories, "fruit_bowl_carrier", kind);
                }
                for piece in fruit {
                    for (field, key) in [("variety", "fruit_variety"), ("skin", "fruit_skin")] {
                        if let Some(name) = piece[field].as_str() {
                            category(categories, key, name);
                        }
                    }
                    if let Some(height) = piece["height_m"].as_f64() {
                        numeric.push("fruit_height_m", height)?;
                    }
                }
            }
            if let Some(panel) = fields
                .get("construction")
                .and_then(|c| c.get("operator_panel"))
            {
                let prefix = format!("kitchen_{:?}", o.kind);
                if let Some(family) = panel["family"].as_str() {
                    category(categories, &format!("{prefix}_control_family"), family);
                }
                if let Some(mount) = panel["fascia"]["mount"].as_str() {
                    category(categories, &format!("{prefix}_control_mount"), mount);
                }
                for key in ["keys", "dials"] {
                    if let Some(controls) = panel[key].as_array() {
                        numeric.push(&format!("{prefix}_control_{key}"), controls.len() as f64)?;
                    }
                }
                category(
                    categories,
                    &format!("{prefix}_readout"),
                    panel["readout"]["film"].as_str().unwrap_or("none"),
                );
            }
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
        ObjectKind::Books | ObjectKind::Notebook => {
            use super::super::objects::literature::{
                parameters_at, spread::Spread, SPREAD_BINDING,
            };
            let p = parameters_at(o, scene.generator_version);
            if o.kind == ObjectKind::Notebook {
                category(categories, "notebook_open", p.opened);
            }
            if !p.opened {
                let (binding, back_up) = if o.kind == ObjectKind::Books {
                    ("reading_copy_binding", "reading_copy_back_cover_up")
                } else {
                    ("notebook_binding", "notebook_back_cover_up")
                };
                for i in 0..p.volumes {
                    let volume = p.volume(o, scene.generator_version, i);
                    category(
                        categories,
                        binding,
                        ["case", "soft", "wire"][usize::from(volume.binding)],
                    );
                    category(categories, back_up, volume.back_up);
                    if i > 0 {
                        category(categories, "book_stack_head_to_tail", volume.reversed);
                    }
                }
            }
            if p.opened && o.variant & SPREAD_BINDING != 0 {
                let spread = Spread::new(o, &p);
                let slot = materials::paper::spread_slot(o.seed);
                let stock = materials::paper::parameters(materials::variants::content_seed(
                    scene.material_seed(),
                    Surface::PrintedPaper,
                    slot,
                ));
                numeric.push("notebook_spread_width_m", f64::from(spread.size.x))?;
                numeric.push("notebook_spread_depth_m", f64::from(spread.size.z))?;
                numeric.push("notebook_page_balance", f64::from(p.page_balance))?;
                category(categories, "notebook_paper_layout", stock.layout);
            }
        }
        ObjectKind::Notepad => {
            let p = super::super::clutter::tabletop::pad::parameters(o);
            category(categories, "pad_binding", p.binding);
            numeric.push("pad_body_height_m", p.height as f64)?;
            numeric.push("pad_ring_pitch_m", p.ring_pitch as f64)?;
            let slot =
                materials::variants::object_slot(Surface::PrintedPaper, o.seed, o.finish_seed);
            let paper = materials::paper::parameters(materials::variants::content_seed(
                scene.material_seed(),
                Surface::PrintedPaper,
                slot,
            ));
            category(categories, "paper_layout", paper.layout);
            numeric.push("paper_ruling_spacing_texels", paper.spacing as f64)?;
            numeric.push("paper_writing_fraction", paper.writing as f64)?;
            numeric.push("paper_roughness", paper.roughness as f64)?;
        }
        ObjectKind::PenHolder => {
            use super::super::clutter::pen_holder;
            let p = pen_holder::parameters(o);
            let pens = pen_holder::contents(o, &p);
            category(categories, "pen_cup_finish", format!("{:?}", p.finish));
            numeric.push("pen_cup_requested_tools", p.requested_pens as f64)?;
            numeric.push("pen_cup_fitted_tools", pens.len() as f64)?;
            numeric.push(
                "pen_cup_wall_m",
                f64::from(p.body_span_m.min_element() * 0.5 * p.wall_fraction),
            )?;
            for pen in &pens {
                category(categories, "held_writing_mechanism", pen.tool.mechanism);
                category(
                    categories,
                    "held_writing_upper_end",
                    format!("{:?}", pen.upper_end),
                );
                numeric.push(
                    "held_writing_inclination_deg",
                    f64::from(pen.inclination_rad().to_degrees()),
                )?;
                numeric.push("held_writing_length_m", f64::from(pen.tool.length))?;
            }
        }
        ObjectKind::Pencil => {
            let p = super::super::clutter::tabletop::writing::parameters(o);
            category(categories, "writing_mechanism", p.mechanism);
            category(categories, "writing_clip", p.clip);
            category(
                categories,
                "writing_rest",
                if p.table_rest(o.seed).1 {
                    "BarrelAndClip"
                } else {
                    "BarrelFace"
                },
            );
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
        ObjectKind::TeachingBoard => {
            let p = super::super::objects::public::furnishing::parameters_at(
                o,
                scene.generator_version,
            );
            if let Some(board) = p.writing_board {
                category(
                    categories,
                    "teaching_board_arrangement",
                    format!("{:?}", board.arrangement),
                );
                if let Some(pad) = board.foot_pads {
                    numeric.push("teaching_board_foot_pad_height_m", f64::from(pad.height_m))?;
                }
                let leaves = board.leaves.len();
                numeric.push("teaching_board_leaves", leaves as f64)?;
                if leaves > 1 {
                    category(
                        categories,
                        "teaching_board_front_content",
                        if board.continuous_front {
                            "SharedDiagram"
                        } else {
                            "Independent"
                        },
                    );
                }
                let mut faces = leaves;
                for leaf in board.leaves {
                    numeric.push("teaching_board_leaf_width_m", f64::from(leaf.width_m))?;
                    if leaf.reverse_slot.is_some() {
                        faces += 1;
                        category(categories, "teaching_board_reverse", "Writable");
                    } else {
                        category(
                            categories,
                            "teaching_board_reverse",
                            format!("{:?}", board.rear_stock),
                        );
                    }
                    if let Some(hinge) = leaf.hinge {
                        numeric.push(
                            "teaching_board_wing_turn_deg",
                            f64::from(hinge.angle_rad.to_degrees()),
                        )?;
                    }
                }
                numeric.push("teaching_board_writable_faces", faces as f64)?;
                category(
                    categories,
                    "teaching_board_ledge",
                    if board.ledge.is_some() {
                        "Fitted"
                    } else {
                        "Absent"
                    },
                );
                if let Some(ledge) = board.ledge {
                    numeric.push("teaching_board_fitted_tools", ledge.contents.len() as f64)?;
                }
            }
        }
        ObjectKind::TrayReturn => {
            let case =
                super::super::objects::public::casework::parameters_at(o, scene.generator_version);
            if let Some(station) = case.tray_station {
                category(
                    categories,
                    "tray_return_bearing",
                    format!("{:?}", station.bearing),
                );
                category(
                    categories,
                    "tray_return_ware",
                    format!("{:?}", station.ware.kind),
                );
                category(
                    categories,
                    "tray_return_stock",
                    format!("{:?}", station.ware.stock),
                );
                let cells = station.bearings_m.len() * station.bay_centers_m.len();
                for (name, value) in [
                    ("tray_return_bays", station.bay_centers_m.len() as f64),
                    ("tray_return_levels", station.bearings_m.len() as f64),
                    (
                        "tray_return_loaded_fraction",
                        station.loads.len() as f64 / cells as f64,
                    ),
                    (
                        "tray_return_item_count",
                        station.loads.iter().map(|load| load.count).sum::<usize>() as f64,
                    ),
                    (
                        "tray_return_ware_long_side_m",
                        f64::from(station.ware.size_m.max_element()),
                    ),
                    ("tray_return_ware_depth_m", f64::from(station.ware.height_m)),
                ] {
                    numeric.push(name, value)?;
                }
            }
        }
        ObjectKind::TrashCan => {
            let p = super::super::objects::containers::bin_parameters(o);
            category(categories, "bin_lid", p.lid);
            category(categories, "bin_finish", format!("{:?}", p.finish));
            if let Some(inlet) = p.inlet {
                category(categories, "bin_station_inlet", format!("{:?}", inlet.kind));
                numeric.push(
                    "bin_station_inlet_width_m",
                    f64::from(inlet.width_depth_m.x),
                )?;
                numeric.push(
                    "bin_station_inlet_depth_m",
                    f64::from(inlet.width_depth_m.y),
                )?;
            }
            if let Some(film) = p.liner {
                category(categories, "bin_liner", format!("{:?}", film.mount));
                category(categories, "bin_liner_finish", format!("{:?}", film.finish));
            } else {
                category(categories, "bin_liner", "NoneVisible");
            }
            numeric.push("bin_squareness", p.squareness as f64)?;
            numeric.push("bin_taper", p.taper as f64)?;
            numeric.push("bin_rim_width_fraction", p.rim_width as f64)?;
        }
        ObjectKind::TissueBox => {
            let p = super::super::objects::containers::tissue_parameters(o);
            category(categories, "tissue_box_finish", format!("{:?}", p.finish));
            numeric.push("tissue_box_height_fraction", p.box_height as f64)?;
            use super::super::objects::containers::tissue::{Case, Draw};
            if let Some(d) = p.dispenser {
                category(categories, "tissue_case", format!("{:?}", d.case));
                numeric.push("tissue_case_height_m", f64::from(o.size.y * p.box_height))?;
                numeric.push("tissue_footprint_aspect", f64::from(o.size.x / o.size.z))?;
                category(categories, "tissue_draw_state", format!("{:?}", d.draw));
                if d.case == Case::FoldingCarton {
                    category(
                        categories,
                        "tissue_carton_print",
                        format!(
                            "{:?}",
                            super::super::materials::tissue_stock::pattern(
                                o,
                                scene.material_seed()
                            )
                        ),
                    );
                }
                numeric.push("tissue_mouth_width_m", f64::from(o.size.x * p.opening.x))?;
                numeric.push("tissue_mouth_depth_m", f64::from(o.size.z * p.opening.y))?;
                if d.draw != Draw::Sealed {
                    numeric.push("tissue_pack_fill_fraction", f64::from(d.fill_fraction))?;
                    numeric.push(
                        "tissue_drawn_height_m",
                        f64::from(o.size.y * (1. - p.box_height)),
                    )?;
                }
            }
            if p.dispenser.is_none_or(|d| d.draw != Draw::Sealed) {
                numeric.push("tissue_sheet_fold", p.fold as f64)?;
                numeric.push("tissue_sheet_lean", p.lean as f64)?;
            }
        }
        ObjectKind::Mug | ObjectKind::CoffeeCup => {
            drinkware::cup(o, scene.generator_version, categories, numeric)?
        }
        ObjectKind::SodaCan => drinkware::can(o, categories, numeric)?,
        ObjectKind::WaterBottle => drinkware::bottle(o, categories, numeric)?,
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
        ObjectKind::Thermostat => {
            use super::super::objects::wall::thermostat;
            let p = thermostat::parameters(o);
            category(categories, "thermostat_style", format!("{:?}", p.style));
            category(categories, "thermostat_case_tone", p.case_tone);
            category(categories, "thermostat_face", format!("{:?}", p.film));
            category(
                categories,
                "thermostat_unit",
                if p.fahrenheit { "F" } else { "C" },
            );
            numeric.push("thermostat_setpoint_c", p.setpoint_c as f64)?;
            numeric.push("thermostat_width_m", o.size.x as f64)?;
            numeric.push(
                "thermostat_center_above_floor_m",
                thermostat::height_above_floor(scene, o) as f64,
            )?;
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
            if let Some(bank) = &p.bank {
                category(categories, "outlet_face", format!("{:?}", bank.face));
                category(categories, "outlet_rows_per_gang", bank.rows);
                category(
                    categories,
                    "outlet_has_blank_module",
                    bank.blank_slot.is_some(),
                );
                numeric.push("outlet_mains_sockets", bank.socket_count() as f64)?;
                if bank.well_depth_m > 0. {
                    numeric.push("outlet_recess_depth_m", bank.well_depth_m as f64)?;
                }
            }
            if let Some(bank) = &p.switch_bank {
                use super::super::objects::wall::switches::Actuator;
                numeric.push(
                    "switch_cover_thickness_m",
                    (bank.face_z_m - bank.rear_z_m) as f64,
                )?;
                numeric.push(
                    "switch_recess_depth_m",
                    (bank.face_z_m - bank.pocket_z_m) as f64,
                )?;
                numeric.push(
                    "switch_control_height_above_floor_m",
                    (o.position.y + bank.case_size.y * 0.5 - scene.floor_height(o.position.xz()))
                        as f64,
                )?;
                let mixed = bank
                    .modules
                    .iter()
                    .any(|m| m.name() != bank.modules[0].name());
                category(categories, "switch_mixed_controls", mixed);
                for &control in &bank.modules {
                    category(categories, "switch_actuator", control.name());
                    match control {
                        Actuator::Paddle { angle_rad } => {
                            numeric.push("switch_paddle_deg", angle_rad.to_degrees() as f64)?
                        }
                        Actuator::Split { angles_rad } => {
                            for angle in angles_rad {
                                numeric.push("switch_paddle_deg", angle.to_degrees() as f64)?;
                            }
                        }
                        Actuator::Toggle { angle_rad } => {
                            numeric.push("switch_toggle_deg", angle_rad.to_degrees() as f64)?
                        }
                        Actuator::Rotary { angle_rad } => {
                            numeric.push("switch_dimmer_deg", angle_rad.to_degrees() as f64)?
                        }
                        Actuator::Slider { level } => {
                            numeric.push("switch_slider_level", level as f64)?
                        }
                        Actuator::Buttons { latched } => {
                            category(categories, "switch_button_latched", latched.is_some())
                        }
                    }
                }
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
        use super::super::objects::computers::displays::{self, enclosure};
        let p = displays::display_parameters(o);
        let d = displays::geometry_at_version(o, &p, scene.generator_version);
        use super::super::materials::input_housing::{Housing, FINISH_VERSION};
        if scene.generator_version >= FINISH_VERSION {
            let (process, tone, roughness) =
                Housing::for_object(o).display_case_metrics(p.casing, scene.material_seed());
            category(categories, "display_case_finish_process", process);
            category(categories, "display_case_tone", tone);
            if let Some(value) = roughness {
                numeric.push("display_case_nominal_roughness", value as f64)?;
            }
        }
        if scene.generator_version >= enclosure::INTRODUCED {
            let section = enclosure::PanelSection::sample(o, &d);
            category(
                categories,
                "display_glass_fit",
                format!("{:?}", section.glass_fit),
            );
            numeric.push("display_glass_offset_m", section.glass_offset_m as f64)?;
            numeric.push("display_case_edge_bevel_m", section.edge_bevel_m as f64)?;
            numeric.push(
                "display_rear_shoulder_fraction",
                section.shoulder_fraction as f64,
            )?;
        }
        numeric.push("display_aspect", (d.width_m / d.height_m) as f64)?;
        numeric.push("display_width_m", d.width_m as f64)?;
        numeric.push("display_height_m", d.height_m as f64)?;
        numeric.push("display_tilt_radians", d.tilt_radians as f64)?;
        numeric.push("display_bezel_m", d.bezel_m as f64)?;
        numeric.push("display_panel_m", d.panel_m as f64)?;
        numeric.push("display_rear_width_fraction", p.back_fraction.x as f64)?;
        numeric.push("display_rear_height_fraction", p.back_fraction.y as f64)?;
        if o.kind == ObjectKind::Monitor {
            numeric.push("display_curvature_radius_m", d.curvature_radius_m as f64)?;
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
        numeric.push("workspace_top_height_m", o.size.y as f64)?;
        numeric.push("workspace_monitor_count", monitors as f64)?;
        numeric.push("workspace_laptop_count", laptops as f64)?;
    }
    if o.kind == ObjectKind::PcTower {
        category(
            categories,
            "pc_case_placement",
            if o.support.is_some() {
                "desktop"
            } else {
                "floor"
            },
        );
    }
    if o.kind == ObjectKind::Cable {
        let path = o.cable_path.as_ref().ok_or("cable missing route")?;
        if let Some((source, target)) = o
            .interaction_target
            .and_then(|id| scene.objects.get(id))
            .zip(o.cable_target.and_then(|id| scene.objects.get(id)))
        {
            category(
                categories,
                "cable_devices",
                format!(
                    "{} to {}",
                    source.kind.class_name(),
                    target.kind.class_name()
                ),
            );
        }
        category(
            categories,
            "cable_function",
            match o.variant {
                equipment::wires::USB_LEAD => "usb",
                equipment::wires::DISPLAY_LEAD => "display",
                _ => "power",
            },
        );
        numeric.push(
            "cable_length_m",
            path.windows(2).map(|w| w[0].distance(w[1]) as f64).sum(),
        )?;
        if let Some(socket) = o.cable_target_socket {
            category(categories, "cable_used_first_mains_socket", socket == 0);
            numeric.push("cable_mains_socket_index", f64::from(socket))?;
        }
        numeric.push("cable_radius_m", equipment::wires::radius(o) as f64)?;
        numeric.push("cable_route_points", path.len() as f64)?;
    }
    if o.kind == ObjectKind::WallArt {
        let p = materials::art::parameters(finish_seed(scene, o));
        let frame = materials::art::frame_parameters(o);
        let set = super::super::objects::presentation::artwork::canvas_set(o);
        numeric.push(
            "art_panel_count",
            set.as_ref().map_or(1, |s| s.panels.len()) as f64,
        )?;
        if let Some(set) = set {
            category(categories, "art_panel_axis", format!("{:?}", set.axis));
            numeric.push("art_panel_gap_m", f64::from(set.gap_m))?;
        }
        category(categories, "art_subject", p.subject);
        category(categories, "art_frame", frame.style);
        if let Some(section) = frame.moulding {
            category(
                categories,
                "art_moulding_shape",
                format!("{:?}", section.shape),
            );
            numeric.push("art_moulding_relief_m", section.rise_m as f64)?;
        }
        if frame.mat_thickness_m > 0. {
            numeric.push("art_mat_thickness_m", frame.mat_thickness_m as f64)?;
            numeric.push("art_mat_bevel_m", frame.mat_bevel_m as f64)?;
        }
        if frame.float_depth_m > 0. {
            numeric.push("art_float_depth_m", frame.float_depth_m as f64)?;
        }
        if matches!(frame.style, 0 | 4) {
            category(categories, "art_printed_returns", frame.image_wrap);
        }
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
        if let Some(camera) = p.front_camera {
            category(
                categories,
                "phone_front_camera_mount",
                format!("{:?}", camera.mount),
            );
            numeric.push(
                "phone_front_camera_x_fraction",
                (camera.center.x / p.span.x) as f64,
            )?;
            numeric.push(
                "phone_front_camera_width_m",
                (camera.half_span.x * 2.) as f64,
            )?;
            numeric.push("phone_screen_width_m", p.screen_span.x as f64)?;
            numeric.push("phone_screen_length_m", p.screen_span.y as f64)?;
        } else {
            category(
                categories,
                "phone_flash_location",
                match p.rear_flash {
                    Some(f) if f.on_island => "island",
                    Some(_) => "back_cover",
                    None => "absent",
                },
            );
            category(categories, "phone_glazed_camera_island", p.glazed_island);
            category(categories, "phone_camera_layout", p.camera_layout);
            category(categories, "phone_lens_count", p.lens_count);
            category(categories, "phone_back", format!("{:?}", p.back));
            numeric.push("phone_lens_radius_m", p.lens_radius as f64)?;
            for radius in &p.lens_radii[..p.lens_count] {
                numeric.push("phone_individual_lens_radius_m", f64::from(*radius))?;
            }
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
            use super::super::layout::TelevisionMount;
            if let Some(mount) = scene.television_mount(o) {
                category(
                    categories,
                    "tv_mount",
                    match mount {
                        TelevisionMount::PerimeterWall => "perimeter_wall",
                        TelevisionMount::Partition(_) => "partition",
                        TelevisionMount::Pillar(_) => "pillar",
                    },
                );
            }
            category(
                categories,
                "tv_in_primary_room",
                scene
                    .primary_region()
                    .contains_projection(o.position.xz(), 0.),
            );
            numeric.push(
                "tv_lower_edge_above_floor_m",
                (o.position.y - scene.floor_height(o.position.xz())) as f64,
            )?;
        }
        category(categories, "screen_dormant", p.dormant);
        if o.kind == ObjectKind::Phone {
            let state = match &p.ambient_clock {
                Some(clock) if clock.stacked => "ambient_stacked",
                Some(_) => "ambient_horizontal",
                None if p.dormant => "off",
                None => "active",
            };
            category(categories, "phone_display_state", state);
        }
        numeric.push(
            "screen_luminance",
            if p.dormant { 0. } else { p.luminance as f64 },
        )?;
        if o.kind == ObjectKind::Phone && p.dormant {
            return Ok(());
        }
        numeric.push("screen_time_minutes", p.minutes as f64)?;
        for (key, value) in [
            ("screen_white_kelvin", p.white_kelvin),
            ("screen_white_tint", p.white_tint),
        ] {
            numeric.push(key, value as f64)?;
        }
        if p.ambient_clock.is_some() {
            return Ok(());
        }
        for (key, value) in [
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
        category(categories, "screen_content", p.layout);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bookcase_metrics_follow_installed_bays_and_versioned_supports() {
        use super::super::super::objects::{audit, storage::bookcase};
        let directory = tempfile::tempdir().unwrap();
        let mut scene = IndoorManifest::generate_with_humans(
            41803,
            super::super::IndoorLayout::OpenOffice,
            0.65,
            0,
            0.,
        )
        .unwrap();
        for version in [bookcase::RACK_VERSION - 1, bookcase::RACK_VERSION] {
            scene.generator_version = version;
            let mut categories = Categories::new();
            let mut numeric = NumericCollector::new(directory.path()).unwrap();
            let mut bays = 0;
            let mut compartments = 0;
            let mut racks = 0;
            let mut fill = 0.;
            for seed in 0..32 {
                let o = audit::fixture(ObjectKind::Bookcase, seed);
                let p = bookcase::parameters_at(&o, version);
                bays += p.bays.len();
                compartments += p.bays[0].levels.len() - 1;
                fill += f64::from(p.fill);
                racks += usize::from(p.rack.is_some());
                record_object(&scene, &o, &mut categories, &mut numeric).unwrap();
            }
            let stats = numeric.finish().unwrap();
            assert_eq!(stats["bookshelf_columns"].mean, bays as f64 / 32.);
            assert_eq!(stats["bookshelf_levels"].mean, compartments as f64 / 32.);
            assert_eq!(stats["bookshelf_fill"].mean, fill / 32.);
            assert_eq!(stats["bookshelf_usable_depth_m"].count, 32);
            assert_eq!(
                categories
                    .get("bookshelf_rear_bracing")
                    .into_iter()
                    .flat_map(|v| v.values())
                    .sum::<usize>(),
                racks
            );
            assert_eq!(
                stats.get("bookshelf_deck_bearing_m").map_or(0, |s| s.count),
                racks
            );
        }
    }

    #[test]
    fn sofa_throw_metrics_count_the_cushions_installed_on_each_sofa() {
        use super::super::super::objects::audit;
        let directory = scratch();
        let mut scene = IndoorManifest::generate_with_humans(
            97504,
            super::super::IndoorLayout::Lounge,
            0.65,
            0,
            0.,
        )
        .unwrap();
        for version in [
            seating::pillows::INTRODUCED - 1,
            seating::pillows::INTRODUCED,
        ] {
            scene.generator_version = version;
            let mut categories = Categories::new();
            let mut numeric = NumericCollector::new(&directory).unwrap();
            let mut installed = 0usize;
            let sofas = 16;
            for seed in 0..sofas {
                let o = audit::fixture(ObjectKind::Sofa, seed);
                let p = seating::parameters(&o);
                installed +=
                    seating::pillows::metadata(&o, &p, &seating::module_widths_for(&o, &p)).len();
                record_object(&scene, &o, &mut categories, &mut numeric).unwrap();
            }
            assert!(installed > 0);
            let values = numeric.finish().unwrap();
            if version < seating::pillows::INTRODUCED {
                assert!(
                    !values.contains_key("sofa_throw_count")
                        && !categories.contains_key("sofa_throw_edge")
                );
            } else {
                assert_eq!(
                    values["sofa_throw_count"].mean,
                    installed as f64 / sofas as f64
                );
                assert_eq!(
                    categories["sofa_throw_edge"].values().sum::<usize>(),
                    installed
                );
                assert_eq!(
                    categories["sofa_throw_support"].values().sum::<usize>(),
                    installed
                );
            }
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn organizer_metrics_describe_the_built_compartments_at_the_captured_version() {
        use super::super::super::objects::{accessories::organizers, audit};
        let directory = scratch();
        let mut scene = IndoorManifest::generate_with_humans(
            94018,
            super::super::IndoorLayout::OpenOffice,
            0.65,
            0,
            0.,
        )
        .unwrap();
        for version in [organizers::INTRODUCED - 1, organizers::INTRODUCED] {
            scene.generator_version = version;
            let mut categories = Categories::new();
            let mut numeric = NumericCollector::new(&directory).unwrap();
            let mut counts = [0usize; 3];
            let mut sheets = 0;
            let mut pens = 0;
            for seed in 0..16 {
                for kind in [ObjectKind::DeskOrganizer, ObjectKind::DocumentTray] {
                    let o = audit::fixture(kind, seed);
                    for b in organizers::compartments(&o, &organizers::parameters(&o)) {
                        match b {
                            organizers::Contents::Empty => counts[0] += 1,
                            organizers::Contents::Paper(s) => {
                                counts[1] += 1;
                                sheets += usize::from(s.sheets);
                            }
                            organizers::Contents::Tools { tools, .. } => {
                                counts[2] += 1;
                                pens += tools.len();
                            }
                        }
                    }
                    record_object(&scene, &o, &mut categories, &mut numeric).unwrap();
                }
            }
            let values = numeric.finish().unwrap();
            if version >= organizers::INTRODUCED {
                let bins = &categories["organizer_compartment"];
                for (i, name) in ["empty", "paper", "tools"].into_iter().enumerate() {
                    assert_eq!(bins[name], counts[i]);
                }
                assert_eq!(
                    values["organizer_sheet_count"].mean,
                    sheets as f64 / counts[1] as f64
                );
                assert_eq!(
                    values["organizer_writing_tools"].mean,
                    pens as f64 / counts[2] as f64
                );
            } else {
                assert!(
                    !categories.contains_key("organizer_compartment")
                        && !values.contains_key("organizer_sheet_count")
                );
            }
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn fitted_hood_metrics_count_installed_cassettes_and_active_features() {
        use crate::scene::procedural_indoor::objects::{audit, public::service::hood};
        let directory = scratch();
        for version in [hood::FITTED_INTAKE - 1, hood::FITTED_INTAKE] {
            let mut numeric = NumericCollector::new(&directory).unwrap();
            let mut categories = Categories::new();
            let mut counts = [0usize; 2];
            for seed in 0..64 {
                let mut o = audit::fixture(ObjectKind::ExtractionHood, seed);
                o.size.x = 0.45;
                let deck = hood::FilterDeck::fit(&o, &hood::hood_parameters(&o));
                counts[0] += deck.cassettes;
                counts[1] += deck.lamps;
                kitchen_fittings::record(&o, version, &mut categories, &mut numeric).unwrap();
            }
            let values = numeric.finish().unwrap();
            if version < hood::FITTED_INTAKE {
                assert!(categories.is_empty() && values.is_empty());
            } else {
                let styles = &categories["hood_filter_style"];
                assert_eq!(values["hood_baffle_pitch_m"].count, styles["folded_baffle"]);
                assert_eq!(
                    values["hood_perimeter_gap_m"].count,
                    styles["perimeter_panel"]
                );
                assert_eq!(styles.values().sum::<usize>(), 64);
                assert_eq!(categories["hood_lamp_style"].values().sum::<usize>(), 64);
                assert_eq!(values["hood_filter_count"].mean, counts[0] as f64 / 64.);
                assert_eq!(values["hood_lamp_count"].mean, counts[1] as f64 / 64.);
            }
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn kitchen_control_metrics_follow_the_manifest_generator() {
        use super::super::super::objects::{audit, kitchen::appliances::controls};
        let directory = scratch();
        let mut scene = IndoorManifest::generate_with_humans(
            94000,
            super::super::IndoorLayout::Kitchen,
            0.65,
            0,
            0.,
        )
        .unwrap();
        let o = audit::fixture(ObjectKind::Dishwasher, 14);
        let panel = controls::for_object(&o).unwrap();
        for version in [controls::INTRODUCED - 1, controls::INTRODUCED] {
            scene.generator_version = version;
            let mut categories = Categories::new();
            let mut numeric = NumericCollector::new(&directory).unwrap();
            record_object(&scene, &o, &mut categories, &mut numeric).unwrap();
            let values = numeric.finish().unwrap();
            let count = values
                .get("kitchen_Dishwasher_control_keys")
                .map(|v| v.mean);
            if version < controls::INTRODUCED {
                assert!(count.is_none());
                assert!(!categories.contains_key("kitchen_Dishwasher_control_mount"));
            } else {
                assert_eq!(count, Some(panel.keys.len() as f64));
                assert_eq!(
                    values["kitchen_Dishwasher_control_dials"].mean,
                    panel.dials.len() as f64
                );
                let mount = serde_json::to_value(panel.fascia.mount).unwrap();
                assert_eq!(
                    categories["kitchen_Dishwasher_control_mount"][mount.as_str().unwrap()],
                    1
                );
            }
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn divider_metrics_count_installed_bays_and_only_real_hinges() {
        use super::super::super::objects::{audit, services::divider};
        let directory = scratch();
        let mut categories = Categories::new();
        let mut numeric = NumericCollector::new(&directory).unwrap();
        let mut hinges = Vec::new();
        let mut types = BTreeMap::<String, usize>::new();
        for seed in 0..96 {
            let o = audit::fixture(ObjectKind::RoomDivider, seed);
            let p = divider::parameters(&o);
            service_furniture::record(
                &o,
                super::super::super::layout::GENERATOR_VERSION,
                &mut categories,
                &mut numeric,
            )
            .unwrap();
            *types
                .entry(format!("{:?}", p.plan.arrangement))
                .or_default() += 1;
            for edges in p.plan.hinge_centres_xz.windows(3) {
                let a = (edges[1] - edges[0]).as_dvec2();
                let b = (edges[2] - edges[1]).as_dvec2();
                hinges.push((b.y.atan2(b.x) - a.y.atan2(a.x)).to_degrees());
            }
        }
        let mut desktop = audit::fixture(ObjectKind::RoomDivider, 37);
        desktop.variant = divider::desktop::VARIANT;
        desktop.support = Some(4);
        desktop.size = bevy::prelude::Vec3::new(1.14, 0.49, 0.15);
        let desk_program = divider::desktop::parameters(&desktop);
        service_furniture::record(
            &desktop,
            divider::desktop::INTRODUCED,
            &mut categories,
            &mut numeric,
        )
        .unwrap();
        assert_eq!(categories["divider_location"]["floor"], 96);
        assert_eq!(categories["divider_location"]["desk"], 1);
        assert_eq!(
            categories["desk_screen_mount"][&format!("{:?}", desk_program.mount)],
            1
        );
        assert_eq!(categories["divider_arrangement"], types);
        assert_eq!(categories["divider_bay_count"].values().sum::<usize>(), 96);
        let values = numeric.finish().unwrap();
        assert_eq!(
            values["desk_screen_width_m"].mean,
            f64::from(desktop.size.x)
        );
        assert_eq!(
            values["desk_screen_cable_gap_m"].mean,
            f64::from(desk_program.cable_gap_m)
        );
        let turns = &values["divider_hinge_turn_deg"];
        assert_eq!(turns.count, hinges.len());
        assert!((turns.mean - hinges.iter().sum::<f64>() / hinges.len() as f64).abs() < 1e-4);
        assert_eq!(values["divider_insert_thickness_m"].count, 96);
        std::fs::remove_dir_all(directory).unwrap();
    }
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
        let notices = audit::fixture(ObjectKind::NoticeBoard, 19);
        let shelf = audit::fixture(ObjectKind::WallShelf, 9);
        let screen = audit::fixture(ObjectKind::ProjectionScreen, 10);
        for o in [&board, &shelf, &screen, &notices] {
            record_object(&scene, o, &mut categories, &mut numeric).unwrap();
        }
        let values = numeric.finish().unwrap();
        let b = presentation::parameters(&board);
        let s = equipment::wall_shelf::parameters(&shelf);
        let p = presentation::projection::parameters(&screen);
        let notes = equipment::pinups::placements(&notices);
        assert_eq!(values["noticeboard_note_count"].mean, notes.len() as f64);
        let facing = equipment::board_section::Facing::for_board(&notices, scene.generator_version);
        assert_eq!(categories["noticeboard_facing"][&format!("{facing:?}")], 1);
        let frame = equipment::board_section::parameters(&notices).frame_surface;
        assert_eq!(categories["noticeboard_frame"][&format!("{frame:?}")], 1);
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
                        materials::variants::content_seed(scene.material_seed(), *s, slot),
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
            cable_target_socket: None,
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
            let pendant_has_diffuser = kind == ObjectKind::CeilingPendant
                && equipment::pendant::parameters(&o)
                    .diffuser_fraction()
                    .is_some();
            assert_eq!(
                values.contains_key("equipment_diffuser_radius_m"),
                pendant_has_diffuser
            );
            assert_eq!(
                values.contains_key("equipment_diffuser_fraction"),
                kind == ObjectKind::WallSconce || pendant_has_diffuser
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
                let (beam, power) =
                    super::super::super::architecture::object_fixture_photometry(&scene, &o);
                assert_eq!(values["equipment_fixture_lumens"].mean, power.lumens as f64);
                assert_eq!(
                    values["equipment_emitter_room_x_m"].mean,
                    beam.position.x as f64
                );
            }
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn art_panel_metrics_measure_the_installed_mounts() {
        use super::super::super::objects::presentation::artwork;
        let directory = scratch();
        let scene = IndoorManifest::generate_with_humans(
            26,
            super::super::IndoorLayout::Mixed,
            0.65,
            0,
            0.,
        )
        .unwrap();
        let mut categories = Categories::new();
        let mut numeric = NumericCollector::new(&directory).unwrap();
        let mut art = object(ObjectKind::WallArt);
        art.size = Vec3::new(1.8, 0.85, 0.042);
        art.variant = 4;
        record_object(&scene, &art, &mut categories, &mut numeric).unwrap();
        art.variant = 10;
        record_object(&scene, &art, &mut categories, &mut numeric).unwrap();
        let set = artwork::canvas_set(&art).unwrap();
        let values = numeric.finish().unwrap();
        assert_eq!(values["art_panel_count"].count, 2);
        assert_eq!(
            values["art_panel_count"].mean,
            (1 + set.panels.len()) as f64 / 2.
        );
        assert_eq!(values["art_panel_gap_m"].count, 1);
        assert_eq!(values["art_panel_gap_m"].mean, f64::from(set.gap_m));
        assert_eq!(categories["art_panel_axis"][&format!("{:?}", set.axis)], 1);
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
            materials::variants::content_seed(
                991,
                surface,
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
