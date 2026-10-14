//! Built furnishing/content diagnostics, with the same finish seeds and physical
//! emitters used by the renderer. Equipment fields exclude dormant parameters.
use super::super::{
    cameras::viewpoint::ViewpointProgram,
    layout::{IndoorManifest, IndoorObject, ObjectKind},
    materials::{self, program::MaterialRecipe, Surface},
    metrics_sort::NumericCollector,
    objects::equipment,
};
use std::{collections::BTreeMap, io};

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
) -> io::Result<()> {
    let groups = &scene.assembly_diagnostics;
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
    ] {
        numeric.push(name, value as f64)?;
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
    Ok(())
}

pub(super) fn record_material(
    material: &MaterialRecipe,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
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
    materials::variants::screen_seed(
        scene.material_seed(),
        materials::variants::slot(o.finish_seed.unwrap_or(o.seed)),
    )
}

fn record_equipment(
    scene: &IndoorManifest,
    o: &IndoorObject,
    categories: &mut Categories,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    if !matches!(
        o.kind,
        ObjectKind::Projector
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
            let foot = (o.size.y * 0.09).min(0.013);
            let corner = p
                .corner_m
                .min(o.size.x.min(o.size.y - foot).min(o.size.z * 0.91) * 0.45);
            category(categories, "projector_housing", format!("{:?}", p.housing));
            for (name, value) in [
                ("equipment_corner_m", corner),
                ("projector_lens_fraction", p.lens_fraction),
                ("projector_lens_offset", p.lens_offset),
                (
                    "projector_lens_radius_m",
                    o.size.y.min(o.size.x) * p.lens_fraction,
                ),
                ("equipment_vent_count", p.vent_count as f32),
            ] {
                numeric.push(name, value as f64)?;
            }
        }
        ObjectKind::PowerStrip => {
            category(categories, "powerstrip_housing", format!("{:?}", p.housing));
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
            numeric.push("equipment_corner_m", p.corner_m.min(o.size.y * 0.22) as f64)?;
            numeric.push(
                "equipment_socket_count",
                p.socket_count
                    .min((o.size.x / 0.039).floor().max(3.) as usize) as f64,
            )?;
        }
        ObjectKind::WallShelf => {
            category(
                categories,
                "wall_shelf_surface",
                format!(
                    "{:?}",
                    if p.housing == Surface::Metal {
                        Surface::Metal
                    } else {
                        Surface::Wood
                    }
                ),
            );
            numeric.push("wall_shelf_brackets", p.shelf_brackets as f64)?;
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
                numeric.push(
                    "equipment_diffuser_width_m",
                    (o.size.x * p.diffuser_fraction) as f64,
                )?;
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
    match o.kind {
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
        ObjectKind::WallOutlet | ObjectKind::LightSwitch => {
            let p = super::super::objects::wall::parameters(o);
            let name = if o.kind == ObjectKind::WallOutlet {
                "outlet"
            } else {
                "switch"
            };
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
        if o.kind == ObjectKind::Monitor {
            numeric.push("display_curvature_radius_m", d.curvature_radius_m as f64)?;
            numeric.push("display_tilt_radians", d.tilt_radians as f64)?;
            category(categories, "display_stand", p.stand);
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
            ("art_horizon", p.horizon),
            ("art_relief", p.relief),
            ("art_viewpoint", p.viewpoint),
            ("art_hue", p.hue),
            ("art_saturation", p.saturation),
            ("art_density", p.density),
            ("art_brush", p.brush),
            ("art_border_m", frame.border_width),
            ("art_mat_m", frame.mat_width),
            ("art_shadow_gap_m", frame.shadow_gap),
        ] {
            numeric.push(key, value as f64)?;
        }
    }
    if o.kind == ObjectKind::Whiteboard {
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
    ) {
        let p = materials::screens::parameters(finish_seed(scene, o));
        if o.kind == ObjectKind::Display {
            category(categories, "tv_content", p.layout);
        }
        numeric.push("screen_time_minutes", p.minutes as f64)?;
        numeric.push(
            "screen_luminance",
            if p.dormant { 0. } else { p.luminance as f64 },
        )?;
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
        category(categories, "screen_dormant", p.dormant);
        category(categories, "screen_content", p.layout);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
            if kind == ObjectKind::PowerStrip {
                let p = equipment::parameters(&o);
                assert_eq!(
                    values["equipment_socket_count"].mean,
                    p.socket_count.min(4) as f64
                );
            }
            if matches!(kind, ObjectKind::WallSconce | ObjectKind::CeilingPendant) {
                let (position, _, lumens, _, _) = equipment::fixture_photometry(&o);
                assert_eq!(
                    values["equipment_fixture_lumens"].mean,
                    (lumens * 0.25) as f64
                );
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
        record_scene(&scene, &mut numeric).unwrap();
        for kind in [
            ObjectKind::WallArt,
            ObjectKind::Whiteboard,
            ObjectKind::Display,
        ] {
            record_object(&scene, &object(kind), &mut categories, &mut numeric).unwrap();
        }
        let actual_seed = materials::variants::screen_seed(991, materials::variants::slot(83));
        let art = materials::art::parameters(actual_seed);
        let board = materials::boards::parameters(actual_seed);
        let screen = materials::screens::parameters(actual_seed);
        let values = numeric.finish().unwrap();
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
