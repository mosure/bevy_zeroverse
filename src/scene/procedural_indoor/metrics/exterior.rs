//! Exterior counts describe built apertures, excluding the neighboring room.
use super::{CoverageReport, IndoorManifest};
use crate::scene::procedural_indoor::{
    architecture::facade::{SashHinge, Shade},
    metrics_sort::NumericCollector,
};
use std::io::{self, Write};

pub(super) fn header(writer: &mut impl Write) -> io::Result<()> {
    writeln!(
        writer,
        "seed,side,opening,left_m,bottom_m,right_m,top_m,width_m,height_m,head_clearance_m,columns,transom,operable,frame,frame_width_m,frame_depth_m,recess_m,shade,shade_coverage,full_height,crossbar_count,transom_fractions,facade_shade_coverage,fixed_panes,operating_panes,open_panes,hinges"
    )
}

pub(super) fn record(
    scene: &IndoorManifest,
    report: &mut CoverageReport,
    numeric: &mut NumericCollector,
    writer: &mut impl Write,
) -> io::Result<()> {
    // Edge identifiers and dimensions come from the built geometry. The legacy
    // three-side program is only used when replaying a rectangular envelope.
    let facades: Vec<_> = if let Some(e) = &scene.envelope {
        e.walls
            .iter()
            .filter_map(|w| {
                w.facade.as_ref().map(|f| {
                    let a = e.footprint[w.edge];
                    let b = e.footprint[(w.edge + 1) % e.footprint.len()];
                    (
                        format!("edge{}", w.edge),
                        a.distance(b),
                        e.ceiling_height(scene.room_size, a)
                            .min(e.ceiling_height(scene.room_size, b)),
                        f,
                    )
                })
            })
            .collect()
    } else if let Some(e) = &scene.exterior {
        numeric.push(
            "exterior_exposure_probability",
            e.exposure_probability as f64,
        )?;
        e.facades
            .iter()
            .map(|f| {
                (
                    format!("{:?}", f.side),
                    f.side.span(scene.room_size),
                    scene.room_size.y,
                    f,
                )
            })
            .collect()
    } else {
        return Ok(());
    };
    let mut category = |key: &str, value: String| {
        *report
            .categories
            .entry(key.into())
            .or_default()
            .entry(value)
            .or_default() += 1;
    };
    category("exterior_wall_count", facades.len().to_string());
    category(
        "exterior_sides",
        facades
            .iter()
            .map(|(side, ..)| side.as_str())
            .collect::<Vec<_>>()
            .join("+"),
    );
    let mut full_height = 0;
    let mut openings = 0;
    for (side, span, height, f) in facades {
        category("exterior_side", side.clone());
        category("exterior_frame", format!("{:?}", f.frame));
        category("exterior_shade", format!("{:?}", f.shade));
        let area: f32 = f.openings.iter().map(|o| o.area()).sum();
        for (key, value) in [
            ("exterior_opening_area_fraction", area / (span * height)),
            ("exterior_frame_width_m", f.frame_width),
            ("exterior_frame_depth_m", f.frame_depth),
            ("exterior_recess_m", f.recess),
        ] {
            numeric.push(key, value as f64)?;
        }
        if f.shade != Shade::None {
            numeric.push("exterior_shade_coverage", f.shade_coverage as f64)?;
        }
        if let Some(j) = &f.joinery {
            if let Some(r) = &j.reveal {
                category("window_lining_material", format!("{:?}", r.lining_surface));
                category(
                    "window_surround_profile",
                    r.surround
                        .as_ref()
                        .map_or_else(|| "Absent".into(), |s| format!("{:?}", s.profile)),
                );
                numeric.push(
                    "window_reveal_depth_m",
                    (f.recess - f.frame_depth * 0.5) as f64,
                )?;
                numeric.push("window_lining_thickness_m", r.lining_thickness_m as f64)?;
                if let Some(s) = &r.surround {
                    numeric.push("window_surround_width_m", s.height_m as f64)?;
                    numeric.push("window_surround_projection_m", s.depth_m as f64)?;
                }
            }
            category("exterior_sill_material", format!("{:?}", j.sill_surface));
            category(
                "exterior_frame_finish_slot",
                j.frame_finish_slot.to_string(),
            );
            for (key, value) in [
                (
                    "exterior_mullion_width_m",
                    f.frame_width * j.mullion_fraction,
                ),
                (
                    "exterior_mullion_depth_m",
                    f.frame_depth * j.mullion_depth_fraction,
                ),
                ("exterior_sash_width_m", j.sash_width_m),
                ("exterior_sash_depth_m", j.sash_depth_m),
                ("exterior_sill_thickness_m", j.sill_thickness_m),
                (
                    "exterior_frame_bead_width_m",
                    f.frame_width * j.face_bead_fraction,
                ),
            ] {
                numeric.push(key, value as f64)?;
            }
        }
        let vertical = f
            .treatment
            .as_ref()
            .filter(|_| f.shade == Shade::Vertical)
            .and_then(|p| p.vertical.map(|stock| (p, stock)));
        let pleated = f
            .treatment
            .as_ref()
            .filter(|_| f.shade == Shade::Pleated)
            .and_then(|p| p.pleated.map(|stock| (p, stock)));
        if let Some(p) = f.treatment.as_ref().filter(|_| f.shade != Shade::None) {
            category("window_treatment_finish_slot", p.finish_slot.to_string());
            if let Some(stock) = p.textile {
                category("window_textile_lining", format!("{:?}", stock.lining));
                category("window_textile_colorway", stock.colorway.to_string());
            }
            if let Some(track) = p.track {
                numeric.push("window_panel_target_width_m", track.panel_width_m as f64)?;
                numeric.push("window_panel_reed_pitch_m", track.reed_pitch_m as f64)?;
                numeric.push("window_panel_weave_openness", track.openness as f64)?;
            }
            if matches!(f.shade, Shade::Curtain | Shade::Vertical) {
                category("window_treatment_stack", format!("{:?}", p.stack));
            }
            if f.shade == Shade::Venetian {
                category(
                    "window_treatment_slat_material",
                    format!("{:?}", p.slat_material),
                );
            } else {
                category("window_treatment_fabric", format!("{:?}", p.fabric));
            }
            if let Some((_, vane)) = vertical {
                numeric.push("window_vane_stock_width_m", vane.width_m as f64)?;
                numeric.push("window_vane_stock_crown_m", vane.crown_m as f64)?;
                numeric.push("window_vane_hem_depth_m", vane.hem_m as f64)?;
            } else if matches!(f.shade, Shade::Venetian | Shade::Vertical) {
                numeric.push("window_treatment_slat_pitch_m", p.slat_pitch as f64)?;
            }
            if matches!(f.shade, Shade::Curtain | Shade::Roman | Shade::Pleated) {
                numeric.push("window_treatment_fold_pitch_m", p.fold_pitch as f64)?;
                numeric.push("window_treatment_fold_depth_m", p.fold_depth as f64)?;
            }
            if f.shade == Shade::Curtain {
                if let Some(heading) = p.curtain_heading {
                    category("window_curtain_heading", format!("{heading:?}"));
                    numeric.push(
                        "window_curtain_parked_width_fraction",
                        heading.parked_fraction(p) as f64,
                    )?;
                }
                numeric.push("window_treatment_fullness", p.fullness as f64)?;
                numeric.push("window_treatment_tieback", p.tieback as f64)?;
            }
        }
        let mut min_draw = 1_f32;
        let mut max_draw = 0_f32;
        for (i, o) in f.openings.iter().enumerate() {
            let panes = f.pane_installations(o);
            let fixed = panes.iter().filter(|p| p.hinge == SashHinge::Fixed).count();
            let open = panes.iter().filter(|p| p.swing_radians > 0.).count();
            let operating = panes.len() - fixed;
            for pane in &panes {
                category("exterior_pane_hinge", format!("{:?}", pane.hinge));
                if let Some(handle) = pane.handle {
                    numeric.push("exterior_sash_handle_height_m", handle.y as f64)?;
                    numeric.push(
                        "exterior_sash_swing_degrees",
                        pane.swing_radians.to_degrees() as f64,
                    )?;
                }
            }
            let draw = f.shade_coverage_at(o);
            if let Some(stock) = f
                .treatment
                .as_ref()
                .and_then(|p| p.roller)
                .filter(|_| f.shade == Shade::Roller)
            {
                category("window_roller_housing", format!("{:?}", stock.head));
                category("window_roller_drive", format!("{:?}", stock.drive));
                numeric.push("window_roller_housing_radius_m", stock.radius_m as f64)?;
                numeric.push("window_roller_hem_height_m", stock.hem_height_m as f64)?;
                if let Some(drop) = stock.loop_drop(o.max.y - o.min.y - 0.028 - 0.095) {
                    numeric.push("window_roller_control_drop_m", drop as f64)?;
                }
            }
            if let Some((p, stock)) = pleated {
                let l = stock.layout(p, o, draw);
                category(
                    "window_pleat_construction",
                    format!("{:?}", stock.construction),
                );
                category(
                    "window_pleat_lift",
                    if stock.cordless {
                        "Cordless"
                    } else {
                        "CordLoop"
                    }
                    .into(),
                );
                numeric.push("window_pleat_count", l.count as f64)?;
                numeric.push("window_pleat_face_length_m", l.face_m as f64)?;
                numeric.push("window_pleat_installed_pitch_m", (2. * l.rise) as f64)?;
                numeric.push("window_pleat_crest_depth_m", l.depth as f64)?;
            }
            if let Some((p, stock)) = vertical {
                let vanes = stock.layout(p, o, f.shade_tilt, draw);
                numeric.push("window_treatment_slat_pitch_m", vanes.pitch as f64)?;
                numeric.push("window_vane_packed_pitch_m", vanes.packed as f64)?;
                numeric.push("window_vane_count", vanes.count as f64)?;
                numeric.push("window_vane_tilt_degrees", vanes.angle.to_degrees() as f64)?;
            }
            if f.shade != Shade::None {
                min_draw = min_draw.min(draw);
                max_draw = max_draw.max(draw);
                numeric.push("exterior_opening_shade_coverage", draw as f64)?;
            }
            if let Some(glazing) = &o.glazing {
                category("exterior_glazing_treatment", glazing.treatment.to_string());
                numeric.push("exterior_glass_thickness_m", glazing.thickness_m as f64)?;
                numeric.push("exterior_glass_privacy_pitch_m", glazing.pitch_m as f64)?;
                numeric.push("exterior_sash_requested_opening", glazing.opening as f64)?;
                numeric.push(
                    "exterior_sash_realized_opening",
                    panes.iter().map(|p| p.opening_fraction).fold(0., f32::max) as f64,
                )?;
            }
            let size = o.max - o.min;
            let full = o.full_height(height);
            full_height += usize::from(full);
            openings += 1;
            for (key, value) in [
                ("exterior_opening_width_m", size.x),
                ("exterior_opening_height_m", size.y),
                ("exterior_opening_aspect", size.x / size.y),
                ("exterior_sill_height_m", o.min.y),
                ("exterior_head_clearance_m", height - o.max.y),
                ("exterior_columns", o.columns as f32),
                ("exterior_crossbars", o.rows().len() as f32),
                ("exterior_panes", panes.len() as f32),
                ("exterior_operating_panes", operating as f32),
                ("exterior_open_panes", open as f32),
            ] {
                numeric.push(key, value as f64)?;
            }
            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{:?},{},{},{},{:?},{},{},{},{},{},{fixed},{operating},{open},{}",
                scene.seed,
                side,
                i,
                o.min.x,
                o.min.y,
                o.max.x,
                o.max.y,
                size.x,
                size.y,
                height - o.max.y,
                o.columns,
                o.transom,
                o.operable,
                f.frame,
                f.frame_width,
                f.frame_depth,
                f.recess,
                f.shade,
                draw,
                full,
                o.rows().len(),
                o.rows()
                    .iter()
                    .map(f32::to_string)
                    .collect::<Vec<_>>()
                    .join(";"),
                f.shade_coverage,
                panes
                    .iter()
                    .map(|p| format!("{:?}", p.hinge))
                    .collect::<Vec<_>>()
                    .join(";")
            )?;
        }
        if f.shade != Shade::None && f.openings.len() > 1 {
            numeric.push(
                "exterior_bank_shade_coverage_range",
                (max_draw - min_draw) as f64,
            )?;
        }
    }
    category("exterior_full_height_room", (full_height > 0).to_string());
    numeric.push("exterior_openings_per_scene", openings as f64)?;
    numeric.push(
        "exterior_full_height_openings_per_scene",
        full_height as f64,
    )?;
    Ok(())
}
