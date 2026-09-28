//! Exterior counts describe built apertures, excluding the neighboring room.
use super::{CoverageReport, IndoorManifest};
use crate::scene::procedural_indoor::{
    architecture::facade::Shade, metrics_sort::NumericCollector,
};
use std::io::{self, Write};

pub(super) fn header(writer: &mut impl Write) -> io::Result<()> {
    writeln!(writer, "seed,side,opening,left_m,bottom_m,right_m,top_m,width_m,height_m,head_clearance_m,columns,transom,operable,frame,frame_width_m,frame_depth_m,recess_m,shade,shade_coverage,full_height")
}

pub(super) fn record(
    scene: &IndoorManifest,
    report: &mut CoverageReport,
    numeric: &mut NumericCollector,
    writer: &mut impl Write,
) -> io::Result<()> {
    let Some(exterior) = &scene.exterior else {
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
    category("exterior_wall_count", exterior.facades.len().to_string());
    category(
        "exterior_sides",
        exterior
            .facades
            .iter()
            .map(|f| format!("{:?}", f.side))
            .collect::<Vec<_>>()
            .join("+"),
    );
    numeric.push(
        "exterior_exposure_probability",
        exterior.exposure_probability as f64,
    )?;
    let mut full_height = 0;
    let mut openings = 0;
    for f in &exterior.facades {
        category("exterior_side", format!("{:?}", f.side));
        category("exterior_frame", format!("{:?}", f.frame));
        category("exterior_shade", format!("{:?}", f.shade));
        let area: f32 = f.openings.iter().map(|o| o.area()).sum();
        for (key, value) in [
            (
                "exterior_opening_area_fraction",
                area / (f.side.span(scene.room_size) * scene.room_size.y),
            ),
            ("exterior_frame_width_m", f.frame_width),
            ("exterior_frame_depth_m", f.frame_depth),
            ("exterior_recess_m", f.recess),
        ] {
            numeric.push(key, value as f64)?;
        }
        if f.shade != Shade::None {
            numeric.push("exterior_shade_coverage", f.shade_coverage as f64)?;
        }
        for (i, o) in f.openings.iter().enumerate() {
            let size = o.max - o.min;
            let full = o.full_height(scene.room_size.y);
            full_height += usize::from(full);
            openings += 1;
            for (key, value) in [
                ("exterior_opening_width_m", size.x),
                ("exterior_opening_height_m", size.y),
                ("exterior_opening_aspect", size.x / size.y),
                ("exterior_sill_height_m", o.min.y),
                ("exterior_head_clearance_m", scene.room_size.y - o.max.y),
                ("exterior_columns", o.columns as f32),
            ] {
                numeric.push(key, value as f64)?;
            }
            writeln!(
                writer,
                "{},{:?},{},{},{},{},{},{},{},{},{},{},{},{:?},{},{},{},{:?},{},{}",
                scene.seed,
                f.side,
                i,
                o.min.x,
                o.min.y,
                o.max.x,
                o.max.y,
                size.x,
                size.y,
                scene.room_size.y - o.max.y,
                o.columns,
                o.transom,
                o.operable,
                f.frame,
                f.frame_width,
                f.frame_depth,
                f.recess,
                f.shade,
                f.shade_coverage,
                full
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
