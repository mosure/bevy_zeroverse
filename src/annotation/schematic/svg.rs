use super::*;
use std::fmt::Write;
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
fn text(s: &mut String, x: f32, y: f32, size: u32, value: &str, color: &str) {
    let _ = write!(
        s,
        r##"<text x="{x:.2}" y="{y:.2}" font-size="{size}" fill="{color}">{}</text>"##,
        escape(value)
    );
}
fn points(p: &[[f32; 3]], project: &Projection) -> Result<String, String> {
    if p.iter().flatten().any(|x| !x.is_finite()) {
        return Err("schematic contains a non-finite point".into());
    }
    Ok(p.iter()
        .map(|p| {
            let q = project.project(*p);
            format!("{:.2},{:.2}", q[0], q[1])
        })
        .collect::<Vec<_>>()
        .join(" "))
}

fn camera(
    s: &mut String,
    c: &Camera,
    p: &Projection,
    options: &RenderOptions,
    index: usize,
    pred: bool,
) -> Result<(), String> {
    let gizmo = camera_presentation(c, p, options)?;
    let index = crate::camera::gizmo::color_index(&c.label, index);
    let [r, g, b] = crate::camera::gizmo::rgb(index);
    let rgb = format!("#{r:02x}{g:02x}{b:02x}");
    let dash = if pred {
        r##" stroke-dasharray="6 4""##
    } else {
        ""
    };
    let _ = write!(
        s,
        r##"<g data-lens="{}"><title>{}</title>"##,
        gizmo.lens_name(),
        escape(&gizmo.label)
    );
    if options.camera_paths && c.path.len() > 1 {
        let _ = write!(
            s,
            r##"<polyline points="{}" fill="none" stroke="{rgb}" opacity="0.65" stroke-width="1.5"{dash}/>"##,
            points(&c.path, p)?
        );
    }
    let _ = write!(
        s,
        r##"<polygon points="{}" fill="{rgb}" fill-opacity="0.10" stroke="{rgb}" stroke-width="1.4"{dash}/>"##,
        points(&gizmo.footprint, p)?
    );
    let _ = write!(
        s,
        r##"<polyline points="{}" fill="none" stroke="{rgb}" stroke-width="2.5"{dash}/>"##,
        points(&[gizmo.origin, gizmo.axis_end], p)?
    );
    let q = p.project(gizmo.origin);
    let radius = gizmo.marker_radius();
    let fill = if gizmo.fisheye { "white" } else { &rgb };
    let stroke = if gizmo.fisheye { &rgb } else { "white" };
    let _ = write!(
        s,
        r##"<circle cx="{}" cy="{}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>"##,
        q[0], q[1]
    );
    if gizmo.fisheye {
        let _ = write!(
            s,
            r##"<circle cx="{}" cy="{}" r="2.5" fill="{rgb}"/>"##,
            q[0], q[1]
        );
    }
    text(
        s,
        q[0] + radius + 3.,
        q[1] - radius - 3.,
        13,
        &gizmo.caption,
        &rgb,
    );
    s.push_str("</g>");
    Ok(())
}
pub(super) fn render(
    plan: &Schematic,
    options: &RenderOptions,
    predictions: &Overlay,
) -> Result<String, String> {
    let p = plan.projection(options)?;
    let (w, h) = (options.width, options.height);
    let mut s = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="#f3f5f1"/><g font-family="DejaVu Sans"><defs><clipPath id="map"><rect x="12" y="42" width="{}" height="{}"/></clipPath></defs>"##,
        w - 24,
        h - 88
    );
    text(
        &mut s,
        24.,
        27.,
        16,
        &format!("ROOM {}  /  TOP-DOWN", plan.seed),
        "#203337",
    );
    s.push_str(r##"<g clip-path="url(#map)">"##);
    for (pred, shapes) in [(false, &plan.footprints), (true, &predictions.footprints)] {
        for f in shapes {
            let (fill, stroke, width) = match f.kind {
                FootprintKind::Floor => ("#ffffff", "#293f43", 3.),
                FootprintKind::Level => ("#e9e7dc", "#b7b3a4", 0.7),
                FootprintKind::Mezzanine => ("#dce7f0", "#6287a1", 1.2),
                FootprintKind::Wall => ("#455b60", "#455b60", 4.),
                FootprintKind::Window => ("none", "#4eabc4", 4.),
                FootprintKind::Door => ("none", "#ffffff", 5.),
                FootprintKind::Furniture => ("#d7c4a8", "#8e7e64", 1.),
                FootprintKind::Prop => ("#e3dfd6", "#9a978e", 0.65),
                FootprintKind::Plant => ("#b1c99c", "#648451", 1.),
            };
            let stroke = if pred { "#bb297c" } else { stroke };
            let tag = if f.points.len() == 2 {
                "polyline"
            } else {
                "polygon"
            };
            let _ = write!(
                s,
                r##"<{tag} points="{}" fill="{fill}" fill-opacity="0.85" stroke="{stroke}" stroke-width="{width}" {}><title>{}</title></{tag}>"##,
                points(&f.points, &p)?,
                if pred {
                    r##"stroke-dasharray="6 4""##
                } else {
                    ""
                },
                escape(&f.label)
            );
            if options.labels
                && (f.kind == FootprintKind::Furniture || f.kind == FootprintKind::Mezzanine)
                && !f.points.is_empty()
            {
                let center = f
                    .points
                    .iter()
                    .fold(Vec3::ZERO, |a, x| a + Vec3::from_array(*x))
                    / f.points.len() as f32;
                let q = p.project(center.to_array());
                text(&mut s, q[0] + 3., q[1] - 3., 9, &f.label, "#435457");
            }
        }
    }
    for (pred, poses) in [(false, &plan.poses), (true, &predictions.poses)] {
        for pose in poses {
            if pose.parents.len() != pose.joints.len() {
                return Err("schematic joint/parent counts differ".into());
            }
            let rgb = if pred { "#bb297c" } else { "#a64d37" };
            let _ = points(&pose.joints, &p)?;
            for (i, parent) in pose.parents.iter().enumerate() {
                if *parent < -1 || *parent >= pose.joints.len() as i64 || *parent == i as i64 {
                    return Err("invalid schematic joint parent".into());
                }
                if *parent >= 0 {
                    let _ = write!(
                        s,
                        r##"<polyline points="{}" fill="none" stroke="{rgb}" stroke-width="2" {}/>"##,
                        points(&[pose.joints[i], pose.joints[*parent as usize]], &p)?,
                        if pred {
                            r##"stroke-dasharray="4 3""##
                        } else {
                            ""
                        }
                    );
                }
                let q = p.project(pose.joints[i]);
                let _ = write!(
                    s,
                    r##"<circle cx="{}" cy="{}" r="2.3" fill="{rgb}"/>"##,
                    q[0], q[1]
                );
            }
            if let Some(root) = pose.joints.first() {
                let q = p.project(*root);
                text(&mut s, q[0] + 7., q[1] + 15., 11, &pose.label, rgb);
            }
        }
    }
    for (pred, cameras) in [(false, &plan.cameras), (true, &predictions.cameras)] {
        for (i, c) in cameras.iter().enumerate() {
            camera(&mut s, c, &p, options, i, pred)?;
        }
    }
    s.push_str("</g>");
    let y = h as f32 - 26.;
    let length = p.pixels_per_metre;
    let _ = write!(
        s,
        r##"<path d="M24 {y} h{length}" stroke="#203337" stroke-width="3"/>"##
    );
    text(&mut s, 24., y - 7., 11, "1 m", "#203337");
    text(
        &mut s,
        24. + length + 18.,
        y + 4.,
        11,
        "+X right / +Z down",
        "#52666b",
    );
    if !predictions.cameras.is_empty()
        || !predictions.poses.is_empty()
        || !predictions.footprints.is_empty()
    {
        text(
            &mut s,
            w as f32 - 180.,
            y + 4.,
            11,
            "Dashed: predictions",
            "#bb297c",
        );
    }
    s.push_str("</g></svg>");
    Ok(s)
}
pub(super) fn raster(svg: &str) -> Result<Vec<u8>, String> {
    static OPTIONS: std::sync::OnceLock<resvg::usvg::Options<'static>> = std::sync::OnceLock::new();
    let options = OPTIONS.get_or_init(|| {
        let mut o = resvg::usvg::Options::default();
        o.fontdb_mut()
            .load_font_data(include_bytes!("DejaVuSans.ttf").to_vec());
        o
    });
    let tree = resvg::usvg::Tree::from_str(svg, options).map_err(|e| e.to_string())?;
    let size = tree.size().to_int_size();
    let mut pixmap = resvg::tiny_skia::Pixmap::new(size.width(), size.height())
        .ok_or("schematic image allocation failed")?;
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::identity(),
        &mut pixmap.as_mut(),
    );
    Ok(pixmap.take())
}
