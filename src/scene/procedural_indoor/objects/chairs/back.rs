//! Shaped, closed back shells and mechanically connected open-back constructions.
use super::*;

pub(super) fn shell(
    g: &mut super::super::Geometry,
    p: &ChairProgram,
    width: f32,
    height: f32,
    profile: [f32; 6],
    tf: Transform,
) {
    let first = g.positions.len();
    g.chair_back_contour(width, height, profile, Transform::IDENTITY);
    for i in first..g.positions.len() {
        let v = Vec3::from_array(g.positions[i]);
        let t = (v.y / height).clamp(0., 1.);
        let x = v.x / (width * 0.5);
        let wave = (PI * t).sin();
        let f = 1. - p.back_waist * wave * wave;
        let active = if v.y > 0. && v.y < height { 1. } else { 0. };
        let dx_dy = -v.x * p.back_waist * PI * (2. * PI * t).sin() / height * active;
        let dy_dx = -p.crest_rounding * height * 4. * x.powi(3) / (width * 0.5) * t.powi(4);
        let dy_dy = 1. - 4. * p.crest_rounding * x.powi(4) * t.powi(3) * active;
        let det = f * dy_dy - dx_dy * dy_dx;
        let n = Vec3::from_array(g.normals[i]);
        // Inverse-transpose of the deformation Jacobian, including closed rims.
        let n = Vec3::new(
            (dy_dy * n.x - dy_dx * n.y) / det,
            (-dx_dy * n.x + f * n.y) / det,
            n.z,
        )
        .normalize();
        let shaped = Vec3::new(
            v.x * f,
            v.y - p.crest_rounding * height * x.powi(4) * t.powi(4),
            v.z,
        );
        g.positions[i] = tf.transform_point(shaped).to_array();
        g.normals[i] = (tf.rotation * n).to_array();
    }
}

pub(super) fn segmented(
    a: &mut Assembly,
    p: &ChairProgram,
    width: f32,
    height: f32,
    tf: Transform,
) {
    // Continuous rails carry the separated panels. Every panel's backing touches
    // both rails; the openings are geometry in all annotation modes.
    for side in [-1., 1.] {
        a.part(p.frame_surface, "chair").rod(
            tf.transform_point(Vec3::new(side * width * 0.31, 0., 0.020)),
            tf.transform_point(Vec3::new(
                side * width * 0.31,
                height - 0.025,
                height * 0.13 + 0.020,
            )),
            0.012,
        );
    }
    let count = if p.back_construction == 3 {
        3
    } else if p.back_construction == 4 {
        2
    } else {
        (height / p.slat_pitch_m).round().clamp(3., 9.) as usize
    };
    let gap = if count == 2 {
        height * p.back_aperture
    } else {
        height / count as f32 * 0.25
    };
    let panel_height = (height - gap * (count - 1) as f32) / count as f32;
    for i in 0..count {
        let y = i as f32 * (panel_height + gap);
        let w = width * (1. - p.back_waist * (PI * (y + panel_height * 0.5) / height).sin());
        shell(
            a.part(p.back_surface, "chair"),
            p,
            w,
            panel_height,
            [
                p.curvature * 0.45,
                p.taper * 0.35,
                0.13,
                p.shell_thickness,
                0.,
                p.shoulder_flare * 0.25,
            ],
            tf.with_translation(tf.transform_point(Vec3::new(0., y, y * 0.13))),
        );
    }
}

/// Split ergonomic lobes, an open lumbar aperture, upholstered ribs, and
/// fanned vertical slats. Every element is connected to a carrying frame.
pub(super) fn constructed(
    a: &mut Assembly,
    p: &ChairProgram,
    width: f32,
    height: f32,
    tf: Transform,
) {
    let rail = |x, y| tf.transform_point(Vec3::new(x, y, y * 0.13 + 0.024));
    for side in [-1., 1.] {
        a.part(p.frame_surface, "chair").rod(
            rail(side * width * 0.32, 0.),
            rail(side * width * 0.32, height - 0.018),
            0.012,
        );
    }
    for y in [0.025, height - 0.028] {
        a.part(p.frame_surface, "chair")
            .rod(rail(-width * 0.45, y), rail(width * 0.45, y), 0.012);
    }
    let mut panel = |x: f32, y: f32, w: f32, h: f32, curve: f32| {
        // Inner offsets must fit the local bend radius. A full-back thickness
        // cannot be reused on a 30 mm slat without folding its inner surface.
        let thickness = p.shell_thickness.min(w * 0.10).min(h * 0.12);
        let curve = curve.min(w * 0.14);
        shell(
            a.part(p.back_surface, "chair"),
            p,
            w,
            h,
            [
                curve,
                p.taper * 0.25,
                0.13,
                thickness,
                (p.lumbar * 0.3).min(h * 0.035),
                p.shoulder_flare * 0.3,
            ],
            tf.with_translation(tf.transform_point(Vec3::new(x, y, y * 0.13))),
        );
        // Short brackets terminate inside the panel and on the continuous
        // vertical rails, including the offset split-lobe constructions.
        for side in [-1., 1.] {
            if x.abs() > 1e-5 && x.signum() != side {
                continue;
            }
            let mid = y + h * 0.5;
            a.part(p.frame_surface, "chair").rod(
                rail(side * width * 0.32, mid),
                tf.transform_point(Vec3::new(
                    x,
                    mid,
                    mid * 0.13 + curve - (p.lumbar * 0.3).min(h * 0.035) + thickness * 0.5,
                )),
                0.009,
            );
        }
    };
    match p.back_construction {
        6 => {
            let gap = width * p.back_aperture;
            for side in [-1., 1.] {
                panel(
                    side * (width + gap) * 0.25,
                    0.,
                    (width - gap) * 0.5,
                    height,
                    p.curvature * 0.45,
                );
            }
        }
        7 => {
            let aperture = height * (0.18 + p.back_aperture);
            let lower = height * 0.22;
            panel(0., 0., width, lower, p.curvature);
            panel(
                0.,
                lower + aperture,
                width,
                height - lower - aperture,
                p.curvature,
            );
        }
        8 => {
            let count = (height / (p.slat_pitch_m * 1.7)).round().clamp(3., 6.) as usize;
            let pitch = height / count as f32;
            for i in 0..count {
                panel(
                    0.,
                    i as f32 * pitch,
                    width,
                    pitch * 0.83,
                    p.curvature * 0.75,
                );
            }
        }
        _ => {
            let count = (width / p.slat_pitch_m).round().clamp(4., 8.) as usize;
            let pitch = width / count as f32;
            for i in 0..count {
                let x = (i as f32 - (count - 1) as f32 * 0.5) * pitch;
                // Broad formed slats carry real gaps through the back.
                panel(x, 0.014, pitch * 0.68, height - 0.028, p.curvature * 0.25);
            }
        }
    }
}
