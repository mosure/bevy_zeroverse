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
