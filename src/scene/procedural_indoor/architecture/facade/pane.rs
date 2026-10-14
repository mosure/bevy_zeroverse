//! One hinged sash owns its pane, seals, frame and handle. Static open states
//! leave a real gap; the renderer and every annotation see the same geometry.
use super::*;

pub(super) fn build(
    a: &mut Assembly,
    tf: Transform,
    f: &Facade,
    opening: &WindowOpening,
    min: Vec2,
    max: Vec2,
) {
    let Some(program) = &opening.glazing else {
        let c = (min + max) * 0.5;
        a.part(Surface::Glass, "window").cuboid(
            Vec3::new(max.x - min.x + 0.004, max.y - min.y + 0.004, 0.008),
            0.,
            Transform::from_translation(tf.transform_point(Vec3::new(c.x, c.y, -f.recess - 0.004)))
                .with_rotation(tf.rotation),
        );
        return;
    };
    let sash = if opening.operable {
        0.021_f32.min((max - min).min_element() * 0.09)
    } else {
        0.004
    };
    let lo = min + Vec2::splat(sash);
    let hi = max - Vec2::splat(sash);
    let height = max.y - min.y;
    let angle = if opening.operable && f.allows_open_sash() {
        program.opening * (0.14 / height).min(0.20).asin()
    } else {
        0.
    };
    let hinge =
        Transform::from_xyz(0., min.y, -f.recess).with_rotation(Quat::from_rotation_x(angle));
    let local = |p: Vec3| hinge.transform_point(p - Vec3::Y * min.y);
    let rotation = tf.rotation * hinge.rotation;
    let mut block = |surface, label: &str, p: Vec3, size: Vec3, bevel| {
        a.part(surface, label).cuboid(
            size,
            bevel,
            Transform::from_translation(tf.transform_point(local(p))).with_rotation(rotation),
        );
    };
    // Real sash profiles and captured rubber edges, with noncoincident joints.
    for x in [min.x + sash * 0.5, max.x - sash * 0.5] {
        block(
            f.frame,
            "window",
            Vec3::new(x, (min.y + max.y) * 0.5, 0.006),
            Vec3::new(sash, height - 0.004, 0.026),
            0.001,
        );
    }
    for y in [min.y + sash * 0.5, max.y - sash * 0.5] {
        block(
            f.frame,
            "window",
            Vec3::new((min.x + max.x) * 0.5, y, 0.006),
            Vec3::new(max.x - min.x - 2. * sash - 0.002, sash, 0.026),
            0.001,
        );
    }
    // Visible compressible beads capture the pane edge. Only their back face
    // meets the glazing; their front face is 2 mm forward, avoiding z fighting.
    let bead = 0.002_f32.min(sash * 0.3);
    for x in [lo.x + bead * 0.5, hi.x - bead * 0.5] {
        block(
            Surface::Rubber,
            "window",
            Vec3::new(x, (lo.y + hi.y) * 0.5, 0.001),
            Vec3::new(bead, hi.y - lo.y, 0.002),
            0.,
        );
    }
    for y in [lo.y + bead * 0.5, hi.y - bead * 0.5] {
        block(
            Surface::Rubber,
            "window",
            Vec3::new((lo.x + hi.x) * 0.5, y, 0.001),
            Vec3::new(hi.x - lo.x - 2. * bead, bead, 0.002),
            0.,
        );
    }
    if opening.operable {
        block(
            Surface::Chrome,
            "window",
            Vec3::new(max.x - sash * 0.5, (min.y + max.y) * 0.5, 0.032),
            Vec3::new(0.009, 0.08_f32.min(height * 0.35), 0.022),
            0.002,
        );
    }
    if program.treatment == 6 {
        // Frit is an opaque ceramic deposit with actual ownership. Through-glass
        // annotations retain it; frosting is instead a roughness-only treatment.
        let mut y = (lo.y / program.pitch_m).ceil() * program.pitch_m;
        while y < hi.y {
            let bottom = y.max(lo.y + bead + 0.0005).max(program.band_m.x);
            let top = (y + program.pitch_m * program.duty)
                .min(hi.y - bead - 0.0005)
                .min(program.band_m.y);
            if top - bottom > 0.001 {
                block(
                    Surface::Paint,
                    "window#frit",
                    Vec3::new((lo.x + hi.x) * 0.5, (bottom + top) * 0.5, 0.0008),
                    Vec3::new(hi.x - lo.x - 2. * (bead + 0.0005), top - bottom, 0.0006),
                    0.,
                );
            }
            y += program.pitch_m;
        }
    }
    let mut g = super::super::super::geometry::Geometry::default();
    let t = program.thickness_m;
    g.cuboid(
        Vec3::new(hi.x - lo.x, hi.y - lo.y, t),
        0.,
        Transform::from_translation(Vec3::new(
            (lo.x + hi.x) * 0.5,
            (lo.y + hi.y) * 0.5,
            -t * 0.5,
        )),
    );
    for ((position, normal), uv) in g.positions.iter_mut().zip(&mut g.normals).zip(&mut g.uvs) {
        let p = Vec3::from_array(*position);
        *uv = ((p.truncate() - opening.min) / (opening.max - opening.min)).to_array();
        *position = tf.transform_point(local(p)).to_array();
        *normal = (rotation * Vec3::from_array(*normal)).to_array();
    }
    a.part(
        Surface::Glass,
        &format!("window#glazing{}", program.identity),
    )
    .append(g);
}
