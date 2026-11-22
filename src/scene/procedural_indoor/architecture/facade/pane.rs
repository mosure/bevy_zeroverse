//! One hinged sash owns its pane, seals, frame and handle. Static open states
//! leave a real gap; the renderer and every annotation see the same geometry.
use super::*;

use crate::scene::procedural_indoor::materials::glazing::GlazingProgram;

fn edge_widths(f: &Facade, opening: &WindowOpening, span: Vec2) -> (f32, f32) {
    let sash = if opening.operable {
        f.joinery
            .as_ref()
            .map_or(0.021, |j| j.sash_width_m)
            .min(span.min_element() * 0.09)
    } else {
        f.joinery.as_ref().map_or(0.004, |j| j.seal_width_m * 2.)
    };
    let bead = f
        .joinery
        .as_ref()
        .map_or(0.002, |j| j.seal_width_m)
        .min(sash * 0.3);
    (sash, bead)
}

/// Metric strip intervals shared by the ceramic deposit and daylight metering.
fn frit_bands(
    program: &GlazingProgram,
    lo: f32,
    hi: f32,
    bead: f32,
) -> impl Iterator<Item = (f32, f32)> + '_ {
    let start = (lo / program.pitch_m).ceil() * program.pitch_m;
    std::iter::successors(Some(start), |y| Some(*y + program.pitch_m))
        .take_while(move |&y| y < hi)
        .filter_map(move |y| {
            let bottom = y.max(lo + bead + 0.0005).max(program.band_m.x);
            let top = (y + program.pitch_m * program.duty)
                .min(hi - bead - 0.0005)
                .min(program.band_m.y);
            (top - bottom > 0.001).then_some((bottom, top))
        })
}

/// Frontal glass area inside the actually constructed mullions, sash and
/// glazing beads. Rough privacy glazing still passes light; fired frit blocks it.
pub(crate) fn visible_pane_area(f: &Facade, o: &WindowOpening) -> f32 {
    visible_pane_area_with(f, o, |_, _| 0.)
}

/// Subtract an opaque covering's projected area inside each clear rectangle.
/// A strip of frit behind the covering only blocks the light once.
pub(super) fn visible_pane_area_with(
    f: &Facade,
    o: &WindowOpening,
    covered: impl Fn(Vec2, Vec2) -> f32,
) -> f32 {
    f.pane_rectangles(o)
        .into_iter()
        .map(|(lo, hi)| {
            let Some(program) = &o.glazing else {
                return ((hi - lo).element_product() - covered(lo, hi)).max(0.);
            };
            let (sash, bead) = edge_widths(f, o, hi - lo);
            let clear = (hi - lo - 2. * (sash + bead)).max(Vec2::ZERO);
            let inner_lo = lo + Vec2::splat(sash + bead);
            let inner_hi = hi - Vec2::splat(sash + bead);
            let mut blocked = covered(inner_lo, inner_hi);
            if program.treatment == 6 {
                let mut strips = 0.;
                let mut overlap = 0.;
                for (a, b) in frit_bands(program, lo.y + sash, hi.y - sash, bead) {
                    strips += b - a;
                    overlap += covered(
                        Vec2::new(inner_lo.x + 0.0005, a),
                        Vec2::new(inner_hi.x - 0.0005, b),
                    );
                }
                blocked += ((clear.x - 0.001).max(0.) * strips - overlap).max(0.);
            }
            (clear.element_product() - blocked).max(0.)
        })
        .sum()
}

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
    let (sash, bead) = edge_widths(f, opening, max - min);
    let sash_depth = f.joinery.as_ref().map_or(0.026, |j| j.sash_depth_m);
    let frame_label = f.frame_label();
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
            &frame_label,
            Vec3::new(x, (min.y + max.y) * 0.5, 0.006),
            Vec3::new(sash, height - 0.004, sash_depth),
            0.001,
        );
    }
    for y in [min.y + sash * 0.5, max.y - sash * 0.5] {
        block(
            f.frame,
            &frame_label,
            Vec3::new((min.x + max.x) * 0.5, y, 0.006),
            Vec3::new(max.x - min.x - 2. * sash - 0.002, sash, sash_depth),
            0.001,
        );
    }
    // Visible compressible beads capture the pane edge. Only their back face
    // meets the glazing; their front face is 2 mm forward, avoiding z fighting.
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
        for (bottom, top) in frit_bands(program, lo.y, hi.y, bead) {
            block(
                Surface::Paint,
                "window#frit",
                Vec3::new((lo.x + hi.x) * 0.5, (bottom + top) * 0.5, 0.0008),
                Vec3::new(hi.x - lo.x - 2. * (bead + 0.0005), top - bottom, 0.0006),
                0.,
            );
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
