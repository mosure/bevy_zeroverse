//! One hinged sash owns its pane, seals, frame and handle. Static open states
//! leave a real gap; the renderer and every annotation see the same geometry.
use super::*;
mod hardware;

use crate::scene::procedural_indoor::materials::glazing::GlazingProgram;

fn edge_widths(f: &Facade, hinge: SashHinge, span: Vec2) -> (f32, f32) {
    let sash = if hinge != SashHinge::Fixed {
        f.joinery
            .as_ref()
            .map_or(0.021, |j| {
                if j.sash_stock.is_some() {
                    super::sash::fitted_rail_width(j, span)
                } else {
                    j.sash_width_m
                }
            })
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
    // A transom or sill can interrupt a strip that started on the preceding
    // pitch. Anchor the phase in wall metres on both sides of that interruption.
    let first = (lo / program.pitch_m).floor() as i32;
    (first..)
        .map(move |index| index as f32 * program.pitch_m)
        .take_while(move |&y| y < hi)
        .filter_map(move |y| {
            let bottom = y.max(lo + bead + 0.0005).max(program.band_m.x);
            let top = (y + program.pitch_m * program.duty)
                .min(hi - bead - 0.0005)
                .min(program.band_m.y);
            (top - bottom > 0.001).then_some((bottom, top))
        })
}

#[cfg(test)]
#[test]
fn ceramic_strips_continue_above_an_interrupting_transom() {
    let mut glass = GlazingProgram::sample(17, 0, false);
    glass.band_m = Vec2::new(0.5, 2.);
    glass.pitch_m = 0.125;
    glass.duty = 0.5;
    let pieces: Vec<_> = frit_bands(&glass, 1.02, 1.18, 0.002).collect();
    assert_eq!(pieces.len(), 2);
    assert!((pieces[0].0 - 1.0225).abs() < 1e-5);
    assert_eq!(pieces[0].1, 1.0625);
    assert_eq!(pieces[1].0, 1.125);
    assert!((pieces[1].1 - 1.1775).abs() < 1e-5);
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
    let fitted = f.joinery.as_ref().is_some_and(|j| j.sash_stock.is_some());
    f.pane_installations(o)
        .into_iter()
        .map(|pane| {
            let (lo, hi) = pane.bounds(fitted);
            let Some(program) = &o.glazing else {
                return ((hi - lo).element_product() - covered(lo, hi)).max(0.);
            };
            let (sash, bead) = edge_widths(f, pane.hinge, hi - lo);
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
            // An installed treatment keeps fitted leaves shut. An open,
            // unshaded leaf presents less glazed area along the wall normal.
            let projection = if fitted { pane.swing_radians.cos() } else { 1. };
            (clear.element_product() - blocked).max(0.) * projection
        })
        .sum()
}

pub(super) fn build(
    a: &mut Assembly,
    tf: Transform,
    f: &Facade,
    opening: &WindowOpening,
    pane: PaneInstallation,
) {
    let stock = f.joinery.as_ref().and_then(|j| j.sash_stock.as_ref());
    let (min, max) = pane.bounds(stock.is_some());
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
    let (sash, bead) = edge_widths(f, pane.hinge, max - min);
    let sash_depth = f.joinery.as_ref().map_or(0.026, |j| j.sash_depth_m);
    let frame_label = f.frame_label();
    let lo = min + Vec2::splat(sash);
    let hi = max - Vec2::splat(sash);
    let height = max.y - min.y;
    let face = 0.006 + sash_depth * 0.5;
    let rail_back = 0.006 - sash_depth * 0.5;
    let back = rail_back.min(-program.thickness_m);
    // Keep the original order of float operations when replaying archived glass.
    let (hinge, origin) = if stock.is_none() || pane.hinge == SashHinge::Fixed {
        (
            Transform::from_xyz(0., min.y, -f.recess)
                .with_rotation(Quat::from_rotation_x(pane.swing_radians)),
            Vec3::Y * min.y,
        )
    } else {
        (pane.pose(f, face, back), Vec3::ZERO)
    };
    let local = |p: Vec3| hinge.transform_point(p - origin);
    if stock.is_some() && pane.hinge != SashHinge::Fixed {
        let seat = if pane.hinge == SashHinge::TopAwning {
            face + 0.0015
        } else {
            back - 0.0015
        };
        hardware::seals(a, tf, &pane, min, max, seat - f.recess);
        hardware::hinges(a, tf, f.recess, hinge, &pane, [back, rail_back, face]);
    }
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
    if let (Some(stock), Some(handle)) = (stock, pane.handle) {
        hardware::handle(stock, &pane, handle, sash, face, &mut block);
    } else if pane.handle.is_some() {
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
