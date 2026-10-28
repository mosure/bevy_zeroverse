//! Wall fixtures with real hollow shades and connected, sampled mounting arms.
//! Construction and emitter queries consume the same dimensioned program.
use super::super::luminaires;
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct SconceProgram {
    pub plate_span: Vec2,
    pub plate_thickness: f32,
    pub plate_round: bool,
    pub arm_style: u8,
    pub arm_radius: f32,
    pub head_bottom: Vec3,
    pub head_span: Vec2,
    pub head_height: f32,
    pub taper: f32,
    pub shade: luminaires::ShadeProgram,
    pub diffuser_radius: f32,
    pub diffuser_thickness: f32,
    pub emitter: Vec3,
}

pub(crate) fn dimensions(r: &mut impl Rng) -> Vec3 {
    let width = r.random_range(0.12..0.40);
    Vec3::new(
        width,
        (width * r.random_range(0.80_f32..2.5)).clamp(0.18, 0.60),
        (width * r.random_range(0.42_f32..1.0)).clamp(0.08, 0.25),
    )
}

pub fn parameters(o: &IndoorObject) -> SconceProgram {
    let s = o.size;
    let mut r = stream(o.seed, 0x53434f4e43454745);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x53434f4e43454649);
    let mut shade = luminaires::parameters(o.seed);
    shade.finish = if shade.shape == 3 {
        Surface::FabricAlt
    } else {
        [
            Surface::FabricAlt,
            Surface::Metal,
            Surface::Ceramic,
            Surface::WoodEdge,
        ][f.random_range(0..4)]
    };
    shade.trim = [Surface::Chrome, Surface::Metal, Surface::WoodEdge][f.random_range(0..3)];
    let head_span = s.xz() * Vec2::new(r.random_range(0.55..0.94), r.random_range(0.46..0.75));
    let height = s.y * r.random_range(0.25..0.57);
    let bottom = Vec3::new(
        (s.x - head_span.x) * r.random_range(-0.38..0.38),
        (s.y - height) * r.random_range(0.12..0.53),
        s.z * 0.5 - head_span.y * 0.5 - s.z * 0.02,
    );
    let diffuser_thickness = (height * 0.022).min(0.002);
    let radius = head_span.min_element() * r.random_range(0.22..0.35);
    SconceProgram {
        plate_span: s.xy() * Vec2::new(r.random_range(0.24..0.64), r.random_range(0.34..0.79)),
        plate_thickness: s.z * r.random_range(0.045..0.11),
        plate_round: r.random_bool(0.45),
        arm_style: r.random_range(0..3),
        arm_radius: (s.min_element() * r.random_range(0.028..0.046)).min(0.009),
        head_bottom: bottom,
        head_span,
        head_height: height,
        taper: r.random_range(0.48..0.98),
        shade,
        diffuser_radius: radius,
        diffuser_thickness,
        // Just outside the lower diffuser, in the open downlight aperture.
        emitter: bottom - Vec3::Y * 0.002,
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let label = o.kind.class_name();
    let rear = -o.size.z * 0.5;
    let plate_center = Vec3::new(0., o.size.y * 0.5, rear + p.plate_thickness * 0.5);
    if p.plate_round {
        // Cylinder axis is local Y; transform into a vertical oval wall plate.
        a.part(p.shade.trim, label).cylinder(
            0.5,
            p.plate_thickness,
            Transform::from_translation(plate_center)
                .with_rotation(Quat::from_rotation_x(FRAC_PI_2))
                .with_scale(Vec3::new(p.plate_span.x, 1., p.plate_span.y)),
        );
    } else {
        a.box_part(
            p.shade.trim,
            label,
            plate_center,
            Vec3::new(p.plate_span.x, p.plate_span.y, p.plate_thickness),
            p.plate_thickness * 0.24,
        );
    }
    let top = p.head_bottom + Vec3::Y * p.head_height;
    let start = Vec3::new(
        0.,
        plate_center.y,
        rear + p.plate_thickness + p.arm_radius * 0.8,
    );
    let high = top.y.max(start.y) + p.arm_radius * 2.;
    let elbow = Vec3::new(
        if p.arm_style == 2 { top.x } else { 0. },
        high,
        if p.arm_style == 0 {
            top.z
        } else {
            start.z + o.size.z * 0.13
        },
    );
    for (from, to) in [
        (start, elbow),
        (elbow, Vec3::new(top.x, high, top.z)),
        (Vec3::new(top.x, high, top.z), top),
    ] {
        if from.distance_squared(to) > 1e-10 {
            a.part(p.shade.trim, label).rod(from, to, p.arm_radius);
        }
    }
    // Build shared shade geometry in an elliptical local frame. Correct the
    // normals with inverse scale; metric UVs use the mean perimeter stretch.
    let radius = p.head_span.y * 0.5;
    let mut shade = Assembly::default();
    luminaires::shade(
        &mut shade,
        label,
        &p.shade,
        Vec3::ZERO,
        radius,
        p.head_height,
        p.taper,
    );
    let scale = Vec3::new(p.head_span.x / p.head_span.y, 1., 1.);
    for ((surface, l), mut g) in shade.parts {
        for point in &mut g.positions {
            *point = (Vec3::from_array(*point) * scale + p.head_bottom).to_array();
        }
        for n in &mut g.normals {
            *n = (Vec3::from_array(*n) / scale).normalize().to_array();
        }
        for uv in &mut g.uvs {
            uv[0] *= (scale.x + 1.) * 0.5;
        }
        a.part(surface, &l).append(g);
    }
    // Cross supports physically connect the arm/socket to the upper shade rim.
    let taper = if matches!(p.shade.shape, 1 | 3) {
        p.taper.max(0.88)
    } else {
        p.taper
    };
    // The spoke ends must enter the actual rim wall, including when the shade
    // is shallow/elliptical. Oversized spokes stopped short of that inner edge.
    let wall = (radius * 0.018).clamp(0.001, 0.004);
    let rib = p.arm_radius.min(wall * 0.40 * scale.x.min(1.));
    for side in [-1., 1.] {
        a.part(p.shade.trim, label).rod(
            top - Vec3::Y * rib,
            top + Vec3::new(side * (p.head_span.x * 0.5 * taper - rib), -rib, 0.),
            rib,
        );
    }
    a.part(Surface::Light, label).cylinder(
        p.diffuser_radius,
        p.diffuser_thickness,
        Transform::from_translation(p.head_bottom + Vec3::Y * p.diffuser_thickness * 0.5),
    );
    for direction in [
        Vec3::X * (p.head_span.x * 0.5 - rib),
        Vec3::Z * (radius - rib),
    ] {
        a.part(p.shade.trim, label).rod(
            p.head_bottom - direction + Vec3::Y * rib,
            p.head_bottom + direction + Vec3::Y * rib,
            rib,
        );
    }
    let bulb_radius = (p.head_height * 0.14).min(p.head_span.min_element() * 0.17);
    let bulb_center = p.head_bottom + Vec3::Y * p.head_height * 0.52;
    let mut bulb = Assembly::default();
    luminaires::bulb(&mut bulb, label, &p.shade, Vec3::ZERO, bulb_radius);
    super::super::public::append_at(
        a,
        bulb,
        Transform::from_translation(bulb_center).with_rotation(Quat::from_rotation_x(PI)),
    );
    a.part(p.shade.trim, label).rod(
        top,
        bulb_center
            + Vec3::Y * (luminaires::bulb_radii(&p.shade, bulb_radius).y + bulb_radius * 0.25),
        rib,
    );
}

#[cfg(test)]
mod tests;
