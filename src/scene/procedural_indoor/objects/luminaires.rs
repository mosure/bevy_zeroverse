//! Shared, bounded lamp components. Hollow shades, sockets and light sources
//! remain separate material roles; profiles use metres, not stretched textures.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ShadeProgram {
    /// Cone, drum, bell, pleated drum, or open wire cage.
    pub shape: u8,
    pub finish: Surface,
    pub trim: Surface,
    pub shoulder: f32,
    pub pleats: u32,
    pub bulb: u8,
}

pub fn parameters(seed: u64) -> ShadeProgram {
    let mut r = stream(seed, 7590);
    ShadeProgram {
        shape: r.random_range(0..5),
        finish: [
            Surface::FabricAlt,
            Surface::Metal,
            Surface::Ceramic,
            Surface::WoodEdge,
        ][r.random_range(0..4)],
        trim: [Surface::Chrome, Surface::Metal, Surface::WoodEdge][r.random_range(0..3)],
        shoulder: r.random_range(0.35..0.72),
        pleats: r.random_range(18..=36),
        bulb: r.random_range(0..4),
    }
}

/// A socket is below the luminous envelope. `center` is also the light's
/// photometric center; all four envelopes have the same exact center.
pub(super) fn bulb(a: &mut Assembly, label: &str, p: &ShadeProgram, center: Vec3, radius: f32) {
    let radii = bulb_radii(p, radius);
    a.part(Surface::Light, label)
        .ellipsoid(radii, Transform::from_translation(center));
    let base = center - Vec3::Y * (radii.y + radius * 0.20);
    a.part(Surface::Ceramic, label).cylinder(
        radius * 0.36,
        radius * 0.44,
        Transform::from_translation(base),
    );
    // Separate raised screw threads; no coincident cylinder caps.
    for i in 0..3 {
        a.part(Surface::Chrome, label).cylinder(
            radius * 0.39,
            radius * 0.045,
            Transform::from_translation(base + Vec3::Y * radius * (i as f32 - 1.) * 0.11),
        );
    }
}

/// Open bottom/top, with a real inner surface and rolled rims. The cage has
/// bent ribs rather than an opaque silhouette pretending to be wire.
#[allow(clippy::too_many_arguments)]
pub(super) fn shade(
    a: &mut Assembly,
    label: &str,
    p: &ShadeProgram,
    center: Vec3,
    radius: f32,
    height: f32,
    taper: f32,
) {
    let upper = radius
        * if matches!(p.shape, 1 | 3) {
            taper.clamp(0.88, 1.)
        } else {
            taper
        };
    let wall = (radius * 0.018).clamp(0.001, 0.004);
    let mut profile = if p.shape == 2 {
        vec![
            (radius, 0.),
            (radius * 0.80, height * 0.18),
            ((upper * 1.05).min(radius.max(upper)), height * p.shoulder),
            (upper, height),
        ]
    } else {
        vec![(radius, 0.), (upper, height)]
    };
    for (_, y) in &mut profile {
        *y = wall + *y / height * (height - 2. * wall);
    }
    if p.shape == 4 {
        for i in 0..12 {
            let v = Vec3::new(
                (TAU * i as f32 / 12.).cos(),
                0.,
                (TAU * i as f32 / 12.).sin(),
            );
            let points = [
                center + v * (radius - wall) + Vec3::Y * wall,
                center + v * radius * 0.91 + Vec3::Y * height * 0.35,
                center + v * (upper - wall) + Vec3::Y * (height - wall),
            ];
            a.part(p.trim, label).tube(&points, wall * 0.5, 6);
        }
    } else if p.shape == 3 {
        // Alternating radii create real folded cloth. Flat normals at each
        // crease preserve the pleat silhouette without a shader illusion.
        let g = a.part(Surface::FabricAlt, label);
        let n = p.pleats * 2;
        for i in 0..n {
            let point = |j: u32, y: f32, inside: bool| {
                let t = TAU * j as f32 / n as f32;
                let r = radius + (upper - radius) * (y / height);
                let r = r * if j.is_multiple_of(2) { 1. } else { 0.945 }
                    - if inside { wall } else { 0. };
                center
                    + Vec3::new(
                        r * t.cos(),
                        wall + y / height * (height - 2. * wall),
                        r * t.sin(),
                    )
            };
            for inside in [false, true] {
                let q = [
                    point(i, 0., inside),
                    point(i, height, inside),
                    point(i + 1, height, inside),
                    point(i + 1, 0., inside),
                ];
                let normal = (q[1] - q[0]).cross(q[2] - q[0]) * if inside { -1. } else { 1. };
                for t in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
                    super::super::envelope::construction::triangle(g, t, normal);
                }
            }
        }
    } else {
        let mut closed = profile.clone();
        closed.extend(profile.iter().rev().map(|&(r, y)| (r - wall, y)));
        closed.push(profile[0]);
        a.part(p.finish, label)
            .lathe(&closed, 32, Transform::from_translation(center));
    }
    for (r, y) in [(radius, 0.), (upper, height)] {
        // Rolled rims sit inside the declared shade extent.
        let low = if y == 0. { y } else { y - wall };
        let inner = r - if p.shape == 3 {
            r * 0.065 + wall
        } else {
            wall * 2.
        };
        a.part(p.trim, label).lathe(
            &[
                (inner, low),
                (r, low),
                (r, low + wall),
                (inner, low + wall),
                (inner, low),
            ],
            32,
            Transform::from_translation(center),
        );
    }
}

pub(super) fn bulb_radii(p: &ShadeProgram, radius: f32) -> Vec3 {
    match p.bulb {
        0 => Vec3::splat(radius),
        1 => Vec3::new(radius * 0.67, radius * 1.35, radius * 0.67),
        2 => Vec3::new(radius * 0.48, radius * 1.65, radius * 0.48),
        _ => Vec3::new(radius, radius * 0.55, radius),
    }
}
