//! Shared, bounded lamp components. Hollow shades, sockets and light sources
//! remain separate material roles; profiles use metres, not stretched textures.
use super::super::geometry::{ProfileJoints, RevolvedOutline};
use super::*;
mod pleats;

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

pub(super) fn wall_thickness(radius: f32) -> f32 {
    (radius * 0.018).clamp(0.001, 0.004)
}

pub(super) fn upper_radius(p: &ShadeProgram, lower: f32, taper: f32) -> f32 {
    lower
        * if matches!(p.shape, 1 | 3) {
            taper.clamp(0.88, 1.)
        } else {
            taper
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
    ShadeStock::Round.build(a, label, p, center, radius, height, taper);
}

/// One cutting pattern for the outer skin, lining and both rolled hems.
pub(super) enum ShadeStock<'a> {
    Round,
    Cut {
        outline: &'a RevolvedOutline,
        rotation: Quat,
        lining: Surface,
    },
}

impl ShadeStock<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build(
        &self,
        a: &mut Assembly,
        label: &str,
        p: &ShadeProgram,
        center: Vec3,
        radius: f32,
        height: f32,
        taper: f32,
    ) {
        let frame = Transform::from_translation(center);
        let cut = |g: &mut Geometry, profile: &[(f32, f32)]| match self {
            Self::Round => g.lathe(profile, 32, frame),
            Self::Cut {
                outline, rotation, ..
            } => outline.sweep(
                g,
                profile,
                ProfileJoints::Creased,
                frame.with_rotation(*rotation),
            ),
        };
        let upper = upper_radius(p, radius, taper);
        let wall = wall_thickness(radius);
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
            pleats::build(
                a.part(Surface::FabricAlt, label),
                center,
                radius,
                upper,
                height,
                wall,
                p.pleats,
            );
        } else {
            cut(a.part(p.finish, label), &profile);
            profile.reverse();
            for (r, _) in &mut profile {
                *r -= wall;
            }
            let lining = match self {
                Self::Cut { lining, .. } => *lining,
                Self::Round => p.finish,
            };
            cut(a.part(lining, label), &profile);
        }
        for (r, y) in [(radius, 0.), (upper, height)] {
            // Rolled rims sit inside the declared shade extent.
            let low = if y == 0. { y } else { y - wall };
            let inner = shade_opening(p, radius, r);
            let trim = a.part(p.trim, label);
            if p.shape < 3 {
                // The solid shade skins terminate on the trim. Exclude the buried
                // annulus from its bearing face; otherwise both parts own that face.
                let edge = r - wall;
                if y == 0. {
                    cut(trim, &[(inner, low), (r, low), (r, low + wall)]);
                    cut(
                        trim,
                        &[(edge, low + wall), (inner, low + wall), (inner, low)],
                    );
                } else {
                    cut(trim, &[(inner, low), (edge, low)]);
                    cut(
                        trim,
                        &[(r, low), (r, low + wall), (inner, low + wall), (inner, low)],
                    );
                }
            } else {
                cut(
                    trim,
                    &[
                        (inner, low),
                        (r, low),
                        (r, low + wall),
                        (inner, low + wall),
                        (inner, low),
                    ],
                );
            }
        }
    }
}

pub(super) fn shade_opening(p: &ShadeProgram, lower: f32, rim: f32) -> f32 {
    let wall = wall_thickness(lower);
    rim - if p.shape == 3 {
        rim * 0.065 + wall
    } else {
        wall * 2.
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
