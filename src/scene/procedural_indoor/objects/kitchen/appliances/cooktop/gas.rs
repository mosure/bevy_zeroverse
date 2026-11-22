use super::*;
use crate::scene::procedural_indoor::materials::cooktop::IRON_LABEL;

pub(super) fn build(
    a: &mut Assembly,
    s: Vec3,
    base: Vec3,
    plan: &CooktopProgram,
    grate: &GasGrate,
) {
    a.box_part(
        Surface::Metal,
        "stove",
        base + Vec3::Y * (s.y * 0.12),
        Vec3::new(s.x, s.y * 0.24, s.z),
        0.005,
    );
    for burner in &plan.burners {
        let deck = base + Vec3::new(burner.center.x, s.y * 0.24, burner.center.y);
        let r = burner.radius_m;
        let height = s.y * grate.height_fraction;
        // Burner head, dark enamel cap and radial pot supports are separate
        // pieces. The cast fingers have feet on the deck and a common pot plane.
        a.part(Surface::Chrome, "stove").lathe(
            &[
                (r * 0.35, 0.),
                (r * 0.44, height * 0.14),
                (r * 0.44, height * 0.42),
                (r * 0.38, height * 0.48),
            ],
            24,
            Transform::from_translation(deck),
        );
        a.part(Surface::Metal, IRON_LABEL).lathe(
            &[
                (0., height * 0.44),
                (r * 0.45, height * 0.44),
                (r * 0.49, height * 0.50),
                (r * 0.46, height * 0.58),
                (0., height * 0.59),
            ],
            24,
            Transform::from_translation(deck),
        );
        // A removable grate is one casting. Join every foot below the cap
        // with an open frame, leaving drainage and cleaning space underneath.
        let outer = r * 1.23;
        let inner = outer - r * grate.width_fraction * 1.12;
        let low = height * 0.10;
        let high = height * 0.34;
        let bevel = ((outer - inner) * 0.16).min((high - low) * 0.22);
        a.part(Surface::Metal, IRON_LABEL).lathe(
            &[
                (outer - bevel, low),
                (outer, low + bevel),
                (outer, high - bevel),
                (outer - bevel, high),
                (inner + bevel, high),
                (inner, high - bevel),
                (inner, low + bevel),
                (inner + bevel, low),
                (outer - bevel, low),
            ],
            grate.frame_segments,
            Transform::from_translation(deck)
                .with_rotation(Quat::from_rotation_y(grate.phase_radians)),
        );
        let inner = r * grate.tip_fraction;
        let outer = r * 1.24;
        let profile = [
            Vec2::new(inner, -height),
            Vec2::new(outer - r * 0.07, -height),
            Vec2::new(outer, -height + r * 0.07),
            Vec2::new(outer, 0.),
            Vec2::new(outer - r * 0.16, 0.),
            Vec2::new(inner, -height * 0.72),
        ];
        let center = profile.iter().sum::<Vec2>() / profile.len() as f32;
        let outline = profile.map(|v| v - center);
        for i in 0..grate.fingers {
            let q =
                Quat::from_rotation_y(grate.phase_radians + TAU * i as f32 / grate.fingers as f32)
                    * Quat::from_rotation_x(FRAC_PI_2);
            let tf = Transform::from_translation(deck + q * Vec3::new(center.x, 0., center.y))
                .with_rotation(q);
            a.part(Surface::Metal, IRON_LABEL).profile_slab(
                &outline,
                r * grate.width_fraction,
                0.001,
                tf,
            );
        }
    }
}
