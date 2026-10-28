//! Original vector escape pictogram; the downward arrow denotes the door below.
use super::*;
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &ExitProgram, size: Vec2) {
    let label = o.kind.class_name();
    let eu = p.legend == ExitLegend::EuropeanPictogram;
    let background = if eu {
        format!("{label}#bulb1")
    } else {
        format!("{label}#finish{}", if p.dark_face { 303 } else { 301 })
    };
    a.box_part(
        if eu { Surface::Light } else { Surface::Plastic },
        &background,
        Vec3::ZERO,
        size.extend(0.0008),
        0.0003,
    );
    let foreground = format!("{label}#bulb0");
    let center = Vec3::Z * 0.001;
    if !eu {
        lettering::build(
            a,
            &foreground,
            "EXIT",
            size * p.legend_fraction,
            center,
            Surface::Light,
        );
        return;
    }
    let mut draw = |points: &[(f32, f32)]| {
        let mut outline: Vec<_> = points
            .iter()
            .map(|&(x, y)| (Vec2::new(x, y) - Vec2::splat(0.5)) * size)
            .collect();
        lettering::contour(a.part(Surface::Light, &foreground), &mut outline, center);
    };
    draw(&[(0.04, 0.12), (0.16, 0.12), (0.16, 0.88), (0.04, 0.88)]);
    // A continuous outline avoids overlapping coplanar limb rectangles.
    draw(&[
        (0.47, 0.69),
        (0.55, 0.64),
        (0.59, 0.51),
        (0.68, 0.50),
        (0.68, 0.43),
        (0.53, 0.44),
        (0.48, 0.55),
        (0.44, 0.43),
        (0.56, 0.31),
        (0.58, 0.13),
        (0.50, 0.13),
        (0.48, 0.27),
        (0.36, 0.34),
        (0.31, 0.15),
        (0.21, 0.10),
        (0.18, 0.17),
        (0.25, 0.22),
        (0.31, 0.45),
        (0.37, 0.62),
        (0.29, 0.57),
        (0.25, 0.42),
        (0.18, 0.44),
        (0.23, 0.64),
        (0.40, 0.72),
    ]);
    let head: Vec<_> = (0..16)
        .map(|i| {
            let angle = i as f32 * 2. * PI / 16.;
            (
                0.52 + angle.cos() * 0.071 * size.y / size.x,
                0.82 + angle.sin() * 0.071,
            )
        })
        .collect();
    draw(&head);
    draw(&[
        (0.84, 0.76),
        (0.91, 0.76),
        (0.91, 0.40),
        (0.97, 0.40),
        (0.875, 0.18),
        (0.78, 0.40),
        (0.84, 0.40),
    ]);
}
