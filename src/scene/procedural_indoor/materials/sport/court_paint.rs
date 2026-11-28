use super::*;
use bevy::color::Luminance;

fn contrast(a: f32, b: f32) -> f32 {
    (a.max(b) + 0.05) / (a.min(b) + 0.05)
}

/// For two sports on one floor, distinguish the secondary pigment both from
/// the timber and from the primary line. The installer chooses one lot per hall.
pub(super) fn secondary(floor: [f32; 3], primary: Color, r: &mut impl Rng) -> Color {
    let floor_y = Color::srgb_from_array(floor).to_linear().luminance();
    let primary = primary.to_srgba();
    let palette = [
        [0.60, 0.11, 0.12],
        [0.12, 0.36, 0.25],
        [0.10, 0.24, 0.49],
        [0.96, 0.75, 0.25],
        [0.34, 0.79, 0.89],
        [0.95, 0.91, 0.85],
        [0.31, 0.08, 0.24],
        [0.08, 0.10, 0.12],
    ];
    let start = r.random_range(0..palette.len());
    let mut best = (0., primary.into());
    for i in 0..palette.len() {
        let rgb = palette[(i + start) % palette.len()];
        let dye = Color::srgb_from_array(rgb);
        if contrast(floor_y, dye.to_linear().luminance()) < 3. {
            continue;
        }
        let distance =
            Vec3::from_array(rgb).distance(Vec3::new(primary.red, primary.green, primary.blue));
        if distance >= 0.36 {
            return dye;
        }
        if distance > best.0 {
            best = (distance, dye);
        }
    }
    best.1
}

/// Under the same illumination, white paint can disappear on pale maple.
/// Choose the installed pigment using the substrate's diffuse reflectance.
pub(super) fn on_floor(seed: u64, floor: [f32; 3], preferred: Color) -> Color {
    let [r, g, b] = floor;
    let y = Color::srgb(r, g, b).to_linear().luminance();
    let readable = |c: Color| contrast(y, c.to_linear().luminance()) >= 3.;
    if readable(preferred) {
        return preferred;
    }
    let mut rng = stream(seed, u64::from_le_bytes(*b"LineDyes"));
    let palette = [
        [0.14, 0.23, 0.36],
        [0.46, 0.19, 0.13],
        [0.13, 0.33, 0.25],
        [0.23, 0.24, 0.26],
    ];
    let [r, g, b] = palette[rng.random_range(0..palette.len())];
    let value = rng.random_range(0.78..1.04);
    let dye = Color::srgb(r * value, g * value, b * value);
    if readable(dye) {
        return dye;
    }
    let warm_white = Color::srgb(0.96, 0.95, 0.92);
    if readable(warm_white) {
        warm_white
    } else {
        Color::srgb(0.11, 0.12, 0.14)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gymnasium_line_stock_keeps_diffuse_contrast_on_light_and_dark_timber() {
        for i in 0..=32 {
            let v = i as f32 / 32.;
            let substrate = [v, v * 0.88, v * 0.72];
            let y = Color::srgb(substrate[0], substrate[1], substrate[2])
                .to_linear()
                .luminance();
            for seed in 0..16 {
                let preferred = if seed % 2 == 0 {
                    Color::srgb(0.82, 0.82, 0.80)
                } else {
                    Color::srgb(0.17, 0.30, 0.55)
                };
                let paint = on_floor(seed, substrate, preferred);
                assert!(contrast(y, paint.to_linear().luminance()) >= 3.);
                if contrast(y, preferred.to_linear().luminance()) >= 3. {
                    assert_eq!(paint, preferred);
                }
            }
        }
    }
}

#[cfg(test)]
#[test]
fn practice_line_pigments_read_on_timber_and_differ_from_the_volleyball_paint() {
    let mut dyes = std::collections::BTreeSet::new();
    for i in 0..=32 {
        let v = i as f32 / 32.;
        let floor = [v, v * 0.88, v * 0.72];
        for seed in 0..20 {
            let primary = on_floor(seed, floor, Color::srgb(0.84, 0.84, 0.81));
            let color = secondary(floor, primary, &mut stream(seed, 912));
            assert!(
                contrast(
                    Color::srgb_from_array(floor).to_linear().luminance(),
                    color.to_linear().luminance()
                ) >= 3.
            );
            let a = primary.to_srgba();
            let b = color.to_srgba();
            assert!(Vec3::new(a.red - b.red, a.green - b.green, a.blue - b.blue).length() > 0.15);
            dyes.insert(color.to_srgba().to_u8_array());
        }
    }
    assert!(dyes.len() >= 6);
}
