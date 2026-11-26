//! Cylindrical product artwork in one normalized wrap.
use super::*;

fn hsl(h: f32, s: f32, l: f32) -> [u8; 3] {
    let c = Color::hsl(h.rem_euclid(360.), s, l).to_srgba();
    [c.red, c.green, c.blue].map(|v| (v * 255.).round() as u8)
}

pub(super) fn label_pixels(r: &mut impl Rng, bottle: bool) -> Vec<u8> {
    use raster::Canvas;
    let hue = if bottle && r.random_bool(0.66) {
        r.random_range(152.0..237.)
    } else {
        r.random_range(0.0..360.)
    };
    let pale = r.random_bool(if bottle { 0.69 } else { 0.29 });
    let base = if pale {
        hsl(hue, r.random_range(0.13..0.37), r.random_range(0.84..0.95))
    } else {
        hsl(hue, r.random_range(0.46..0.81), r.random_range(0.20..0.38))
    };
    let accent = hsl(
        hue + r.random_range(-76.0..70.),
        r.random_range(0.52..0.86),
        if pale {
            r.random_range(0.33..0.47)
        } else {
            r.random_range(0.49..0.68)
        },
    );
    let ink = if pale {
        hsl(hue, 0.41, 0.13)
    } else {
        [250, 246, 230]
    };
    let mut c = Canvas::new(base);
    let front = r.random_range(73..118);
    let y = r.random_range(70..118);
    // Graphics meet at the wrap seam on the same pigment. Leave a quiet field for the wordmark.
    let motif = r.random_range(0..4);
    if motif == 0 || (bottle && motif == 1) {
        let line = r.random_range(157..195) as f32;
        let mut skyline = vec![[0., 256.], [0., line]];
        for i in 1..8 {
            skyline.push([
                i as f32 * 32. + r.random_range(-8.0..8.),
                line - r.random_range(-17.0..37.),
            ]);
        }
        skyline.extend([[256., line], [256., 256.]]);
        c.polygon(&skyline, accent);
    } else if motif == 1 {
        let cy = r.random_range(143.0..203.);
        let rad = r.random_range(26.0..40.);
        c.ellipse([front as f32 + 20., cy], [rad, rad * 1.45], accent);
        for i in 0..8 {
            let (sn, cs) = (i as f32 * std::f32::consts::TAU / 8.).sin_cos();
            c.line(
                [front as f32 + 20. + cs * 5., cy + sn * 7.],
                [front as f32 + 20. + cs * rad * 0.84, cy + sn * rad * 1.20],
                2,
                base,
            );
        }
    } else if motif == 2 {
        let x = r.random_range(28.0..74.);
        let w = r.random_range(12.0..38.);
        c.polygon(
            &[[x, 0.], [x + w, 0.], [x + w + 36., 256.], [x + 36., 256.]],
            accent,
        );
    } else {
        for _ in 0..r.random_range(6..14) {
            let x = r.random_range(16.0..158.);
            let y = r.random_range(122.0..234.);
            let a = r.random_range(2.5..10.);
            c.ring([x, y], [a, a * 1.7], accent);
        }
    }
    let name = format!(
        "{}{}",
        ["OR", "VE", "LU", "KA", "AR", "RI", "NO", "EL", "MO", "SA"][r.random_range(0..10)],
        ["VA", "NDA", "LO", "RIS", "VIA", "REN", "NA", "TA", "RA", "MER"][r.random_range(0..10)]
    );
    let sx = r.random_range(2..=4);
    let sy = sx * r.random_range(2..=3);
    let start = front - (name.len() as i32 * 4 - 1) * sx / 2;
    // A printed cartouche is optional; it also keeps light lettering legible above a stripe.
    if motif == 2 || r.random_bool(0.32) {
        c.rect(
            start - 5,
            y - 7,
            (name.len() as i32 * 4 - 1) * sx + 10,
            5 * sy + 14,
            base,
        );
    }
    c.lettering(&name, start, y, [sx, sy], ink);
    let sub = if bottle {
        ["WATER", "SPRING", "MINERAL"][r.random_range(0..3)]
    } else {
        ["SODA", "FIZZ", "TONIC", "POP"][r.random_range(0..4)]
    };
    c.lettering(
        sub,
        front - 2 * sub.len() as i32,
        y + sy * 5 + 8,
        [1, 3],
        ink,
    );
    let x = r.random_range(176..195);
    let width = r.random_range(41..54);
    let py = r.random_range(39..85);
    let stock = [240, 237, 224];
    let stamp = [38, 42, 43];
    c.rect(x, py, width, 129, stock);
    c.rect(x + 3, py + 7, width - 6, 3, stamp);
    for row in 0..r.random_range(4..8) {
        c.rect(
            x + 3,
            py + 15 + row * 9,
            r.random_range(12..width - 9),
            r.random_range(1..=2),
            stamp,
        );
    }
    let mut ix = x + 4;
    while ix < x + width - 5 {
        let w = r.random_range(1..=3);
        c.rect(ix, py + 89, w, 29, stamp);
        ix += w + r.random_range(1..=3);
    }
    c.0
}
