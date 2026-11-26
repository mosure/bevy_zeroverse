//! A continuous back, spine and front. Ink is laid out in print space.
use super::*;

type Ink = [u8; 3];

fn tint(a: Ink, b: Ink, amount: f32) -> Ink {
    std::array::from_fn(|i| (a[i] as f32 + (b[i] as f32 - a[i] as f32) * amount).round() as u8)
}

pub(super) fn print(seed: u64, notebook: bool) -> (Vec<u8>, Ink) {
    let mut r = stream(seed, u64::from_le_bytes(*b"BookInks"));
    let pigments = [
        [37, 80, 87],
        [163, 70, 49],
        [170, 127, 49],
        [73, 88, 120],
        [104, 72, 88],
        [90, 112, 78],
        [178, 96, 57],
        [54, 100, 123],
    ];
    let i = r.random_range(0..pigments.len());
    let j = (i + r.random_range(1..pigments.len())) % pigments.len();
    let cream = tint([236, 230, 212], [205, 201, 181], r.random_range(0.0..0.6));
    let carbon = tint([31, 34, 38], pigments[i], r.random_range(0.04..0.55));
    let (ground, type_ink) = if r.random_bool(0.48) {
        (cream, carbon)
    } else {
        (carbon, cream)
    };
    let accent = tint(pigments[j], cream, r.random_range(0.05..0.42));
    let second = tint(accent, type_ink, r.random_range(0.30..0.75));
    let mut c = Canvas::new(ground);
    let (first, second_word) = if notebook {
        [
            ("FIELD", "NOTES"),
            ("STUDIO", "LOG"),
            ("LAB", "RECORD"),
            ("PROJECT", "BOOK"),
        ][r.random_range(0..4)]
    } else {
        let first = [
            "FIELD", "URBAN", "MODERN", "LIVING", "HIDDEN", "COASTAL", "OPEN", "LOCAL", "SOCIAL",
            "MATERIAL", "HUMAN", "CIVIC", "SPATIAL", "VISUAL", "NATURAL", "DAILY", "BUILT",
            "SHARED",
        ];
        let second = [
            "FORMS",
            "SPACES",
            "LINES",
            "ATLAS",
            "SYSTEMS",
            "PLACES",
            "WORLDS",
            "STUDIES",
            "PATTERNS",
            "HISTORIES",
            "STRUCTURES",
            "LANDMARKS",
            "TRACES",
            "DESIGN",
            "NETWORKS",
            "GARDENS",
            "TEXTURES",
            "LANDSCAPES",
            "FUTURES",
            "INTERIORS",
            "NOTES",
            "PATHS",
        ];
        (
            first[r.random_range(0..first.len())],
            second[r.random_range(0..second.len())],
        )
    };

    let title_high = r.random_bool(0.57);
    let (art_y, title_y) = if title_high {
        (r.random_range(77..125), r.random_range(12..32))
    } else {
        (r.random_range(9..28), r.random_range(171..192))
    };
    let height = if title_high {
        239 - art_y
    } else {
        title_y - art_y - 13
    };
    let margin = r.random_range(5..19);
    let x = FRONT_START + margin;
    let w = 256 - x - r.random_range(3..17);
    c.clipped([x, art_y, w, height], |c| {
        let cx = (x + r.random_range(0..w)) as f32;
        let cy = (art_y + r.random_range(0..height)) as f32;
        match r.random_range(0..4) {
            0 => {
                let radius = r.random_range(24.0..73.);
                c.ellipse([cx, cy], [radius, radius * 1.6], accent);
                c.ellipse(
                    [cx + radius * 0.9, cy + 11.],
                    [radius * 0.72, radius],
                    second,
                );
            }
            1 => {
                let step = r.random_range(13..34);
                let lean = r.random_range(-36..37) as f32;
                for k in 0..(w / step + 4) {
                    let left = (x + (k - 1) * step) as f32;
                    let top = art_y as f32;
                    let bottom = (art_y + height) as f32;
                    c.polygon(
                        &[
                            [left, top],
                            [left + step as f32 * 0.58, top],
                            [left + step as f32 * 0.58 + lean, bottom],
                            [left + lean, bottom],
                        ],
                        if k % 3 == 0 { second } else { accent },
                    );
                }
            }
            2 => {
                let step = r.random_range(10..27);
                for k in 1..9 {
                    let radius = (k * step) as f32;
                    c.ring(
                        [cx, cy],
                        [radius, radius * 1.48],
                        if k % 4 == 0 { second } else { accent },
                    );
                }
            }
            _ => {
                for i in 0..r.random_range(4..10) {
                    let left = x + r.random_range(0..w * 2 / 3);
                    let top = art_y + r.random_range(0..height * 3 / 4);
                    c.rect(
                        left,
                        top,
                        r.random_range(4..w * 2 / 3),
                        r.random_range(7..height / 2),
                        if i % 3 == 0 { second } else { accent },
                    );
                }
            }
        }
    });
    let mut typography = stream(seed, u64::from_le_bytes(*b"BookType"));
    let height = typography.random_range(12.0..21.);
    let tracking = typography.random_range(1.9..4.6);
    let weight = typography.random_range(1..=2);
    let heading = lettering::Type::fit(&[first, second_word], w as f32, height, tracking, weight);
    let origin = Vec2::new(x as f32 + 1., title_y as f32);
    heading.draw(&mut c, first, origin, false, type_ink);
    heading.draw(
        &mut c,
        second_word,
        origin + Vec2::Y * (height + 4.),
        false,
        type_ink,
    );
    let title = format!("{first} {second_word}");
    let size = typography.random_range(7.5..13.);
    let spine = lettering::Type::fit(&[&title], 168., size, tracking + 0.7, 1);
    let pen = Vec2::new(128. + size * 0.5, r.random_range(11..52) as f32);
    spine.draw(&mut c, &title, pen, true, type_ink);
    let imprints = [
        "ARCHIVE PRESS",
        "SOUTH STUDIO",
        "NORTH HOUSE",
        "PAPER WORKS",
        "EDITION",
        "SURVEY PRESS",
    ];
    let imprint = imprints[typography.random_range(0..imprints.len())];
    lettering::Type::fit(&[imprint], 82., 5.5, 2.5, 1).draw(
        &mut c,
        imprint,
        Vec2::new(x as f32 + 1., 247.),
        false,
        type_ink,
    );
    c.rect(
        BACK_END + 4,
        228,
        FRONT_START - BACK_END - 8,
        r.random_range(3..11),
        accent,
    );
    let left = r.random_range(9..27);
    let start = r.random_range(33..95);
    if notebook {
        c.text("NAME", left, start, 1, type_ink);
        c.rect(left, start + 12, 69, 1, second);
        c.text("DATE", left, start + 27, 1, type_ink);
        c.rect(left, start + 39, 50, 1, second);
    } else {
        for line in 0..r.random_range(5..13) {
            let end = left + r.random_range(38..78);
            let mut pen = left;
            while pen < end {
                let word = r.random_range(3..12).min(end - pen);
                c.rect(pen, start + 5 * line, word, 1, second);
                pen += word + r.random_range(2..5);
            }
        }
        let mut x = left;
        c.rect(x - 3, 208, 49, 23, cream);
        while x < left + 40 {
            let width = r.random_range(1..4);
            c.rect(x, 211, width, 16, carbon);
            x += width + r.random_range(1..3);
        }
    }
    (c.0, accent)
}
