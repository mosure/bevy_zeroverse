//! Large-screen media, presentation and signage layouts, without desktop chrome.
use super::*;
pub(super) fn pixels(seed: u64) -> Vec<u8> {
    let p = parameters(seed);
    let mut rng = stream(seed, 3081);
    let mut c = Canvas(vec![255; (N * N * 4) as usize]);
    c.rect(0, 0, N, N, [10, 17, 26]);
    if p.layout == 7 {
        return c.0;
    }
    let accent = [
        rng.random_range(40..180),
        rng.random_range(70..170),
        rng.random_range(120..220),
    ];
    match p.layout {
        0 | 1 => {
            c.rect(
                0,
                0,
                256,
                256,
                if p.dark {
                    [21, 29, 38]
                } else {
                    [241, 241, 232]
                },
            );
            c.rect(20, 23, 155, 9, accent);
            for row in 0..5 {
                c.rect(25, 54 + row * 23, rng.random_range(50..120), 4, accent);
            }
            for i in 0..4 {
                let h = rng.random_range(24..120);
                c.rect(167 + i * 18, 194 - h, 13, h, accent);
            }
            c.rect(0, 231, 256, 25, accent);
            c.time(p.minutes, 199, 238, 1, [237, 242, 245]);
        }
        2 => {
            c.rect(0, 0, 256, 256, [40, 101, 61]);
            for x in [14, 128, 242] {
                c.line([x as f32, 14.], [x as f32, 242.], 2, [220, 231, 211]);
            }
            for y in [14, 242] {
                c.line([14., y as f32], [242., y as f32], 2, [220, 231, 211]);
            }
            c.ring([128., 128.], [26., 32.], [220, 231, 211]);
            for i in 0..18 {
                let x = rng.random_range(23..232);
                let y = rng.random_range(29..224);
                c.rect(
                    x,
                    y,
                    4,
                    8,
                    if i % 2 == 0 {
                        [219, 76, 58]
                    } else {
                        [71, 115, 220]
                    },
                );
            }
            c.rect(13, 8, 79, 14, [16, 21, 25]);
            c.digit((seed % 5) as u32, 26, 11, 1, [245; 3]);
            c.digit((seed / 5 % 5) as u32, 65, 11, 1, [245; 3]);
        }
        3 | 6 => {
            for y in 0..256 {
                let t = y as f32 / 256.;
                c.rect(
                    0,
                    y,
                    256,
                    1,
                    [
                        (76. + t * 75.) as u8,
                        (123. + t * 67.) as u8,
                        (174. + t * 32.) as u8,
                    ],
                );
                for x in 0..256 {
                    let ridge = 151. + (x as f32 * 0.027 + (seed % 17) as f32).sin() * 31.;
                    if y as f32 > ridge {
                        c.rect(x, y, 1, 1, [35, 69, 64]);
                    }
                }
            }
        }
        4 => {
            // A full-screen call: independently framed participants rather than
            // the taskbar/sidebar used on computer displays.
            for row in 0..2 {
                for col in 0..3 {
                    let x = 7 + col * 83;
                    let y = 17 + row * 109;
                    let background = [
                        rng.random_range(32..90),
                        rng.random_range(39..96),
                        rng.random_range(43..103),
                    ];
                    c.rect(x, y, 76, 99, background);
                    let skin = [
                        rng.random_range(111..210),
                        rng.random_range(75..154),
                        rng.random_range(54..130),
                    ];
                    let cx = (x + 38 + rng.random_range(-8..9)) as f32;
                    for radius in 1..13 {
                        c.ring(
                            [cx, (y + 36) as f32],
                            [radius as f32, radius as f32 * 1.2],
                            skin,
                        );
                    }
                    for radius in 1..26 {
                        c.ring(
                            [cx, (y + 85) as f32],
                            [radius as f32, radius as f32 * 1.1],
                            accent,
                        );
                    }
                    c.rect(x + 5, y + 90, rng.random_range(23..56), 2, [225; 3]);
                }
            }
        }
        _ => {
            c.rect(18, 30, 220, 5, accent);
            c.time(p.minutes, 41, 93, 6, [230, 237, 240]);
            for row in 0..4 {
                c.rect(40, 168 + row * 12, rng.random_range(80..172), 3, accent);
            }
        }
    }
    c.0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn television_content_is_seeded_and_varied() {
        let images: std::collections::BTreeSet<_> = (0..64).map(pixels).collect();
        let layouts: std::collections::BTreeSet<_> =
            (0..64).map(|s| parameters(s).layout).collect();
        assert!(images.len() > 40);
        assert_eq!(layouts.len(), 8);
        assert!(images.iter().all(|i| i.len() == 256 * 256 * 4));
        assert_eq!(pixels(315), pixels(315));
    }
}
