//! Large display content composed without desktop taskbar/sidebar chrome.
use super::*;
pub(super) fn pixels(seed: u64) -> Vec<u8> {
    let p = parameters(seed);
    if p.dormant {
        return dormant_canvas(seed).0;
    }
    let colors = palette(&p);
    let mut rng = stream(seed, 3081);
    let mut c = Canvas::new(colors.panel);
    match p.layout {
        0 | 1 | 7 => {
            let margin = 12 + (p.sidebar * 55.) as i32;
            let split = (256. * p.content_split) as i32;
            c.text(
                ["MONTHLY REVIEW", "PROJECT UPDATE", "TEAM PLANNING"]
                    [if p.layout == 7 { 2 } else { p.layout as usize }],
                margin,
                22,
                2,
                colors.ink,
            );
            if p.layout == 7 {
                for col in 0..3 {
                    let x = margin + col * (256 - margin * 2) / 3;
                    c.text(
                        ["PLAN", "ACTIVE", "DONE"][col as usize],
                        x,
                        54,
                        1,
                        colors.ink,
                    );
                    for row in 0..(p.density * 6.).ceil() as i32 {
                        c.rect(x, 75 + row * 23, (256 - margin * 2) / 3 - 7, 18, colors.bg);
                        c.rect(
                            x + 4,
                            81 + row * 23,
                            (256 - margin * 2) / 3 - 17,
                            2,
                            colors.accent,
                        );
                    }
                }
            } else {
                text_rows(
                    &mut c,
                    &mut rng,
                    [margin, 57, split - margin - 7, 143],
                    colors.ink,
                    p.density,
                    false,
                );
                chart(
                    &mut c,
                    seed,
                    &p,
                    [split + 7, 65, 256 - split - margin - 7, 142],
                    colors,
                    p.panels,
                );
            }
            c.rect(0, 235, 256, 21, colors.accent);
            c.time(p.minutes, 198, 240, 1, colors.panel);
        }
        2 => {
            let green = rng.random_range(79..128);
            c.rect(0, 0, 256, 256, [34, green, 54]);
            for stripe in 0..8 {
                if stripe % 2 == 0 {
                    c.rect(0, stripe * 32, 256, 32, [36, green + 8, 57]);
                }
            }
            for x in [14, 128, 242] {
                c.line([x as f32, 14.], [x as f32, 242.], 2, [218, 228, 211]);
            }
            for y in [14, 242] {
                c.line([14., y as f32], [242., y as f32], 2, [218, 228, 211]);
            }
            c.ring([128., 128.], [26., 32.], [218, 228, 211]);
            c.line([15., 77.], [48., 77.], 1, [218, 228, 211]);
            c.line([48., 77.], [48., 181.], 1, [218, 228, 211]);
            c.line([48., 181.], [15., 181.], 1, [218, 228, 211]);
            for i in 0..22 {
                let x = rng.random_range(23..232);
                let y = rng.random_range(29..224);
                let team = if i % 2 == 0 {
                    [219, 76, 58]
                } else {
                    [230, 232, 237]
                };
                c.ellipse([(x + 2) as f32, (y + 2) as f32], [2., 2.], [198, 146, 96]);
                c.rect(x, y + 3, 4, 5, team);
            }
            c.rect(13, 8, 92, 14, [17, 22, 25]);
            c.text("HOME", 17, 12, 1, [242; 3]);
            c.digit((seed % 5) as u32, 38, 11, 1, [242; 3]);
            c.text("AWAY", 55, 12, 1, [242; 3]);
            c.digit((seed / 5 % 5) as u32, 81, 11, 1, [242; 3]);
        }
        3 | 5 | 6 => {
            let art = art::parameters(seed ^ 3085);
            c = art::landscape(seed, &art, p.layout == 5);
            if p.layout == 6 {
                let height = (256. * p.framing) as i32;
                c.rect(0, 0, 256, (256 - height) / 2, [6; 3]);
                c.rect(0, (256 + height) / 2, 256, (256 - height) / 2, [6; 3]);
            }
        }
        4 => {
            c.rect(0, 0, 256, 256, colors.bg);
            participants(&mut c, seed, &p, [8, 16, 240, 215], colors);
            for i in 0..5 {
                c.ellipse(
                    [87. + i as f32 * 20., 245.],
                    [5., 5.],
                    if i == 4 { [176, 64, 55] } else { colors.muted },
                );
            }
        }
        8 => {
            // Building controls and presentation plans: independent floor-grid
            // subdivisions, occupied zones and sensor readings.
            c.text("BUILDING STATUS", 21, 19, 2, colors.ink);
            let columns = 2 + p.panels.min(4);
            for row in 0..3 {
                for col in 0..columns {
                    let x = 19 + col as i32 * 219 / columns as i32;
                    let y = 51 + row * 56;
                    let w = 219 / columns as i32 - 3;
                    c.rect(x, y, w, 50, colors.bg);
                    if rng.random_bool(p.density as f64) {
                        c.rect(x + 2, y + 2, w - 4, 46, colors.accent);
                    }
                    c.digit(rng.random_range(0..10), x + 5, y + 17, 2, colors.ink);
                }
            }
        }
        9 => {
            map(&mut c, seed, &p, [0, 24, 256, 211], colors);
            c.rect(0, 0, 256, 24, colors.panel);
            c.text("LOCAL AREA", 15, 8, 2, colors.ink);
        }
        10 => {
            c.text("ROOM SCHEDULE", 18, 24, 2, colors.ink);
            let rows = 2 + (p.density * 6.) as i32;
            for row in 0..rows {
                let y = 56 + row * 176 / rows;
                c.rect(15, y, 226, 176 / rows - 3, colors.bg);
                c.time(
                    (p.minutes + row as u32 * 45) % 1440,
                    23,
                    y + 5,
                    1,
                    colors.ink,
                );
                c.rect(
                    72,
                    y + 7,
                    (p.content_split * 137.) as i32 + 19,
                    3,
                    colors.accent,
                );
            }
        }
        _ => {
            // Metric monitoring panels whose curves share a coherent slope;
            // independent inset sizes and traces distinguish each instrument.
            let cols = if p.panels < 3 { 1 } else { 2 };
            let rows = p.panels.div_ceil(cols) as i32;
            c.text("OPERATIONS", 17, 14, 2, colors.ink);
            for i in 0..p.panels {
                let x = 13 + (i % cols) as i32 * 230 / cols as i32;
                let y = 44 + (i / cols) as i32 * 195 / rows;
                chart(
                    &mut c,
                    seed + i as u64,
                    &p,
                    [x + 5, y + 4, 230 / cols as i32 - 15, 195 / rows - 12],
                    colors,
                    i + 1,
                );
            }
        }
    }
    c.0
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn television_compositions_cover_families_and_replay() {
        let images: std::collections::BTreeSet<_> = (0..128)
            .filter(|&s| !parameters(s).dormant)
            .map(pixels)
            .collect();
        let layouts: std::collections::BTreeSet<_> =
            (0..128).map(|s| parameters(s).layout).collect();
        assert!(images.len() > 95);
        assert_eq!(layouts.len(), LAYOUTS as usize);
        assert!(images.iter().all(|i| i.len() == 256 * 256 * 4));
        assert_eq!(pixels(315), pixels(315));
    }
}
