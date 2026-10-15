//! Responsive tiled software/media. Subdivision and content are independent.
use super::*;
pub(super) fn pixels(seed: u64, kind: DisplayKind, p: &ScreenProgram) -> Vec<u8> {
    let colors = palette(p);
    let mut c = Canvas::new(colors.bg);
    let mut rng = stream(seed, 0x4348524f4d45);
    let desktop = matches!(kind, DisplayKind::Desktop);
    if p.chrome != 0 {
        let top = (p
            .composition
            .regions
            .iter()
            .map(|r| r.bounds[1])
            .fold(1., f32::min)
            * 256.) as i32;
        c.rect(0, 0, N, (top - 4).max(12), colors.panel);
        if matches!(kind, DisplayKind::Phone) {
            c.time(p.minutes, 10, 6, 1, colors.ink);
            c.rect(225, 7, 18, 6, colors.muted);
            c.rect(226, 8, rng.random_range(3..=16), 4, colors.ink);
            c.rect(243, 9, 2, 2, colors.muted);
        } else {
            let title = content::phrase(&mut rng, 30);
            c.text(&title, 8, 7, 1, colors.ink);
        }
        if desktop {
            c.time(p.minutes, 213, 6, 1, colors.muted);
        }
    }
    if p.chrome == 2 && desktop {
        let width = (p
            .composition
            .regions
            .iter()
            .map(|r| r.bounds[0])
            .fold(1., f32::min)
            * 256.) as i32
            - 5;
        c.rect(0, 32, width, 224, colors.panel);
        for row in 0..rng.random_range(4..10) {
            let name = content::WORDS[rng.random_range(0..content::WORDS.len())];
            c.clipped([5, 40 + row * 15, width - 8, 10], |c| {
                c.text(
                    name,
                    6,
                    42 + row * 15,
                    1,
                    if row == 2 {
                        colors.accent
                    } else {
                        colors.muted
                    },
                )
            });
        }
    }
    for (i, r) in p.composition.regions.iter().enumerate() {
        let mut content_rng = stream(r.seed, 0x434f4e54454e54);
        let mut kind_id = content_rng.random_range(0..10);
        // The leading region can retain the main task; others mix independent documents.
        if i == 0 && content_rng.random_bool(0.35) {
            kind_id = p.layout % 10;
        }
        if matches!(kind, DisplayKind::Phone) && kind_id == 5 {
            kind_id = 3;
        }
        if matches!(kind, DisplayKind::Television) && kind_id == 5 {
            kind_id = 8;
        }
        content::panel(&mut c, r.seed, r.pixels(), kind_id, p.dark);
    }
    if matches!(kind, DisplayKind::Phone) {
        c.rect(97, 251, 62, 2, colors.muted);
    }
    c.0
}
