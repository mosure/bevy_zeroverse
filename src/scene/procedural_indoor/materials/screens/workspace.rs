//! Responsive tiled software/media. Subdivision and content are independent.
use super::*;
pub(super) fn pixels(seed: u64, kind: DisplayKind, p: &ScreenProgram) -> Vec<u8> {
    let colors = palette(p);
    let mut c = Canvas::new(colors.bg);
    chrome::draw(&mut c, seed, kind, p);
    for (i, r) in p.composition.regions.iter().enumerate() {
        let mut content_rng = stream(r.seed, 0x434f4e54454e54);
        let mut kind_id = content_rng.random_range(0..content::KINDS);
        // The leading region can retain the main task; others mix independent documents.
        if i == 0 && content_rng.random_bool(0.35) {
            kind_id = p.layout % content::KINDS;
        }
        if matches!(kind, DisplayKind::Phone) && kind_id == 5 {
            kind_id = 3;
        }
        if matches!(kind, DisplayKind::Television) && kind_id == 5 {
            kind_id = 8;
        }
        content::panel(&mut c, r.seed, r.pixels(), kind_id, p.dark);
    }
    c.0
}
