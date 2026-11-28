//! Four player parts per room, with independent music on facing pages.
use super::*;

pub(crate) const FIRST: usize = jacket::END;
pub(crate) const END: usize = FIRST + 4;

pub(crate) fn slot(seed: u64) -> usize {
    stream(seed, u64::from_le_bytes(*b"MuscPart")).random_range(FIRST..END)
}

pub(super) fn pixels(seed: u64) -> Vec<u8> {
    let mut r = stream(seed, u64::from_le_bytes(*b"MuscPage"));
    let white = r.random_range(230u8..251);
    let mut page = Canvas::new([white, white - 2, white - r.random_range(3..13)]);
    let ink = [48, 45, 43];
    let stave = [131, 125, 116];
    let systems = r.random_range(5..=8);
    let width = r.random_range(92..111);
    let title = [
        "ANDANTE", "MODERATO", "SCHERZO", "ADAGIO", "ALLEGRO", "RONDO",
    ][r.random_range(0..6)];
    for side in 0..2 {
        let left = 128 * side + (128 - width) / 2;
        let top = if side == 0 { 34 } else { 24 };
        if side == 0 {
            page.text(title, left, 11, 1, ink);
        }
        let measures = r.random_range(2..=4);
        for row in 0..systems {
            let y = top + row * ((244 - top) / systems);
            for line in 0..5 {
                page.rect(left, y + 2 * line, width, 1, stave);
            }
            for bar in 0..measures {
                let x0 = left + bar * width / measures;
                let x1 = left + (bar + 1) * width / measures;
                page.rect(x1 - 1, y, 1, 9, ink);
                let beats = r.random_range(2..=5);
                let mut pitch: i32 = r.random_range(-1..=8);
                for beat in 0..beats {
                    let x = (x0 + 3 + beat * (x1 - x0 - 3) / beats) as f32;
                    let note = (y + pitch) as f32;
                    page.ellipse([x, note], [1.5, 1.1], ink);
                    let down = pitch < 4;
                    let stem_x = if down { x - 1. } else { x + 1. };
                    let end = note + if down { 6. } else { -6. };
                    page.line([stem_x, note], [stem_x, end], 1, ink);
                    pitch = (pitch + r.random_range(-3..=3)).clamp(-2, 10);
                }
            }
        }
    }
    page.0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rehearsal_scores_replay_distinct_pages_and_filter_down_to_one_texel() {
        let map = super::super::prepare_slot(19, FIRST);
        assert_eq!(map.texture_descriptor.mip_level_count, 9);
        let a = pixels(19);
        assert_eq!(a, pixels(19));
        assert_ne!(a, pixels(20));
        let tail = map.data.as_ref().unwrap();
        assert_eq!(&tail[..a.len()], a.as_slice());
        assert!((190..250).contains(&tail[tail.len() - 4]));
        let rows_different = a
            .as_chunks::<{ 256 * 4 }>()
            .0
            .iter()
            .filter(|line| line[..128 * 4] != line[128 * 4..])
            .count();
        assert!(rows_different > 60);
    }
}
