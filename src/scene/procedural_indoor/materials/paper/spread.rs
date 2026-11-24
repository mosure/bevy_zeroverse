//! One mipmapped chart for the facing pages. Paper stock, ruling and line pitch
//! come from one notebook; each page has its own writing schedule and marks.
use super::*;

pub(crate) fn slot(object_seed: u64) -> usize {
    variants::COUNT + variants::object_slot(Surface::PrintedPaper, object_seed, None)
}

pub(super) fn pixels(seed: u64) -> Vec<u8> {
    let stock = parameters(seed);
    let mut canvas = Canvas::new(stock.paper);
    let mut leaves = stream(seed, 0x5041474550414952);
    for x0 in [0, 128] {
        let leaf_seed = leaves.random();
        let mut r = stream(leaf_seed, 0x4d41524b53);
        let mut p = stock.clone();
        if x0 != 0 {
            p.margin.swap(0, 2); // Bound margin faces the gutter on each leaf.
        }
        if p.writing > 0. {
            p.writing = if r.random_bool(0.18) {
                0.
            } else {
                (p.writing * r.random_range(0.35..1.45)).min(1.)
            };
        }
        draw_page(&mut canvas, &p, leaf_seed, x0, 128);
    }
    canvas.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facing_stock_matches_and_the_two_sets_of_notes_have_independent_coverage() {
        let mut varied = 0;
        let mut one_blank = 0;
        let mut both_blank = 0;
        for seed in 0..128 {
            let stock = parameters(seed);
            let bytes = pixels(seed);
            assert_eq!(bytes, pixels(seed));
            let mut ink = [0; 2];
            let mut identical = true;
            for row in bytes.chunks_exact(256 * 4) {
                let (left, right) = row.split_at(128 * 4);
                for (i, page) in [left, right].into_iter().enumerate() {
                    assert_eq!(&page[..3], &stock.paper);
                    assert_eq!(&page[127 * 4..127 * 4 + 3], &stock.paper);
                    ink[i] += page.chunks_exact(4).filter(|v| v[..3] == stock.ink).count();
                }
                identical &= left == right;
            }
            varied += usize::from(!identical && ink.iter().all(|&n| n > 0));
            one_blank += usize::from((ink[0] == 0) != (ink[1] == 0));
            both_blank += usize::from(ink == [0, 0]);
        }
        assert!(
            varied > 40 && one_blank > 10 && both_blank > 10,
            "{varied}/{one_blank}/{both_blank}"
        );
    }
}
