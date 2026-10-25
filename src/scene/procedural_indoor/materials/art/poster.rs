//! Editorial posters: an independently cropped graphic, variable text hierarchy
//! and alignments. Shared palettes keep the sampled marks a coherent print.
use super::*;
pub(super) fn paint(c: &mut Canvas, seed: u64, bounds: [i32; 4], art: &ArtComposition) {
    let mut rng = stream(seed, 0x504f53544552);
    let [x, y, w, h] = bounds;
    let margin = rng.random_range(0.03..0.15);
    let split = rng.random_range(0.24..0.76);
    let image_top = rng.random_bool(0.5);
    let (iy, ih, ty, th) = if image_top {
        (
            margin,
            split - margin,
            split + margin,
            1. - split - 2. * margin,
        )
    } else {
        (split, 1. - split - margin, margin, split - 2. * margin)
    };
    let graphic = [
        x + (margin * w as f32) as i32,
        y + (iy * h as f32) as i32,
        (w as f32 * (1. - 2. * margin)) as i32,
        (ih * h as f32) as i32,
    ];
    c.clipped(graphic, |c| {
        if rng.random_bool(0.55) {
            fields::paint(c, graphic, seed, art, rng.random_range(0..4));
        } else {
            let [gx, gy, gw, gh] = graphic;
            let angle = rng.random_range(0.0..std::f32::consts::PI);
            let (sn, cs) = angle.sin_cos();
            let count = rng.random_range(3..=18);
            for i in 0..count {
                let u = rng.random_range(0.0..1.);
                let v = rng.random_range(0.0..1.);
                let a = rng.random_range(0.06..0.32);
                let b = rng.random_range(0.04..0.4);
                let points = [[-a, -b], [a, -b], [a, b], [-a, b]].map(|[aa, bb]| {
                    [
                        gx as f32 + (u + aa * cs - bb * sn) * gw as f32,
                        gy as f32 + (v + aa * sn + bb * cs) * gh as f32,
                    ]
                });
                c.polygon(&points, art.palette[1 + i % 4]);
            }
        }
    });
    let text = [
        x + (margin * w as f32) as i32,
        y + (ty * h as f32) as i32,
        (w as f32 * (1. - 2. * margin)) as i32,
        (th * h as f32) as i32,
    ];
    c.clipped(text, |c| {
        let [tx, ty, tw, th] = text;
        let scale = rng
            .random_range(1..=4)
            .min((th / 7).max(1))
            .min((tw / 16).max(1));
        let title = content::phrase(&mut rng, (tw / (4 * scale)).max(1) as usize);
        let available = (tw - title.len() as i32 * 4 * scale).max(0);
        let indent = (available as f32 * rng.random_range(0.0..1.0)) as i32;
        c.text(&title, tx + indent, ty, scale, art.palette[1]);
        let gap = scale * 5 + rng.random_range(3..12);
        content::paragraph(
            c,
            seed,
            [tx, ty + gap, tw, (th - gap).max(0)],
            art.palette[1],
            rng.random_range(0.15..0.9),
        );
    });
}
