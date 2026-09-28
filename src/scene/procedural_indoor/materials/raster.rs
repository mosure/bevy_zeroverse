//! Small asset-free raster primitives shared by screens and printed paper.
pub(super) const N: i32 = 256;
pub(super) struct Canvas(pub Vec<u8>);
impl Canvas {
    pub(super) fn line(&mut self, a: [f32; 2], b: [f32; 2], width: i32, color: [u8; 3]) {
        let n = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()).ceil() as usize).max(1);
        for i in 0..=n {
            let t = i as f32 / n as f32;
            let x = a[0] + t * (b[0] - a[0]);
            let y = a[1] + t * (b[1] - a[1]);
            self.rect(
                x.round() as i32 - width / 2,
                y.round() as i32 - width / 2,
                width,
                width,
                color,
            );
        }
    }
    pub(super) fn ring(&mut self, center: [f32; 2], radius: [f32; 2], color: [u8; 3]) {
        for i in 0..72 {
            let point = |i: usize| {
                let t = i as f32 * std::f32::consts::TAU / 72.;
                [
                    center[0] + t.cos() * radius[0],
                    center[1] + t.sin() * radius[1],
                ]
            };
            self.line(point(i), point(i + 1), 2, color);
        }
    }
    pub(super) fn rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: [u8; 3]) {
        for y in y.max(0)..(y + h).min(N) {
            for x in x.max(0)..(x + w).min(N) {
                let i = ((y * N + x) * 4) as usize;
                self.0[i..i + 3].copy_from_slice(&color);
            }
        }
    }
    pub(super) fn digit(&mut self, digit: u32, x: i32, y: i32, scale: i32, color: [u8; 3]) {
        for (index, (dx, dy, w, h)) in [
            (1, 0, 3, 1),
            (4, 1, 1, 3),
            (4, 5, 1, 3),
            (1, 8, 3, 1),
            (0, 5, 1, 3),
            (0, 1, 1, 3),
            (1, 4, 3, 1),
        ]
        .into_iter()
        .enumerate()
        {
            if super::screens::SEGMENTS[digit as usize] & (1 << index) != 0 {
                self.rect(x + dx * scale, y + dy * scale, w * scale, h * scale, color);
            }
        }
    }
    pub(super) fn time(&mut self, minutes: u32, x: i32, y: i32, scale: i32, color: [u8; 3]) {
        for (i, d) in [
            minutes / 600,
            minutes / 60 % 10,
            minutes % 60 / 10,
            minutes % 10,
        ]
        .into_iter()
        .enumerate()
        {
            self.digit(
                d,
                x + i as i32 * scale * 7 + if i >= 2 { scale * 2 } else { 0 },
                y,
                scale,
                color,
            );
        }
        for dy in [2, 6] {
            self.rect(x + scale * 14, y + scale * dy, scale, scale, color);
        }
    }
}
