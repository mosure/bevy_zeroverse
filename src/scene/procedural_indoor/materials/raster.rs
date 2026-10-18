//! Small asset-free raster primitives shared by screens and printed paper.
pub(super) const N: i32 = 256;
pub(super) struct Canvas(pub Vec<u8>, [i32; 4]);
impl Canvas {
    pub(super) fn new(color: [u8; 3]) -> Self {
        Self(
            [color[0], color[1], color[2], 255].repeat((N * N) as usize),
            [0, 0, N, N],
        )
    }
    /// Nested panel clipping prevents graphs, text and media from bleeding into neighbors.
    pub(super) fn clipped(&mut self, bounds: [i32; 4], draw: impl FnOnce(&mut Self)) {
        let saved = self.1;
        let [x, y, w, h] = bounds;
        self.1 = [
            x.max(saved[0]),
            y.max(saved[1]),
            (x + w).min(saved[2]),
            (y + h).min(saved[3]),
        ];
        draw(self);
        self.1 = saved;
    }
    pub(super) fn blend(&mut self, x: i32, y: i32, color: [u8; 3], alpha: f32) {
        if !(self.1[0]..self.1[2]).contains(&x) || !(self.1[1]..self.1[3]).contains(&y) {
            return;
        }
        let i = ((y * N + x) * 4) as usize;
        let alpha = alpha.clamp(0., 1.);
        for (channel, value) in color.into_iter().enumerate() {
            self.0[i + channel] =
                (self.0[i + channel] as f32 * (1. - alpha) + value as f32 * alpha).round() as u8;
        }
    }
    pub(super) fn ellipse(&mut self, center: [f32; 2], radius: [f32; 2], color: [u8; 3]) {
        self.ellipse_clipped(center, radius, color, [0, 0, N, N]);
    }
    pub(super) fn ellipse_clipped(
        &mut self,
        center: [f32; 2],
        radius: [f32; 2],
        color: [u8; 3],
        bounds: [i32; 4],
    ) {
        let [left, top, width, height] = bounds;
        let x0 = ((center[0] - radius[0]).floor() as i32)
            .max(left)
            .max(self.1[0]);
        let x1 = ((center[0] + radius[0]).ceil() as i32)
            .min(left + width - 1)
            .min(self.1[2] - 1);
        let y0 = ((center[1] - radius[1]).floor() as i32)
            .max(top)
            .max(self.1[1]);
        let y1 = ((center[1] + radius[1]).ceil() as i32)
            .min(top + height - 1)
            .min(self.1[3] - 1);
        if x0 > x1 || y0 > y1 {
            return;
        }
        // Reuse the exact divided/squared coordinates, not reciprocal products
        // or an approximate ellipse boundary. Clip once, before pixel work.
        let mut horizontal = [0.; N as usize];
        for (x, value) in (x0..=x1).zip(&mut horizontal) {
            *value = ((x as f32 - center[0]) / radius[0].max(0.1)).powi(2);
        }
        for y in y0..=y1 {
            let vertical = ((y as f32 - center[1]) / radius[1].max(0.1)).powi(2);
            let row = &mut self.0[((y * N + x0) * 4) as usize..((y * N + x1 + 1) * 4) as usize];
            for (pixel, &horizontal) in row.as_chunks_mut::<4>().0.iter_mut().zip(&horizontal) {
                if horizontal + vertical <= 1. {
                    // Opaque blending is exactly this RGB assignment. Preserve
                    // the destination alpha, including externally edited pixels.
                    pixel[..3].copy_from_slice(&color);
                }
            }
        }
    }
    pub(super) fn polygon(&mut self, points: &[[f32; 2]], color: [u8; 3]) {
        if points.len() < 3 {
            return;
        }
        let min_y = points.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let max_y = points
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max);
        // Scanline filling visits each covered pixel once, rather than testing
        // every edge at every texel. Small paths are bounded by the 256px atlas.
        let mut intersections = Vec::with_capacity(points.len());
        for y in (min_y.floor() as i32).max(0)..=(max_y.ceil() as i32).min(N - 1) {
            intersections.clear();
            let fy = y as f32 + 0.5;
            for i in 0..points.len() {
                let a = points[i];
                let b = points[(i + 1) % points.len()];
                if (a[1] <= fy && b[1] > fy) || (b[1] <= fy && a[1] > fy) {
                    intersections.push(a[0] + (fy - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
                }
            }
            intersections.sort_by(f32::total_cmp);
            for pair in intersections.as_chunks::<2>().0 {
                self.rect(
                    pair[0].ceil() as i32,
                    y,
                    (pair[1].ceil() - pair[0].ceil()) as i32,
                    1,
                    color,
                );
            }
        }
    }
    pub(super) fn text(&mut self, text: &str, x: i32, y: i32, scale: i32, color: [u8; 3]) {
        // A compact bitmap alphabet gives headings actual readable content;
        // no font loading, hidden Unicode fallback, or per-frame rasterization.
        const LETTERS: [[u8; 5]; 26] = [
            [2, 5, 7, 5, 5],
            [6, 5, 6, 5, 6],
            [3, 4, 4, 4, 3],
            [6, 5, 5, 5, 6],
            [7, 4, 6, 4, 7],
            [7, 4, 6, 4, 4],
            [3, 4, 5, 5, 3],
            [5, 5, 7, 5, 5],
            [7, 2, 2, 2, 7],
            [1, 1, 1, 5, 2],
            [5, 5, 6, 5, 5],
            [4, 4, 4, 4, 7],
            [5, 7, 7, 5, 5],
            [5, 7, 7, 7, 5],
            [2, 5, 5, 5, 2],
            [6, 5, 6, 4, 4],
            [2, 5, 5, 3, 1],
            [6, 5, 6, 5, 5],
            [3, 4, 2, 1, 6],
            [7, 2, 2, 2, 2],
            [5, 5, 5, 5, 7],
            [5, 5, 5, 5, 2],
            [5, 5, 7, 7, 5],
            [5, 5, 2, 5, 5],
            [5, 5, 2, 2, 2],
            [7, 1, 2, 4, 7],
        ];
        for (index, letter) in text.bytes().enumerate() {
            let x = x + index as i32 * 4 * scale;
            if letter.is_ascii_alphabetic() {
                let glyph = LETTERS[(letter.to_ascii_uppercase() - b'A') as usize];
                for (row, bits) in glyph.into_iter().enumerate() {
                    for col in 0..3 {
                        if bits & (1 << (2 - col)) != 0 {
                            self.rect(x + col * scale, y + row as i32 * scale, scale, scale, color);
                        }
                    }
                }
            }
        }
    }
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
        let x0 = x.max(self.1[0]);
        let x1 = (x + w).min(self.1[2]);
        if x0 >= x1 {
            return;
        }
        for y in y.max(self.1[1])..(y + h).min(self.1[3]) {
            let row = &mut self.0[((y * N + x0) * 4) as usize..((y * N + x1) * 4) as usize];
            for pixel in row.as_chunks_mut::<4>().0 {
                pixel[..3].copy_from_slice(&color);
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

#[cfg(test)]
mod replay_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_clips_restore_and_contain_every_primitive() {
        let mut c = Canvas::new([17, 23, 31]);
        c.clipped([40, 50, 90, 70], |c| {
            c.clipped([0, 0, 100, 100], |c| {
                c.rect(-20, -20, 300, 300, [50; 3]);
                c.line([0., 0.], [255., 255.], 5, [100; 3]);
                c.ellipse([99., 99.], [75.; 2], [200; 3]);
            });
            c.rect(110, 110, 4, 4, [60; 3]);
        });
        for y in 0..N {
            for x in 0..N {
                if !(40..130).contains(&x) || !(50..120).contains(&y) {
                    assert_eq!(&c.0[((y * N + x) * 4) as usize..][..3], &[17, 23, 31]);
                }
            }
        }
        c.rect(1, 1, 1, 1, [255; 3]);
        assert_eq!(&c.0[((N + 1) * 4) as usize..][..3], &[255; 3]);
    }
    #[test]
    fn video_portraits_clip_at_their_panel_boundary() {
        let mut canvas = Canvas::new([17, 23, 31]);
        canvas.ellipse_clipped([40., 55.], [29., 34.], [185, 112, 79], [20, 20, 40, 40]);
        for y in 0..N {
            for x in 0..N {
                let i = ((y * N + x) * 4) as usize;
                if !(20..60).contains(&x) || !(20..60).contains(&y) {
                    assert_eq!(&canvas.0[i..i + 3], &[17, 23, 31]);
                }
                assert_eq!(canvas.0[i + 3], 255);
            }
        }
    }
}
