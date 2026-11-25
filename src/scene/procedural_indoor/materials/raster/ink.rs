use super::Canvas;

impl Canvas {
    /// Round-ended ink stroke with a one-texel coverage transition. Stepping
    /// the major axis bounds the work by stroke length times brush width.
    pub(in super::super) fn ink_stroke(
        &mut self,
        a: [f32; 2],
        b: [f32; 2],
        width: i32,
        color: [u8; 3],
    ) {
        if width <= 0 {
            return;
        }
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let u = usize::from(dy.abs() > dx.abs());
        let v = 1 - u;
        let (a, b) = if a[u] <= b[u] { (a, b) } else { (b, a) };
        let du = b[u] - a[u];
        let dv = b[v] - a[v];
        // Degenerate segments paint a single round dab.
        let inv_len_sq = 1. / (du * du + dv * dv).max(f32::EPSILON);
        let slope = dv / du.max(f32::EPSILON);
        let reach = (width as f32 + 1.) * 0.5;
        let across = reach * std::f32::consts::SQRT_2;
        let bounds = self.1;
        let first = ((a[u] - reach).ceil() as i32).max(bounds[u]);
        let last = ((b[u] + reach).floor() as i32).min(bounds[u + 2] - 1);
        for i in first..=last {
            let offset = i as f32 - a[u];
            let center = a[v] + offset.clamp(0., du) * slope;
            let low = ((center - across).ceil() as i32).max(bounds[v]);
            let high = ((center + across).floor() as i32).min(bounds[v + 2] - 1);
            for j in low..=high {
                let lateral = j as f32 - a[v];
                let t = ((offset * du + lateral * dv) * inv_len_sq).clamp(0., 1.);
                let distance_sq = (offset - t * du).powi(2) + (lateral - t * dv).powi(2);
                if distance_sq < reach * reach {
                    let (x, y) = if u == 0 { (i, j) } else { (j, i) };
                    self.blend(x, y, color, reach - distance_sq.sqrt());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Canvas, N};

    #[test]
    fn fractional_pen_offsets_translate_coverage_and_respect_the_paper_clip() {
        let mut horizontal = Canvas::new([0; 3]);
        horizontal.clipped([8, 4, 12, 12], |c| {
            c.ink_stroke([4., 10.25], [24., 10.25], 1, [255; 3]);
        });
        let value = |c: &Canvas, x, y| c.0[((y * N + x) * 4) as usize];
        for x in 8..20 {
            assert_eq!(value(&horizontal, x, 10), 191);
            assert_eq!(value(&horizontal, x, 11), 64);
        }
        assert_eq!(value(&horizontal, 7, 10), 0);
        assert_eq!(value(&horizontal, 20, 10), 0);
        let mut vertical = Canvas::new([0; 3]);
        vertical.ink_stroke([10.25, 24.], [10.25, 4.], 1, [255; 3]);
        assert_eq!(value(&vertical, 10, 12), 191);
        assert_eq!(value(&vertical, 11, 12), 64);
        vertical.ink_stroke([32., 32.], [32., 32.], 1, [255; 3]);
        assert_eq!(value(&vertical, 32, 32), 255);
        assert_eq!(value(&vertical, 33, 32), 0);
    }
}
