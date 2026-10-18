//! Frozen scalar coverage/opaque-blend oracle. Compare every RGBA byte,
//! including clipped, subpixel, empty and edge-touching paths.
use super::*;
fn ellipse_reference(
    canvas: &mut Canvas,
    center: [f32; 2],
    radius: [f32; 2],
    color: [u8; 3],
    bounds: [i32; 4],
) {
    let [left, top, width, height] = bounds;
    for y in ((center[1] - radius[1]).floor() as i32).max(top).max(0)
        ..=((center[1] + radius[1]).ceil() as i32)
            .min(top + height - 1)
            .min(N - 1)
    {
        for x in ((center[0] - radius[0]).floor() as i32).max(left).max(0)
            ..=((center[0] + radius[0]).ceil() as i32)
                .min(left + width - 1)
                .min(N - 1)
        {
            if ((x as f32 - center[0]) / radius[0].max(0.1)).powi(2)
                + ((y as f32 - center[1]) / radius[1].max(0.1)).powi(2)
                <= 1.
            {
                canvas.blend(x, y, color, 1.);
            }
        }
    }
}
fn rect_reference(canvas: &mut Canvas, x: i32, y: i32, w: i32, h: i32, color: [u8; 3]) {
    for y in y.max(canvas.1[1])..(y + h).min(canvas.1[3]) {
        for x in x.max(canvas.1[0])..(x + w).min(canvas.1[2]) {
            let i = ((y * N + x) * 4) as usize;
            canvas.0[i..i + 3].copy_from_slice(&color);
        }
    }
}

#[test]
fn raster_fast_paths_preserve_original_coverage_and_alpha() {
    for case in 0..96_i32 {
        let background = [case as u8, 37, 239];
        let mut actual = Canvas::new(background);
        let mut expected = Canvas(vec![255; (N * N * 4) as usize], [0, 0, N, N]);
        rect_reference(&mut expected, 0, 0, N, N, background);
        assert_eq!(actual.0, expected.0);
        for (i, pixel) in actual.0.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            pixel[3] = i as u8;
        }
        expected.0.clone_from(&actual.0);
        let clips = [
            [0, 0, N, N],
            [45, 39, 130, 71],
            [-20, 140, 83, 180],
            [300, 300, 20, 20],
            [80, 90, 0, 30],
            [120, 0, -40, 70],
        ];
        for clip in clips {
            let center = [
                (case * 31 % 330 - 35) as f32 + 0.25,
                (case * 71 % 330 - 35) as f32 - 0.5,
            ];
            let radius = match case % 8 {
                0 => [0., 0.],
                1 => [-0.2, 2.],
                2 => [0.01, 13.],
                _ => [
                    ((case * 3) % 150) as f32 + 0.17,
                    ((case * 7) % 130) as f32 + 0.41,
                ],
            };
            let bounds = [-13 + case, -17, 270 - case, 250];
            let color = [251, case as u8, 1];
            expected.clipped(clip, |c| {
                rect_reference(c, -30 + case, 11, 173, 43, color);
                ellipse_reference(c, center, radius, color, bounds);
                c.clipped([100, 53, 91, 37], |c| {
                    ellipse_reference(c, center, radius, [0; 3], bounds)
                });
            });
            actual.clipped(clip, |c| {
                c.rect(-30 + case, 11, 173, 43, color);
                c.ellipse_clipped(center, radius, color, bounds);
                c.clipped([100, 53, 91, 37], |c| {
                    c.ellipse_clipped(center, radius, [0; 3], bounds)
                });
            });
            assert_eq!(actual.0, expected.0, "case {case}, clip {clip:?}");
        }
    }
}
