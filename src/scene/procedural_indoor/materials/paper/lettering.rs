//! Compact, proportional line lettering in units of one tenth of the cap height.
//! Pen lifts separate strokes. The same outlines rotate onto a narrow spine.
use super::*;

type Point = (i8, i8);
const LIFT: Point = (-1, -1);

fn glyph(ch: u8) -> (u8, &'static [Point]) {
    match ch {
        b'0' => (
            8,
            &[
                (3, 0),
                (5, 0),
                (7, 2),
                (8, 5),
                (7, 8),
                (5, 10),
                (3, 10),
                (1, 8),
                (0, 5),
                (1, 2),
                (3, 0),
            ],
        ),
        b'1' => (5, &[(0, 2), (3, 0), (3, 10), LIFT, (0, 10), (5, 10)]),
        b'2' => (
            7,
            &[(0, 2), (2, 0), (5, 0), (7, 2), (7, 4), (0, 10), (7, 10)],
        ),
        b'3' => (
            7,
            &[
                (0, 1),
                (2, 0),
                (5, 0),
                (7, 2),
                (6, 4),
                (3, 5),
                (6, 5),
                (7, 7),
                (7, 8),
                (5, 10),
                (2, 10),
                (0, 9),
            ],
        ),
        b'4' => (8, &[(6, 10), (6, 0), (0, 7), (8, 7)]),
        b'5' => (
            7,
            &[
                (7, 0),
                (1, 0),
                (0, 5),
                (3, 4),
                (5, 4),
                (7, 6),
                (7, 8),
                (5, 10),
                (2, 10),
                (0, 9),
            ],
        ),
        b'6' => (
            7,
            &[
                (6, 0),
                (4, 0),
                (1, 3),
                (0, 6),
                (1, 9),
                (3, 10),
                (5, 10),
                (7, 8),
                (7, 6),
                (5, 4),
                (3, 4),
                (0, 6),
            ],
        ),
        b'7' => (7, &[(0, 0), (7, 0), (2, 10)]),
        b'8' => (
            7,
            &[
                (2, 5),
                (0, 3),
                (1, 1),
                (3, 0),
                (5, 0),
                (7, 2),
                (6, 4),
                (2, 6),
                (0, 8),
                (2, 10),
                (5, 10),
                (7, 8),
                (6, 6),
                (2, 5),
            ],
        ),
        b'9' => (
            7,
            &[
                (1, 10),
                (3, 10),
                (6, 7),
                (7, 4),
                (6, 1),
                (4, 0),
                (2, 0),
                (0, 2),
                (0, 4),
                (2, 6),
                (4, 6),
                (7, 4),
            ],
        ),
        b'A' => (8, &[(0, 10), (4, 0), (8, 10), LIFT, (2, 6), (6, 6)]),
        b'B' => (
            7,
            &[
                (0, 10),
                (0, 0),
                (4, 0),
                (6, 1),
                (7, 3),
                (6, 4),
                (4, 5),
                (0, 5),
                LIFT,
                (4, 5),
                (6, 6),
                (7, 8),
                (6, 10),
                (0, 10),
            ],
        ),
        b'C' => (
            8,
            &[
                (8, 2),
                (6, 0),
                (3, 0),
                (1, 2),
                (0, 5),
                (1, 8),
                (3, 10),
                (6, 10),
                (8, 8),
            ],
        ),
        b'D' => (
            8,
            &[
                (0, 10),
                (0, 0),
                (4, 0),
                (6, 1),
                (8, 4),
                (8, 6),
                (6, 9),
                (4, 10),
                (0, 10),
            ],
        ),
        b'E' => (6, &[(6, 0), (0, 0), (0, 10), (6, 10), LIFT, (0, 5), (5, 5)]),
        b'F' => (6, &[(0, 10), (0, 0), (6, 0), LIFT, (0, 5), (5, 5)]),
        b'G' => (
            8,
            &[
                (8, 2),
                (6, 0),
                (3, 0),
                (1, 2),
                (0, 5),
                (1, 8),
                (3, 10),
                (6, 10),
                (8, 9),
                (8, 6),
                (5, 6),
            ],
        ),
        b'H' => (
            7,
            &[(0, 0), (0, 10), LIFT, (7, 0), (7, 10), LIFT, (0, 5), (7, 5)],
        ),
        b'I' => (1, &[(0, 0), (0, 10)]),
        b'J' => (6, &[(6, 0), (6, 7), (5, 9), (3, 10), (1, 10), (0, 8)]),
        b'K' => (
            7,
            &[(0, 0), (0, 10), LIFT, (7, 0), (0, 6), LIFT, (3, 4), (7, 10)],
        ),
        b'L' => (6, &[(0, 0), (0, 10), (6, 10)]),
        b'M' => (10, &[(0, 10), (0, 0), (5, 7), (10, 0), (10, 10)]),
        b'N' => (8, &[(0, 10), (0, 0), (8, 10), (8, 0)]),
        b'O' => (
            8,
            &[
                (3, 0),
                (5, 0),
                (7, 1),
                (8, 4),
                (8, 6),
                (7, 9),
                (5, 10),
                (3, 10),
                (1, 9),
                (0, 6),
                (0, 4),
                (1, 1),
                (3, 0),
            ],
        ),
        b'P' => (
            7,
            &[
                (0, 10),
                (0, 0),
                (4, 0),
                (6, 1),
                (7, 3),
                (6, 5),
                (4, 6),
                (0, 6),
            ],
        ),
        b'Q' => (
            8,
            &[
                (3, 0),
                (5, 0),
                (7, 1),
                (8, 4),
                (8, 6),
                (7, 9),
                (5, 10),
                (3, 10),
                (1, 9),
                (0, 6),
                (0, 4),
                (1, 1),
                (3, 0),
                LIFT,
                (5, 7),
                (9, 11),
            ],
        ),
        b'R' => (
            7,
            &[
                (0, 10),
                (0, 0),
                (4, 0),
                (6, 1),
                (7, 3),
                (6, 5),
                (4, 6),
                (0, 6),
                LIFT,
                (4, 6),
                (7, 10),
            ],
        ),
        b'S' => (
            7,
            &[
                (7, 2),
                (5, 0),
                (2, 0),
                (0, 2),
                (0, 3),
                (2, 5),
                (5, 6),
                (7, 7),
                (7, 8),
                (5, 10),
                (2, 10),
                (0, 8),
            ],
        ),
        b'T' => (8, &[(0, 0), (8, 0), LIFT, (4, 0), (4, 10)]),
        b'U' => (
            8,
            &[
                (0, 0),
                (0, 6),
                (1, 9),
                (3, 10),
                (5, 10),
                (7, 9),
                (8, 6),
                (8, 0),
            ],
        ),
        b'V' => (8, &[(0, 0), (4, 10), (8, 0)]),
        b'W' => (12, &[(0, 0), (3, 10), (6, 3), (9, 10), (12, 0)]),
        b'X' => (8, &[(0, 0), (8, 10), LIFT, (8, 0), (0, 10)]),
        b'Y' => (8, &[(0, 0), (4, 5), (8, 0), LIFT, (4, 5), (4, 10)]),
        b'Z' => (7, &[(0, 0), (7, 0), (0, 10), (7, 10)]),
        _ => (4, &[]),
    }
}

pub(super) struct Type {
    advance: f32,
    height: f32,
    tracking: f32,
    stroke: i32,
}

impl Type {
    pub(super) fn fit(lines: &[&str], width: f32, height: f32, tracking: f32, stroke: i32) -> Self {
        let longest = lines
            .iter()
            .map(|line| {
                line.bytes()
                    .map(|c| f32::from(glyph(c).0) + tracking)
                    .sum::<f32>()
            })
            .fold(1., f32::max);
        Self {
            advance: ((width - stroke as f32) / longest).min(height * 0.095),
            height: height * 0.1,
            tracking,
            stroke,
        }
    }

    /// Keep lettering proportional when a square texture is printed on rectangular stock.
    pub(super) fn on_sheet(line: &str, width: f32, height: f32, aspect: f32, stroke: i32) -> Self {
        let mut ty = Self::fit(&[line], width * aspect, height, 2.4, stroke);
        ty.advance /= aspect;
        ty
    }

    pub(super) fn draw(&self, c: &mut Canvas, line: &str, origin: Vec2, spine: bool, ink: [u8; 3]) {
        let (along, down) = if spine {
            (Vec2::Y, Vec2::NEG_X)
        } else {
            (Vec2::X, Vec2::Y)
        };
        let mut pen = origin;
        for ch in line.bytes() {
            let (width, points) = glyph(ch);
            for segment in points.windows(2) {
                if segment.contains(&LIFT) {
                    continue;
                }
                let point = |(x, y): Point| {
                    (pen + along * (f32::from(x) * self.advance)
                        + down * (f32::from(y) * self.height))
                        .to_array()
                };
                c.ink_stroke(point(segment[0]), point(segment[1]), self.stroke, ink);
            }
            pen += along * ((f32::from(width) + self.tracking) * self.advance);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn curved_letters_have_partial_coverage_and_rotate_onto_the_spine() {
        let t = Type::fit(&["GUIDE"], 100., 16., 2., 1);
        let mut upright = Canvas::new([0; 3]);
        let mut turned = Canvas::new([0; 3]);
        t.draw(&mut upright, "GUIDE", Vec2::new(20., 30.), false, [240; 3]);
        t.draw(&mut turned, "GUIDE", Vec2::new(130., 20.), true, [240; 3]);
        let mut soft = 0;
        for y in 28..49 {
            for x in 18..120 {
                let a = upright.0[(y * 256 + x) * 4];
                let b = turned.0[(x * 256 + 160 - y) * 4];
                assert!(a.abs_diff(b) <= 1);
                soft += usize::from((1..239).contains(&a));
            }
        }
        assert!(soft > 80);
    }
}
