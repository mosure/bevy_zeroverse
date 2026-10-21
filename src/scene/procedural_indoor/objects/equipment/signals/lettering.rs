//! Small continuous vector alphabet for manufactured fixture legends. Closed
//! outlines replace disconnected pixel boxes and require no image/font assets.
use super::*;
use crate::scene::procedural_indoor::envelope::polygon;

fn outlines(c: char) -> &'static [&'static [(f32, f32)]] {
    match c {
        'E' => &[&[
            (0., 0.),
            (1., 0.),
            (1., 0.18),
            (0.22, 0.18),
            (0.22, 0.41),
            (0.83, 0.41),
            (0.83, 0.59),
            (0.22, 0.59),
            (0.22, 0.82),
            (1., 0.82),
            (1., 1.),
            (0., 1.),
        ]],
        'F' => &[&[
            (0., 0.),
            (0.22, 0.),
            (0.22, 0.41),
            (0.83, 0.41),
            (0.83, 0.59),
            (0.22, 0.59),
            (0.22, 0.82),
            (1., 0.82),
            (1., 1.),
            (0., 1.),
        ]],
        'I' => &[&[
            (0., 0.),
            (1., 0.),
            (1., 0.18),
            (0.61, 0.18),
            (0.61, 0.82),
            (1., 0.82),
            (1., 1.),
            (0., 1.),
            (0., 0.82),
            (0.39, 0.82),
            (0.39, 0.18),
            (0., 0.18),
        ]],
        'T' => &[&[
            (0.39, 0.),
            (0.61, 0.),
            (0.61, 0.82),
            (1., 0.82),
            (1., 1.),
            (0., 1.),
            (0., 0.82),
            (0.39, 0.82),
        ]],
        'X' => &[&[
            (0., 1.),
            (0.28, 1.),
            (0.5, 0.66),
            (0.72, 1.),
            (1., 1.),
            (0.64, 0.5),
            (1., 0.),
            (0.72, 0.),
            (0.5, 0.34),
            (0.28, 0.),
            (0., 0.),
            (0.36, 0.5),
        ]],
        // Disjoint stroke interiors preserve the R counter without overlapping
        // coplanar front faces. Neighboring strokes meet at their exact edges.
        'P' => &[
            &[(0., 0.), (0.2, 0.), (0.2, 1.), (0., 1.)],
            &[(0.2, 0.82), (1., 0.82), (1., 1.), (0.2, 1.)],
            &[(0.8, 0.55), (1., 0.55), (1., 0.82), (0.8, 0.82)],
            &[(0.2, 0.4), (1., 0.4), (1., 0.55), (0.2, 0.55)],
        ],
        'L' => &[&[
            (0., 0.),
            (1., 0.),
            (1., 0.20),
            (0.22, 0.20),
            (0.22, 1.),
            (0., 1.),
        ]],
        'U' => &[&[
            (0., 1.),
            (0.22, 1.),
            (0.22, 0.2),
            (0.78, 0.2),
            (0.78, 1.),
            (1., 1.),
            (1., 0.),
            (0., 0.),
        ]],
        'R' => &[
            &[(0., 0.), (0.2, 0.), (0.2, 1.), (0., 1.)],
            &[(0.2, 0.82), (1., 0.82), (1., 1.), (0.2, 1.)],
            &[(0.8, 0.55), (1., 0.55), (1., 0.82), (0.8, 0.82)],
            &[(0.2, 0.4), (1., 0.4), (1., 0.55), (0.2, 0.55)],
            &[(0.35, 0.4), (0.6, 0.4), (1., 0.), (0.7, 0.)],
        ],
        _ => &[],
    }
}

pub(super) fn build(
    a: &mut Assembly,
    label: &str,
    word: &str,
    size: Vec2,
    center: Vec3,
    surface: Surface,
) {
    let step = size.x / (word.len() * 4 - 1) as f32;
    let g = a.part(surface, label);
    for (letter, c) in word.chars().enumerate() {
        for shape in outlines(c) {
            let mut outline: Vec<Vec2> = shape
                .iter()
                .map(|&(x, y)| {
                    Vec2::new(
                        -size.x * 0.5 + (letter as f32 * 4. + x * 3.) * step,
                        (y - 0.5) * size.y,
                    )
                })
                .collect();
            if polygon::area(&outline) < 0. {
                outline.reverse();
            }
            let vertex = |g: &mut Geometry, p: Vec3, n: Vec3| {
                g.positions.push((p + center).to_array());
                g.normals.push(n.to_array());
                g.uvs.push(p.xy().to_array());
            };
            for mut tri in polygon::triangles(&outline) {
                for sign in [1_f32, -1.] {
                    if sign < 0. {
                        tri.swap(1, 2);
                    }
                    let start = g.positions.len() as u32;
                    for p in tri {
                        vertex(g, p.extend(sign * 0.0004), Vec3::Z * sign);
                    }
                    g.indices.extend([start, start + 1, start + 2]);
                }
            }
            for (p, q) in polygon::edges(&outline) {
                let d = q - p;
                let n = Vec3::new(d.y, -d.x, 0.).normalize();
                let start = g.positions.len() as u32;
                for v in [
                    p.extend(-0.0004),
                    q.extend(-0.0004),
                    q.extend(0.0004),
                    p.extend(0.0004),
                ] {
                    vertex(g, v, n);
                }
                g.indices
                    .extend([start, start + 1, start + 2, start, start + 2, start + 3]);
            }
        }
    }
}
