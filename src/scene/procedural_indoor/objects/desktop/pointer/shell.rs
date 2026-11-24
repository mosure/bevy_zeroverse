//! The deck is a single annulus. Three shallow channels divide its front into
//! independently clickable paddles. Both the bezel and the skirt reuse its edges.
use super::*;
use crate::scene::procedural_indoor::geometry::housing_outline;

const N: usize = 32;
type Ring = [Vec3; N];

fn socket_ring(s: &Socket, outline: &[Vec2; N], half: Vec2, y: f32) -> Ring {
    outline.map(|v| {
        s.tf.transform_point(Vec3::new(v.x * half.x, y, v.y * half.y))
    })
}

fn band_normals(from: &Ring, to: &Ring) -> Ring {
    std::array::from_fn(|i| {
        let next = (i + 1) % N;
        let prev = (i + N - 1) % N;
        ((from[next] - from[prev]) + (to[next] - to[prev]))
            .cross(to[i] - from[i])
            .normalize()
    })
}

fn band(g: &mut Geometry, from: &Ring, to: &Ring) {
    let normals = band_normals(from, to);
    loft(g, from, to, &normals, &normals);
}

// Independent strip at material/crease boundaries. All faces within a strip
// share indices except the texture wrap, with metric UVs around the perimeter.
fn loft(g: &mut Geometry, from: &Ring, to: &Ring, upper_normals: &Ring, lower_normals: &Ring) {
    let start = g.positions.len() as u32;
    let mut distance = 0.;
    for i in 0..=N {
        let j = i % N;
        for (point, normal, height) in [
            (from[j], upper_normals[j], 0.),
            (to[j], lower_normals[j], from[j].distance(to[j])),
        ] {
            g.positions.push(point.to_array());
            g.normals.push(normal.to_array());
            g.uvs.push([distance, height]);
        }
        if i < N {
            let a = start + 2 * i as u32;
            g.indices.extend([a, a + 2, a + 3, a, a + 3, a + 1]);
            distance += from[j].distance(from[(j + 1) % N]);
        }
    }
}

fn disk(g: &mut Geometry, ring: &Ring, normal: Vec3, upward: bool) {
    let base = g.positions.len() as u32;
    let center = ring.iter().sum::<Vec3>() / (N as f32);
    for pt in std::iter::once(&center).chain(ring) {
        g.positions.push(pt.to_array());
        g.normals.push(normal.to_array());
        g.uvs.push(pt.xz().to_array());
    }
    for i in 0..N {
        let a = base + 1 + i as u32;
        let b = base + 1 + ((i + 1) % N) as u32;
        g.indices
            .extend(if upward { [base, b, a] } else { [base, a, b] });
    }
}

#[derive(Clone, Copy)]
struct Station {
    before: Vec3,
    after: Vec3,
    valley: Vec3,
    normal: Vec3,
}

fn deck(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &PointerProgram,
    stock: &Housing,
    inner: &Ring,
    outer: &Ring,
) {
    // Fit both shoulders to the actual projected socket, including its tilt.
    let row = |t: f32| -> Ring {
        std::array::from_fn(|i| {
            let mut v = inner[i].lerp(outer[i], t);
            let inner_error = inner[i].y - p.roof(o.size, inner[i].xz()).0;
            v.y = p.roof(o.size, v.xz()).0 + inner_error * (1. - t).powi(2);
            v
        })
    };
    let rows = [*inner, row(0.25), row(0.5), row(0.75), *outer];
    let seams = [24, p.seam_station, 16 - p.seam_station];
    let stations: [[Station; N]; 5] = std::array::from_fn(|r| {
        std::array::from_fn(|i| {
            let here = rows[r][i];
            let prev = rows[r][(i + N - 1) % N];
            let next = rows[r][(i + 1) % N];
            let normal = (next - prev)
                .cross(rows[(r + 1).min(4)][i] - rows[r.saturating_sub(1)][i])
                .normalize();
            let width = if seams.contains(&i) {
                p.seam_width_m * 0.5
            } else {
                0.
            };
            let towards = |v: Vec3| here + (v - here) * ((width / v.distance(here)).min(0.28));
            Station {
                before: towards(prev),
                after: towards(next),
                normal,
                valley: here - Vec3::Y * if r == 0 || r == 4 { 0. } else { width * 1.8 },
            }
        })
    });
    let case = stock.case_label(o);
    {
        let g = a.part(stock.surface, &case);
        for pair in stations.windows(2) {
            for i in 0..N {
                let j = (i + 1) % N;
                let [a, b, c, d] = [pair[0][i], pair[0][j], pair[1][j], pair[1][i]];
                let pts = [a.after, b.before, c.before, d.after];
                g.vertex_quad(
                    pts,
                    [a.normal, b.normal, c.normal, d.normal],
                    pts.map(|v| v.xz().to_array()),
                );
            }
        }
    }
    let g = a.part(Surface::Rubber, "mouse");
    for pair in stations.windows(2) {
        for i in seams {
            let (a, b) = (pair[0][i], pair[1][i]);
            for points in [
                [a.before, a.valley, b.valley, b.before],
                [a.valley, a.after, b.after, b.valley],
            ] {
                // At each end the channel fades into the rim. Its narrow,
                // twisted facets need both triangle normals for correct shading.
                let [p0, p1, p2, p3] = points;
                let n1 = (p1 - p0).cross(p2 - p0).normalize();
                let n2 = (p2 - p0).cross(p3 - p0).normalize();
                let mid = (n1 + n2).normalize();
                g.vertex_quad(
                    points,
                    [mid, n1, mid, n2],
                    points.map(|v| v.xz().to_array()),
                );
            }
        }
    }
}

pub(super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &PointerProgram,
    stock: &Housing,
    socket: &Socket,
) {
    let s = o.size;
    let outline = housing_outline(p.outline_power);
    let outer = outline.map(|v| {
        // A thumb ball needs a wider shoulder on its operating side.
        let thumb_shelf =
            matches!(p.control, Control::ThumbTrackball { .. }) && p.grip.side() * v.x < 0.;
        let waist = if thumb_shelf {
            0.
        } else {
            p.waist * (1. - v.y * v.y) * (0.58 - 0.38 * p.grip.side() * v.x)
        };
        let x = v.x * s.x * 0.464 * (1. - waist);
        let z = v.y * s.z * 0.463;
        Vec3::new(x, p.roof(s, Vec2::new(x, z)).0, z)
    });
    let aperture = housing_outline(socket.power);
    let inner = socket_ring(socket, &aperture, socket.half + Vec2::splat(socket.lip), 0.);
    deck(a, o, p, stock, &inner, &outer);
    let rim = socket_ring(socket, &aperture, socket.half, 0.);
    let trim = stock.trim_label(o);
    band(a.part(Surface::Plastic, &trim), &rim, &inner);
    // Give the moving control clearance down to its full diameter. A straight
    // taper from the lip to the equator would intersect a spherical trackball.
    let pocket = a.part(Surface::Rubber, "mouse");
    let r = p.radius_m;
    let envelope = |h: f32| {
        let radius = (r * r - (h + socket.buried).powi(2)).max(0.).sqrt();
        if p.control.ball().is_some() {
            Vec2::splat(radius + p.clearance_m)
        } else {
            Vec2::new(socket.half.x, radius + p.clearance_m * 1.5)
        }
    };
    let mut prior = rim;
    for f in [0.5, 1.] {
        let y = -socket.buried * f;
        let ring = socket_ring(socket, &aperture, envelope(y), y);
        band(pocket, &ring, &prior);
        prior = ring;
    }
    // Close the cup against a horizontal internal floor. A cap parallel to
    // the tilted socket can protrude through the chassis on a low desk mouse.
    let up = socket.tf.rotation * Vec3::Y;
    let center = socket.tf.translation - up * socket.buried;
    let floor_height = (center.y - r - p.clearance_m).max(0.00145);
    let floor = prior.map(|v| v - up * ((v.y - floor_height) / up.y));
    band(pocket, &floor, &prior);
    disk(pocket, &floor, Vec3::Y, true);
    // The rounded palm rolls into the sidewall. Carry the deck slope over
    // this shoulder to keep the reflection continuous under grazing light.
    let base = s.y * p.base_height;
    let shoulder = outer.map(|v| Vec3::new(v.x * 1.044, v.y - (v.y - base) * 0.11, v.z * 1.039));
    let flank = outer.map(|v| Vec3::new(v.x * 1.065, v.y - (v.y - base) * 0.32, v.z * 1.065));
    let waist = outer.map(|v| Vec3::new(v.x, base, v.z));
    let sole = outer.map(|v| Vec3::new(v.x * 0.89, 0.0014, v.z * 0.90));
    let top_normals = outer.map(|v| {
        let (_, slope) = p.roof(s, v.xz());
        Vec3::new(-slope.x, 1., -slope.y).normalize()
    });
    let side_normals = band_normals(&flank, &waist);
    let shoulder_normals = band_normals(&outer, &flank);
    let flank_normals = band_normals(&shoulder, &waist);
    let case = stock.case_label(o);
    let g = a.part(stock.surface, &case);
    loft(g, &outer, &shoulder, &top_normals, &shoulder_normals);
    loft(g, &shoulder, &flank, &shoulder_normals, &flank_normals);
    loft(g, &flank, &waist, &flank_normals, &side_normals);
    let g = a.part(Surface::Plastic, &trim);
    band(g, &waist, &sole);
    disk(g, &sole, Vec3::NEG_Y, false);
    // Low-friction polymer feet use two broad contact patches. These small
    // rounded pieces share the bezel stock and need no extra material batch.
    let rx = s.x * 0.245;
    let rz = s.z * 0.055;
    for z in [-0.24, 0.24] {
        g.lathe(
            &[
                (0., 0.),
                (rx * 0.91, 0.),
                (rx, 0.0005),
                (rx * 0.94, 0.0016),
                (0., 0.0016),
            ],
            12,
            Transform::from_xyz(0., 0., s.z * z).with_scale(Vec3::new(1., 1., rz / rx)),
        );
    }
}
