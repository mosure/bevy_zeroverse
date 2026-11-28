use super::*;
use crate::scene::procedural_indoor::geometry::CushionProfile;

pub(super) fn seat(o: &IndoorObject, p: BeamSeating, version: u32) -> Assembly {
    let mut seat = Assembly::default();
    let label = format!("{LABEL}#finish{}#uv_local", p.finish_slot);
    let h = p.seat_height_m;
    let g = seat.part(p.facing, &label);
    let upholstered = p.stock == SeatStock::Upholstered;
    if upholstered {
        g.seat_panels(
            Vec3::new(p.seat_width_m, 0.068, o.size.z * 0.78),
            CushionProfile {
                roundness: 4.3,
                crown: 3.1,
                dish: 0.004,
            },
            Transform::from_xyz(0., h - 0.030, o.size.z * 0.074),
            None,
        );
        let bottom = h + 0.05;
        let tf = Transform::from_xyz(0., bottom, -o.size.z * 0.235)
            .with_rotation(Quat::from_rotation_y(std::f32::consts::PI));
        let height = o.size.y - bottom - 0.004;
        let thickness = 0.047;
        // The curved upholstery and its rear skin share the reserve with the recline.
        let rake = if version >= FITTED_BACK_VERSION {
            ((0.265 * o.size.z - p.wrap_m - thickness - 0.0005) / height).clamp(0., 0.15)
        } else {
            0.15
        };
        g.chair_back_contour(
            p.seat_width_m,
            height,
            [p.wrap_m, 0.11, rake, thickness, 0.012, 0.045],
            tf,
        );
        let v = 0.33;
        let wave = (std::f32::consts::PI * v).sin();
        let half = p.seat_width_m * 0.5 * (1. - 0.11 * v) * (1. + 0.045 * wave);
        for t in [-0.52, 0.52] {
            let z = p.wrap_m * (1. - t * t) + height * v * rake - 0.012 * wave + 0.030;
            let tip = tf.transform_point(Vec3::new(t * half, height * v, z));
            seat.part(Surface::Metal, LABEL).tapered_tube(
                &[
                    Vec3::new(tip.x, p.beam_y(), 0.),
                    Vec3::new(tip.x, p.beam_y() + 0.012, tip.z),
                    tip,
                ],
                0.016,
                0.013,
                8,
            );
        }
    } else {
        moulded(g, o.size, p);
    }
    let bearing = if upholstered {
        h - 0.059
    } else {
        h + p.wrap_m * 0.25 - p.shell_thickness_m * 0.60
    };
    for side in [-1., 1.] {
        seat.box_part(
            Surface::Metal,
            LABEL,
            Vec3::new(
                side * p.seat_width_m * 0.25,
                (p.beam_y() + bearing) * 0.5,
                0.,
            ),
            Vec3::new(0.040, bearing - p.beam_y(), 0.060),
            0.004,
        );
    }
    seat
}

/// Two skins follow a seat-to-back bend. Both share a metric chart, with a
/// closed rim and smooth normals at the lumbar bend and the rolled front.
fn moulded(g: &mut Geometry, size: Vec3, p: BeamSeating) {
    let h = p.seat_height_m;
    let rise = size.y - h;
    // (height, fraction of depth, fraction of width, transverse wrap).
    let sections = [
        (h - 0.052, 0.435, 0.85, 0.55),
        (h - 0.033, 0.447, 0.94, 0.64),
        (h - 0.007, 0.408, 1., 0.79),
        (h + 0.004, 0.234, 1., 0.92),
        (h, 0., 0.99, 1.),
        (h - 0.009, -0.150, 0.95, 1.),
        (h + 0.008, -0.228, 0.91, 0.98),
        (h + 0.044, -0.271, 0.89, 1.),
        (h + rise * 0.26, -0.288, 0.92, 1.08),
        (h + rise * 0.60, -0.345, 0.99, 1.18),
        (size.y - 0.054, -0.434, 0.94, 1.02),
        (size.y - 0.016, -0.450, 0.86, 0.96),
    ];
    let mut v = 0.;
    let mut distance = [0.; 12];
    let mut front = [[Vec3::ZERO; 9]; 12];
    for (row, &(y, z, width, wrap)) in sections.iter().enumerate() {
        let first = sections[row.saturating_sub(1)];
        let last = sections[(row + 1).min(sections.len() - 1)];
        let normal = Vec3::new(0., (first.1 - last.1) * size.z, last.0 - first.0).normalize();
        if row > 0 {
            v += Vec2::new(y - first.0, (z - first.1) * size.z).length();
        }
        distance[row] = v;
        for (col, point) in front[row].iter_mut().enumerate() {
            let t = col as f32 * 0.25 - 1.;
            *point = Vec3::new(t * p.seat_width_m * width * 0.5, y, z * size.z)
                + normal * (p.wrap_m * wrap * t * t)
                - Vec3::Y
                    * (if row == sections.len() - 1 {
                        0.007 * t * t
                    } else {
                        0.
                    });
        }
    }
    let mut back = front;
    let mut normals = [[Vec3::ZERO; 9]; 12];
    for (j, row) in front.iter().enumerate() {
        for (i, &point) in row.iter().enumerate() {
            let u = row[(i + 1).min(8)] - row[i.saturating_sub(1)];
            let v = front[(j + 1).min(11)][i] - front[j.saturating_sub(1)][i];
            let n = u.cross(v).normalize();
            normals[j][i] = n;
            // Translate the skin along the tooling axis. Both the seat pan
            // and its back see positive wall thickness, even at the nose.
            back[j][i] = point - Vec3::new(0., p.shell_thickness_m, p.shell_thickness_m);
        }
    }
    for (side, grid) in [(1., &front), (-1., &back)] {
        let start = g.positions.len() as u32;
        for (j, row) in grid.iter().enumerate() {
            for (i, point) in row.iter().enumerate() {
                g.positions.push(point.to_array());
                g.normals.push((side * normals[j][i]).to_array());
                g.uvs.push([point.x, distance[j]]);
            }
        }
        for j in 0..11 {
            for i in 0..8 {
                let a = start + j * 9 + i;
                if side > 0. {
                    g.indices.extend([a, a + 1, a + 10, a, a + 10, a + 9]);
                } else {
                    g.indices.extend([a, a + 10, a + 1, a, a + 9, a + 10]);
                }
            }
        }
    }
    // Traverse the boundary of the front skin and join its matching rear edge.
    let mut rim = |(j, i): (usize, usize), (k, l): (usize, usize)| {
        let edge = front[k][l] - front[j][i];
        g.metric_quad(
            [front[j][i], back[j][i], back[k][l], front[k][l]],
            edge.cross(normals[j][i]),
        );
    };
    for i in 0..8 {
        rim((0, i), (0, i + 1));
        rim((11, i + 1), (11, i));
    }
    for j in 0..11 {
        rim((j, 8), (j + 1, 8));
        rim((j + 1, 0), (j, 0));
    }
}
