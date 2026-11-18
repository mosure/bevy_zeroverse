//! Face replacement below a metric finish line. Each module is trimmed to the
//! actual window jambs and its grooves end flush at the sill, head and floor.
use super::*;
use crate::scene::procedural_indoor::{
    architecture::facade::WindowOpening, envelope::lining::WallLining,
};

struct Skin {
    board: Geometry,
    joint: Geometry,
    tf: Transform,
}

impl Skin {
    fn patch(&mut self, p: [Vec3; 4], offset: Vec2, joint: bool) {
        let g = if joint {
            &mut self.joint
        } else {
            &mut self.board
        };
        for ids in [[0, 1, 2], [0, 2, 3]] {
            let local = ids.map(|i| p[i]);
            let normal = (local[1] - local[0])
                .cross(local[2] - local[0])
                .normalize_or_zero();
            let points = local.map(|v| self.tf.transform_point(v));
            if normal == Vec3::ZERO
                || (points[1] - points[0])
                    .cross(points[2] - points[0])
                    .length_squared()
                    < 1e-14
            {
                continue;
            }
            let start = g.positions.len() as u32;
            let normal = (self.tf.rotation * normal).to_array();
            for (v, w) in local.into_iter().zip(points) {
                g.positions.push(w.to_array());
                g.normals.push(normal);
                g.uvs.push((Vec2::new(v.x, v.y) + offset).to_array());
            }
            g.indices.extend([start, start + 1, start + 2]);
        }
    }

    fn strip(&mut self, x: [f32; 2], z: [f32; 2], y: [f32; 2], offset: Vec2, joint: bool) {
        if y[1] <= y[0] {
            return;
        }
        let [left, right] = x;
        let mut band = |bottom: f32, top: f32, a: f32, b: f32| {
            self.patch(
                [
                    Vec3::new(left, bottom, z[0] * a),
                    Vec3::new(right, bottom, z[1] * a),
                    Vec3::new(right, top, z[1] * b),
                    Vec3::new(left, top, z[0] * b),
                ],
                offset,
                joint,
            );
        };
        if z == [0., 0.] {
            band(y[0], y[1], 1., 1.);
        } else {
            let end = ((y[1] - y[0]) * 0.5).min((-z[0]).max(-z[1]) * 1.5);
            band(y[0], y[0] + end, 0., 1.);
            if y[1] - end > y[0] + end {
                band(y[0] + end, y[1] - end, 1., 1.);
            }
            band(y[1] - end, y[1], 1., 0.);
        }
    }
}

pub(super) fn build(
    a: &mut Assembly,
    tf: Transform,
    s: &WallLining,
    span: f32,
    height: f32,
    holes: &[WindowOpening],
) {
    let (left, right) = (-span * 0.5, span * 0.5);
    let mut holes: Vec<_> = holes
        .iter()
        .filter(|h| h.max.y > 0. && h.min.y < height)
        .collect();
    holes.sort_unstable_by(|a, b| a.min.y.total_cmp(&b.min.y));
    let mut jambs = vec![left, right];
    for h in &holes {
        jambs.extend([h.min.x.clamp(left, right), h.max.x.clamp(left, right)]);
    }
    jambs.sort_unstable_by(f32::total_cmp);
    jambs.dedup();
    let mut skin = Skin {
        board: Geometry::default(),
        joint: Geometry::default(),
        tf,
    };
    let pitch = s.module_width_m;
    let origin = s.phase * pitch;
    let mut strip = |l: f32, r: f32, zl: f32, zr: f32, joint: bool| {
        if r <= l {
            return;
        }
        let module = (((l + r) * 0.5 - origin) / pitch).floor();
        // Match each cut of a panel through openings without continuing the same
        // patch of veneer/weave into its neighbour. Units remain metres.
        let offset = Vec2::new(module * 0.7548777, module * 0.5698403);
        for xs in jambs.windows(2) {
            let x = [l.max(xs[0]), r.min(xs[1])];
            if x[1] <= x[0] {
                continue;
            }
            let z = x.map(|v| zl + (zr - zl) * ((v - l) / (r - l)));
            let middle = (x[0] + x[1]) * 0.5;
            let mut bottom = 0_f32;
            for h in &holes {
                if middle > h.min.x && middle < h.max.x {
                    skin.strip(x, z, [bottom, h.min.y.min(height)], offset, joint);
                    bottom = bottom.max(h.max.y);
                }
            }
            skin.strip(x, z, [bottom, height], offset, joint);
        }
    };
    let half = s.joint_width_m * 0.5;
    let trough = half * 0.44;
    let d = -s.joint_depth_m;
    let mut last = left;
    for i in ((left - origin) / pitch).ceil() as i32..=((right - origin) / pitch).floor() as i32 {
        let seam = origin + i as f32 * pitch;
        // A field cut tight against a reveal terminates flush with its jamb.
        if jambs.iter().any(|&x| (x - seam).abs() <= half + 0.002) {
            continue;
        }
        strip(last, seam - half, 0., 0., false);
        strip(seam - half, seam - trough, 0., d, false);
        strip(seam - trough, seam + trough, d, d, true);
        strip(seam + trough, seam + half, d, 0., false);
        last = seam + half;
    }
    strip(last, right, 0., 0., false);
    let label = format!("wall#finish{}", s.finish_slot);
    a.part(s.stock.surface(), &label).append(skin.board);
    a.part(Surface::Paint, "wall").append(skin.joint);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::envelope::{lining::LiningStock, tests::ray_triangle};

    #[test]
    fn ray_coverage_preserves_apertures_and_signed_relief_on_oblique_walls() {
        let holes = [
            WindowOpening {
                glazing: None,
                min: Vec2::new(-1.4, 0.45),
                max: Vec2::new(-0.2, 1.25),
                columns: 1,
                transom: 0.,
                transoms: vec![],
                operable: false,
            },
            WindowOpening {
                glazing: None,
                min: Vec2::new(0.65, 0.),
                max: Vec2::new(1.35, 2.),
                columns: 1,
                transom: 0.,
                transoms: vec![],
                operable: false,
            },
        ];
        let s = WallLining {
            stock: LiningStock::Veneer,
            height_m: 1.4,
            module_width_m: 0.58,
            phase: 0.32,
            joint_width_m: 0.012,
            joint_depth_m: 0.005,
            finish_slot: 3,
        };
        for angle in [0., 1.13, 2.68] {
            let tf = Transform::from_xyz(7., 0., -5.).with_rotation(Quat::from_rotation_y(angle));
            let n = tf.rotation * Vec3::Z;
            let mut a = Assembly::default();
            build(&mut a, tf, &s, 5., s.height_m, &holes);
            let mut triangles = vec![];
            let mut area = 0.;
            for ((_, label), g) in &a.parts {
                assert!(label == "wall" || label == "wall#finish3");
                assert!(g.uvs.iter().flatten().all(|u| u.is_finite()));
                for t in g.indices.as_chunks::<3>().0 {
                    let p = t.map(|i| Vec3::from(g.positions[i as usize]));
                    let cross = (p[1] - p[0]).cross(p[2] - p[0]);
                    assert!(cross.dot(n) > 0.);
                    assert!(cross.normalize().dot(Vec3::from(g.normals[t[0] as usize])) > 0.995);
                    area += cross.dot(n) * 0.5;
                    triangles.push(p);
                }
            }
            let expected_area = 5. * s.height_m - holes[0].area() - (1.35 - 0.65) * s.height_m;
            assert!((area - expected_area).abs() < 2e-4);
            let hit = |x, y| {
                let p = tf.transform_point(Vec3::new(x, y, 0.15));
                triangles
                    .iter()
                    .filter_map(|v| ray_triangle(p, -n, *v))
                    .collect::<Vec<_>>()
            };
            for j in 0..31 {
                for i in 0..101 {
                    let x = -2.5 + 5. * (i as f32 + 0.359) / 101.;
                    let y = 1.4 * (j as f32 + 0.171) / 31.;
                    let pt = Vec2::new(x, y);
                    let empty = holes
                        .iter()
                        .any(|h| pt.cmpgt(h.min).all() && pt.cmplt(h.max).all());
                    let hits = hit(x, y);
                    assert_eq!(hits.len(), usize::from(!empty));
                    assert!(hits.iter().all(|t| (0.14999..=0.15501).contains(t)));
                }
            }
            let seam = s.phase * s.module_width_m;
            assert!((hit(seam, 0.70)[0] - 0.155).abs() < 1e-5);
            assert!((hit(seam + 0.04, 0.70)[0] - 0.15).abs() < 1e-5);
        }
    }
}
