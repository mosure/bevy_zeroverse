//! Folded/swivelled ear cups and a padded headband in supported resting poses.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct HeadphoneProgram {
    pub flat: bool,
    pub span: f32,
    pub rise: f32,
    pub cup_size: Vec3,
    pub cup_turn: [f32; 2],
    pub band_width: f32,
    pub band_thickness: f32,
    pub padding: f32,
    pub pad_opening: f32,
    pub roundness: f32,
    pub finish: Surface,
    pub cushion: Surface,
    pub resting_rotation: Quat,
}
pub fn parameters(o: &IndoorObject) -> HeadphoneProgram {
    let mut r = stream(o.seed, 0x4845414450484f4e);
    let flat = o.size.y < o.size.z * 0.70;
    let span = o.size.x * r.random_range(0.56..0.72);
    let rise = if flat { o.size.z } else { o.size.y } * r.random_range(0.56..0.72);
    let depth = if flat { o.size.y } else { o.size.z };
    let cup_size = Vec3::new(
        span * r.random_range(0.17..0.26),
        rise * r.random_range(0.53..0.82),
        depth * r.random_range(0.68..0.91),
    );
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x4845414446494e49);
    HeadphoneProgram {
        flat,
        span,
        rise,
        cup_size,
        // Flat storage swivels both backs onto the support plane. Upright cups
        // retain independent swivel but equal bottom heights. Arbitrary global
        // roll/pitch would leave these rigid objects balanced on one point.
        cup_turn: if flat {
            [-FRAC_PI_2; 2]
        } else {
            std::array::from_fn(|_| r.random_range(-0.35..0.65))
        },
        band_width: depth * r.random_range(0.18..0.31),
        band_thickness: span * r.random_range(0.018..0.033),
        padding: span * r.random_range(0.035..0.065),
        pad_opening: r.random_range(0.45..0.64),
        roundness: r.random_range(2.0..4.0),
        finish: if f.random_bool(0.82) {
            Surface::Plastic
        } else {
            Surface::Metal
        },
        cushion: if f.random_bool(0.58) {
            Surface::Leather
        } else {
            Surface::FabricAlt
        },
        resting_rotation: Quat::from_rotation_y(r.random_range(-0.35..0.35))
            * Quat::from_rotation_x(if flat { FRAC_PI_2 } else { 0. }),
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let l = o.kind.class_name();
    let mut child = Assembly::default();
    band(
        child.part(p.finish, l),
        &p,
        0.,
        std::f32::consts::PI,
        0.,
        p.band_width,
        p.band_thickness,
    );
    band(
        child.part(p.cushion, l),
        &p,
        0.24 * std::f32::consts::PI,
        0.76 * std::f32::consts::PI,
        -p.band_thickness * 0.80,
        p.band_width * 0.86,
        p.band_thickness * 0.60,
    );
    for (i, side) in [-1., 1.].into_iter().enumerate() {
        let center = Vec3::new(side * p.span * 0.5, 0., 0.);
        let turn = Quat::from_rotation_y(side * p.cup_turn[i]);
        let axis = turn * Vec3::X * side;
        let shell_rotation = turn * Quat::from_rotation_z(-side * FRAC_PI_2);
        let tf = Transform::from_translation(center).with_rotation(shell_rotation);
        let half = p.cup_size.yz() * 0.5;
        let h = p.cup_size.x;
        child.part(p.finish, l).housing(
            &[
                (-h * 0.5, half * 0.94),
                (-h * 0.24, half),
                (h * 0.29, half * 0.96),
                (h * 0.5, half * 0.80),
            ],
            p.roundness,
            tf,
        );
        let pad = Transform::from_translation(center - axis * h * 0.50)
            .with_rotation(turn * Quat::from_rotation_z(side * FRAC_PI_2))
            .with_scale(Vec3::new(half.x, 1., half.y));
        child.part(p.cushion, l).lathe(
            &[
                (1., 0.),
                (1., p.padding * 0.65),
                (0.91, p.padding),
                (p.pad_opening, p.padding),
                (p.pad_opening, 0.),
                (1., 0.),
            ],
            32,
            pad,
        );
        child.part(Surface::Fabric, l).housing(
            &[
                (-h * 0.51, half * p.pad_opening * 0.96),
                (-h * 0.50, half * p.pad_opening * 0.96),
            ],
            p.roundness,
            tf,
        );
    }
    // Fit the sampled rigid resting pose uniformly, then ground its actual
    // lowest vertex. No anisotropic deformation or post-placement size growth.
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for g in child.parts.values_mut() {
        for v in &mut g.positions {
            let p = p.resting_rotation * Vec3::from_array(*v);
            lo = lo.min(p);
            hi = hi.max(p);
            *v = p.to_array();
        }
        for n in &mut g.normals {
            *n = (p.resting_rotation * Vec3::from_array(*n)).to_array();
        }
    }
    let scale = (o.size / (hi - lo)).min_element() * 0.99;
    let offset = Vec3::new(-(lo.x + hi.x) * 0.5, -lo.y, -(lo.z + hi.z) * 0.5);
    for ((s, label), mut g) in child.parts {
        for v in &mut g.positions {
            *v = ((Vec3::from_array(*v) + offset) * scale).to_array();
        }
        for uv in &mut g.uvs {
            *uv = (Vec2::from_array(*uv) * scale).to_array();
        }
        a.part(s, &label).append(g);
    }
}

// Closed, shared-edge ribbon surfaces avoid overlapping box segments and their
// coplanar front/back faces. Cross-section UVs remain continuous along the arc.
fn band(
    g: &mut Geometry,
    p: &HeadphoneProgram,
    start: f32,
    end: f32,
    offset: f32,
    width: f32,
    thickness: f32,
) {
    let rings: Vec<_> = (0..=24)
        .map(|i| {
            let t = start + (end - start) * i as f32 / 24.;
            let normal = Vec3::new(t.cos() / (p.span * 0.5), t.sin() / p.rise, 0.).normalize();
            let center = Vec3::new(p.span * 0.5 * t.cos(), p.rise * t.sin(), 0.) + normal * offset;
            let n = normal * thickness * 0.5;
            let z = Vec3::Z * width * 0.5;
            (
                center,
                [
                    center + n + z,
                    center - n + z,
                    center - n - z,
                    center + n - z,
                ],
            )
        })
        .collect();
    let mut along = 0.;
    for pair in rings.windows(2) {
        let distance = pair[1].0.distance(pair[0].0);
        let mut across = 0.;
        for j in 0..4 {
            let k = (j + 1) % 4;
            let edge = pair[0].1[k].distance(pair[0].1[j]);
            let points = [pair[0].1[j], pair[1].1[j], pair[1].1[k], pair[0].1[k]];
            let expected = points.iter().sum::<Vec3>() * 0.25 - (pair[0].0 + pair[1].0) * 0.5;
            quad(
                g,
                points,
                [
                    [along, across],
                    [along + distance, across],
                    [along + distance, across + edge],
                    [along, across + edge],
                ],
                expected,
            );
            across += edge;
        }
        along += distance;
    }
    for (i, neighbor) in [(0, 1), (24, 23)] {
        quad(
            g,
            rings[i].1,
            [[0., 0.], [thickness, 0.], [thickness, width], [0., width]],
            rings[i].0 - rings[neighbor].0,
        );
    }
}
fn quad(g: &mut Geometry, mut points: [Vec3; 4], mut uvs: [[f32; 2]; 4], expected: Vec3) {
    let mut n = (points[1] - points[0])
        .cross(points[2] - points[0])
        .normalize();
    if n.dot(expected) < 0. {
        points.swap(1, 3);
        uvs.swap(1, 3);
        n = -n;
    }
    let offset = g.positions.len() as u32;
    for (p, uv) in points.into_iter().zip(uvs) {
        g.positions.push(p.to_array());
        g.normals.push(n.to_array());
        g.uvs.push(uv);
    }
    g.indices.extend([
        offset,
        offset + 1,
        offset + 2,
        offset,
        offset + 2,
        offset + 3,
    ]);
}
