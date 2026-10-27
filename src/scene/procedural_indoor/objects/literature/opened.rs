//! Open, supported page blocks. Their height field is shared by the closed
//! paper volumes and markings, so writing follows the actual curved surface.
use super::*;

fn quad(g: &mut Geometry, p: [Vec3; 4]) {
    let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize();
    let first = g.positions.len() as u32;
    for v in p {
        g.positions.push(v.to_array());
        g.normals.push(n.to_array());
        g.uvs.push([v.x, v.z]);
    }
    g.indices
        .extend([first, first + 1, first + 2, first, first + 2, first + 3]);
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &BindingProgram) {
    let s = o.size * p.envelope_fraction;
    let label = o.kind.class_name();
    let cover = (s.y * p.cover_fraction).min(0.002);
    a.box_part(
        p.cover,
        label,
        Vec3::Y * cover * 0.5,
        Vec3::new(s.x, cover, s.z),
        cover * 0.2,
    );
    let depth = s.z * 0.5 - cover * 2.;
    let gutter = (s.x * 0.008).min(0.002);
    let width = (s.x * 0.5 - gutter - cover * 2.).max(0.01);
    for side in [-1., 1.] {
        let balance = if side < 0. {
            p.page_balance
        } else {
            1. - p.page_balance
        };
        let height = |t: f32| {
            cover + s.y * (0.10 + balance * 0.62 + p.page_crown * (PI * t).sin() * (1. - 0.4 * t))
        };
        let point = |t: f32, z: f32| Vec3::new(side * (gutter + width * t), height(t), z);
        let g = a.part(Surface::Paper, label);
        for i in 0..12 {
            let x = i as f32 / 12.;
            let y = (i + 1) as f32 / 12.;
            let mut top = [
                point(x, -depth),
                point(x, depth),
                point(y, depth),
                point(y, -depth),
            ];
            if side < 0. {
                top.reverse();
            }
            quad(g, top);
            let mut bottom = top.map(|v| Vec3::new(v.x, cover, v.z));
            bottom.reverse();
            quad(g, bottom);
            for z in [-depth, depth] {
                let mut edge = [
                    point(x, z),
                    point(y, z),
                    Vec3::new(point(y, z).x, cover, z),
                    Vec3::new(point(x, z).x, cover, z),
                ];
                if side * z > 0. {
                    edge.reverse();
                }
                quad(g, edge);
            }
        }
        for t in [0., 1.] {
            let mut edge = [
                point(t, -depth),
                Vec3::new(point(t, -depth).x, cover, -depth),
                Vec3::new(point(t, depth).x, cover, depth),
                point(t, depth),
            ];
            if side * (t - 0.5) > 0. {
                edge.reverse();
            }
            quad(g, edge);
        }
        let mut rng = stream(o.seed, 7521 + u64::from(side > 0.));
        for row in 0..rng.random_range(8..=17) {
            let z = -depth * 0.80 + row as f32 * depth * 0.083;
            let end = rng.random_range(0.40..0.86);
            // Fit strokes to each piecewise-linear paper segment. Raising them
            // slightly prevents coincident faces, without floating off the page.
            for i in 1..10 {
                let t = i as f32 / 12.;
                let u = ((i + 1) as f32 / 12.).min(end);
                if u <= t {
                    break;
                }
                let mut q = [
                    point(t, z),
                    point(t, z + 0.00035),
                    point(u, z + 0.00035),
                    point(u, z),
                ]
                .map(|v| v + Vec3::Y * 0.00018);
                if side < 0. {
                    q.reverse();
                }
                quad(a.part(Surface::Ink, label), q);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn open_notebooks_have_two_curved_page_blocks_with_support_and_reserved_bounds() {
        let mut states = [0; 2];
        for seed in 0..256 {
            let o = super::super::super::audit::fixture(ObjectKind::Notebook, seed);
            let p = parameters(&o);
            states[usize::from(p.opened)] += 1;
            let a = super::super::super::build_object(&o);
            super::super::super::audit::check(&o, &a).unwrap();
            assert!(a.bounds().0.y.abs() < 0.0001);
            if p.opened {
                let paper = &a.parts[&(Surface::Paper, o.kind.class_name().into())];
                assert!(paper.positions.iter().any(|v| v[0] < -0.04));
                assert!(paper.positions.iter().any(|v| v[0] > 0.04));
                assert!(a
                    .parts
                    .contains_key(&(Surface::Ink, o.kind.class_name().into())));
            }
        }
        assert!(states.into_iter().all(|n| n > 64));
    }
}
