//! Open, supported page blocks. Their height field is shared by the closed
//! paper volumes and markings, so writing follows the actual curved surface.
use super::*;

fn quad(g: &mut Geometry, p: [Vec3; 4]) {
    // A sampled ink endpoint can exceed a page segment boundary in parameter
    // space yet round to the same metric position in f32. Do not normalize or
    // emit that collapsed sliver. Check both triangles before adding attributes;
    // real page geometry and every representable stroke retain their winding.
    let cross = (p[1] - p[0]).cross(p[2] - p[0]);
    let other = (p[2] - p[0]).cross(p[3] - p[0]);
    if !cross.is_finite()
        || !other.is_finite()
        || cross.length_squared() < 1e-20
        || other.length_squared() < 1e-20
    {
        return;
    }
    let n = cross.normalize();
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
    fn rounded_ink_endpoint_does_not_emit_invalid_geometry() {
        let mut g = Geometry::default();
        // The actual collapsed stroke from room 71727123, notebook 102.
        let a = Vec3::new(-0.034921136, 0.012436249, -0.019527093);
        let b = Vec3::new(a.x, a.y, -0.019177092);
        quad(&mut g, [a, b, b, a]);
        assert!(g.positions.is_empty());
        assert!(g.normals.is_empty());
        assert!(g.uvs.is_empty());
        assert!(g.indices.is_empty());
        // Preserve thin, valid ink rather than removing the whole line/page.
        quad(&mut g, [a, b, b + Vec3::X * 0.00001, a + Vec3::X * 0.00001]);
        assert_eq!(g.positions.len(), 4);
        assert_eq!(g.indices.len(), 6);
        assert!(g
            .normals
            .iter()
            .all(|n| Vec3::from_array(*n).is_normalized()));
    }

    #[test]
    fn timeout_seed_notebook_remains_valid_at_sampled_sizes() {
        for scale in [0.5, 0.75, 1., 1.25, 2.] {
            let mut o = super::super::super::audit::fixture(
                ObjectKind::Notebook,
                2_530_243_425_158_947_287,
            );
            o.size = Vec3::new(0.13320178, 0.031643983, 0.15321597) * scale;
            assert!(parameters(&o).opened);
            let a = super::super::super::build_object(&o);
            super::super::super::audit::check(&o, &a).unwrap();
            let ink = &a.parts[&(Surface::Ink, o.kind.class_name().into())];
            assert!(ink.positions.len() > 100);
            assert!(ink.positions.iter().any(|p| p[0] < 0.));
            assert!(ink.positions.iter().any(|p| p[0] > 0.));
        }
    }

    #[test]
    fn capture_timeout_room_has_valid_geometry() {
        let scene = super::super::super::super::layout::IndoorManifest::generate_with_humans(
            71_727_123,
            super::super::super::super::layout::IndoorLayout::Mixed,
            0.65,
            0,
            0.,
        )
        .unwrap();
        super::super::super::super::validation::validate_geometry(&scene).unwrap();
    }

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
