//! A continuous case skin with a die-cut mouth. The lining terminates on the
//! refill, so oblique views into the opening see a real first surface.
use super::*;
use crate::scene::procedural_indoor::materials::tissue_stock;

const N: usize = 16;
type Outline = [Vec2; N];

/// Three facets per quarter turn and four straight edges. Nonzero straight
/// segments keep the same topology for both the oval cut and square carton.
fn outline(half: Vec2, radius: Vec2, center: Vec2) -> Outline {
    const ARC: [Vec2; 4] = [
        Vec2::X,
        Vec2::new(0.8660254, 0.5),
        Vec2::new(0.5, 0.8660254),
        Vec2::Y,
    ];
    std::array::from_fn(|i| {
        let a = ARC[i % 4];
        let (dir, quadrant) = match i / 4 {
            0 => (a, Vec2::ONE),
            1 => (a.perp(), Vec2::new(-1., 1.)),
            2 => (-a, Vec2::NEG_ONE),
            _ => (-a.perp(), Vec2::new(1., -1.)),
        };
        center + (half - radius) * quadrant + radius * dir
    })
}
fn ring(g: &mut Geometry, outer: &Outline, inner: &Outline, y: f32, up: bool) {
    for i in 0..N {
        let j = (i + 1) % N;
        let mut uv = [outer[i], inner[i], inner[j], outer[j]];
        if !up {
            uv.reverse();
        }
        g.vertex_quad(
            uv.map(|q| Vec3::new(q.x, y, q.y)),
            [if up { Vec3::Y } else { Vec3::NEG_Y }; 4],
            uv.map(|v| v.to_array()),
        );
    }
}
fn lining(g: &mut Geometry, shape: &Outline, high: f32, low: f32) {
    let at = |q: Vec2, h| Vec3::new(q.x, h, q.y);
    for i in 0..N {
        let j = (i + 1) % N;
        let p = [
            at(shape[i], high),
            at(shape[i], low),
            at(shape[j], low),
            at(shape[j], high),
        ];
        g.metric_quad(p, (p[1] - p[0]).cross(p[2] - p[0]));
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &TissueProgram, d: Dispenser) {
    let h = o.size.y * p.box_height;
    let half = o.size.xz() * 0.5;
    let outer = outline(half, Vec2::splat(d.corner_m), Vec2::ZERO);
    let hole = half * p.opening;
    let mouth = outline(hole, hole * d.mouth_rounding, d.mouth_offset_m);
    let print;
    let label = if d.case == Case::FoldingCarton {
        print = tissue_stock::carton_label(o);
        &print
    } else {
        o.kind.class_name()
    };
    let skin = a.part(p.finish, label);
    skin.profile_slab_with_top(
        &outer,
        h,
        d.lip_m,
        Transform::from_xyz(0., h * 0.5, 0.),
        |g, _, inset, _, _| {
            ring(g, &outer.map(|q| q * inset), &mouth, h, true);
        },
    );
    if d.draw == Draw::Sealed {
        // A scored panel sits in the same printed chart, with a shallow
        // debossed cut line. There is no layer of paper hiding behind it.
        let depth = d.wall_m * 0.58;
        let inner = mouth.map(|q| {
            d.mouth_offset_m + (q - d.mouth_offset_m) * (1. - 0.00065 / hole.min_element())
        });
        lining(skin, &mouth, h, h - depth);
        ring(skin, &mouth, &inner, h - depth, true);
        skin.slab_cap(
            &inner,
            1.,
            d.mouth_offset_m,
            h - depth * 0.25,
            &Transform::IDENTITY,
            |v| v,
        );
        // The panel's slightly raised edge faces out into the groove.
        for i in 0..N {
            let j = (i + 1) % N;
            let at = |q: Vec2, y| Vec3::new(q.x, y, q.y);
            let v = [
                at(inner[i], h - depth),
                at(inner[i], h - depth * 0.25),
                at(inner[j], h - depth * 0.25),
                at(inner[j], h - depth),
            ];
            skin.metric_quad(v, (v[1] - v[0]).cross(v[2] - v[0]));
        }
        return;
    }
    let interior = if d.case == Case::FoldingCarton {
        a.part(Surface::Paper, tissue_stock::KRAFT)
    } else {
        a.part(p.finish, label)
    };
    let inside = outline(
        half - d.wall_m,
        Vec2::splat((d.corner_m - d.wall_m).max(0.0005)),
        Vec2::ZERO,
    );
    let underside = h - d.wall_m.max(d.lip_m);
    let pack = h * d.fill_fraction;
    lining(interior, &mouth, h, underside);
    ring(interior, &inside, &mouth, underside, false);
    lining(interior, &inside, underside, pack);
    a.part(Surface::Paper, tissue_stock::PLY).slab_cap(
        &inside,
        1.,
        Vec2::ZERO,
        pack,
        &Transform::IDENTITY,
        |v| v,
    );
}
