use super::*;

// Corresponding points on the folded outline and neckline. The extra middle
// points allow a turned shirt collar to have two tips and a central opening.
type Ring = [Vec3; 12];
const OUTLINE: [(f32, f32); 12] = [
    (-0.83, -1.),
    (0., -1.),
    (0.83, -1.),
    (1., -0.80),
    (1., 0.),
    (1., 0.78),
    (0.83, 1.),
    (0., 1.),
    (-0.83, 1.),
    (-1., 0.78),
    (-1., 0.),
    (-1., -0.80),
];
const OVAL: [(f32, f32); 12] = [
    (-0.52, -0.88),
    (0., -1.),
    (0.52, -0.88),
    (0.91, -0.48),
    (1., 0.),
    (0.78, 0.56),
    (0.42, 0.91),
    (0., 1.),
    (-0.42, 0.91),
    (-0.78, 0.56),
    (-1., 0.),
    (-0.91, -0.48),
];

/// Keep the grain aligned across adjoining panels using coordinates in metres.
fn face(g: &mut Geometry, mut p: [Vec3; 4]) {
    let a = (p[1] - p[0]).cross(p[2] - p[0]);
    let b = (p[2] - p[0]).cross(p[3] - p[0]);
    let n = (a + b).normalize();
    // An off-centre pocket can make a concave shoulder panel. Choose its
    // interior diagonal so both rendered triangles face the same way.
    if a.dot(n) <= 0. || b.dot(n) <= 0. {
        p.rotate_left(1);
    }
    let uv = p.map(|v| {
        if n.y.abs() > 0.48 {
            [v.x, v.z]
        } else if n.x.abs() > n.z.abs() {
            [v.z, v.y]
        } else {
            [v.x, v.y]
        }
    });
    g.vertex_quad(p, [n; 4], uv);
}

// For counterclockwise X/Z outlines this winding also joins the top shoulder
// to an inner neck rim and turns down the inside wall of the opening.
fn join(g: &mut Geometry, first: &Ring, second: &Ring) {
    for j in 0..first.len() {
        let k = (j + 1) % first.len();
        face(g, [first[j], second[j], second[k], first[k]]);
    }
}

fn cap(g: &mut Geometry, ring: &Ring, up: bool) {
    let start = g.positions.len() as u32;
    let center = ring.iter().sum::<Vec3>() / ring.len() as f32;
    let normal = if up { Vec3::Y } else { Vec3::NEG_Y };
    for &p in std::iter::once(&center).chain(ring) {
        g.positions.push(p.to_array());
        g.normals.push(normal.to_array());
        g.uvs.push([p.x, p.z]);
    }
    for j in 0..ring.len() as u32 {
        let a = start + 1 + j;
        let b = start + 1 + (j + 1) % ring.len() as u32;
        g.indices
            .extend(if up { [start, b, a] } else { [start, a, b] });
    }
}

pub(super) fn build(a: &mut Assembly, pattern: &FoldedGarment, exposed: bool) {
    let label = format!("clothes#finish{}", pattern.dye_slot);
    let g = a.part(pattern.surface, &label);
    let s = pattern.size_m;
    let rim = |y: f32, width: f32, front: f32, back: f32| {
        OUTLINE.map(|(x, z)| {
            let taper = if x < 0. {
                pattern.side_tuck.x
            } else {
                pattern.side_tuck.y
            };
            let fold = if z > 0. {
                1. - (1. - front) * (1. + pattern.edge_skew * x)
            } else {
                back
            };
            Vec3::new(
                x * s.x * 0.5 * width * (1. - taper * (z + 1.) * 0.5),
                y * s.y,
                z * s.z * 0.5 * fold,
            )
        })
    };
    // The return is visible at the open edge. At the back the fabric instead
    // follows a single continuous bend around the compressed layers.
    let bottom = rim(0., 0.93, 0.92, 0.94);
    cap(g, &bottom, false);
    let mut prev = bottom;
    let recessed = 1. - 2. * pattern.open_edge_recess_m / s.z;
    for next in [
        rim(0.16, 0.99, 0.99, 0.99),
        rim(0.48, 1., 1., 1.),
        rim(0.59, 0.989, recessed, 0.999),
        rim(0.72, 1., 0.992, 0.984),
        rim(1., 0.945, 0.949, 0.950),
    ] {
        join(g, &prev, &next);
        prev = next;
    }
    if !exposed {
        cap(g, &prev, true);
        return;
    }
    if pattern.cut == FoldedCut::Trousers {
        pocket(g, pattern, &prev);
    } else {
        neckline(a, pattern, &label, &prev);
    }
}

fn neckline(a: &mut Assembly, pattern: &FoldedGarment, label: &str, top: &Ring) {
    let s = pattern.size_m;
    let rise = pattern.detail_rise_m;
    let shirt = pattern.cut == FoldedCut::ButtonDown;
    let Vec2 { x: rx, y: rz } = pattern.feature_half_size_m;
    let center = pattern.feature_center_m;
    let seam = |i: usize, scale: f32, y: f32| {
        let (x, z) = OVAL[i];
        Vec3::new(center.x + rx * x * scale, y, center.y + rz * z * scale)
    };
    let outer = std::array::from_fn(|i| {
        let spread = if shirt && (i == 6 || i == 8) {
            1.63
        } else {
            1.
        };
        seam(i, spread, s.y)
    });
    let raised = std::array::from_fn(|i| seam(i, if shirt { 0.76 } else { 0.90 }, s.y + rise));
    let well = std::array::from_fn(|i| seam(i, if shirt { 0.67 } else { 0.82 }, s.y * 0.54));
    join(a.part(pattern.surface, label), top, &outer);
    let binding = format!("clothes#finish{}", pattern.binding_slot);
    let g = a.part(pattern.surface, &binding);
    join(g, &outer, &raised);
    join(g, &raised, &well);
    // The back of the folded garment is visible through the actual neck hole.
    cap(a.part(pattern.surface, label), &well, true);
    if shirt {
        let z0 = center.y + 1.20 * rz;
        let z1 = 0.42 * s.z;
        let lift = 0.23 * rise;
        strip(
            a.part(pattern.surface, label),
            Vec2::new(center.x - s.x * 0.040, z0),
            Vec2::new(center.x + s.x * 0.040, z1),
            s.y,
            lift,
        );
        let buttons = a.part(Surface::Plastic, "clothes");
        let r = (s.x * 0.014).min(0.0046);
        for f in [0.20, 0.52, 0.84] {
            buttons.lathe(
                &[(r, 0.), (r, 0.0011), (r * 0.77, 0.0018), (0., 0.0018)],
                8,
                Transform::from_xyz(center.x, s.y + lift, z0 + (z1 - z0) * f),
            );
        }
    }
}

fn pocket(g: &mut Geometry, pattern: &FoldedGarment, top: &Ring) {
    let s = pattern.size_m;
    let rise = pattern.detail_rise_m;
    let shape = |inset: f32, y: f32| {
        OUTLINE.map(|(x, z)| {
            let toe = 1. - 0.68 * z.max(0.);
            let q = pattern.feature_center_m
                + pattern.feature_half_size_m * Vec2::new(x * toe, z) * inset;
            Vec3::new(q.x, y, q.y)
        })
    };
    let edge = shape(1., s.y);
    let seam = shape(0.88, s.y + rise);
    join(g, top, &edge);
    join(g, &edge, &seam);
    cap(g, &seam, true);
    strip(
        g,
        Vec2::new(-s.x * 0.408, -s.z * 0.413),
        Vec2::new(s.x * 0.408, -s.z * 0.298),
        s.y,
        rise * 0.61,
    );
}

/// A tapered sewn strip on its supporting panel. Only the top and the four
/// sloping sides are exposed, so there is no duplicate face at the seam.
fn strip(g: &mut Geometry, lo: Vec2, hi: Vec2, y: f32, lift: f32) {
    let contour = |lo: Vec2, hi: Vec2, y| {
        [
            Vec3::new(lo.x, y, lo.y),
            Vec3::new(hi.x, y, lo.y),
            Vec3::new(hi.x, y, hi.y),
            Vec3::new(lo.x, y, hi.y),
        ]
    };
    let bottom = contour(lo, hi, y);
    let inset = (hi - lo) * 0.085;
    let mut top = contour(lo + inset, hi - inset, y + lift);
    for j in 0..4 {
        let k = (j + 1) % 4;
        face(g, [bottom[j], top[j], top[k], bottom[k]]);
    }
    top.reverse();
    face(g, top);
}
