//! Archived variants 0–3. Replay the original silhouette, support and UVs.
use super::super::*;

fn triangle(g: &mut Geometry, p: Vec3, q: Vec3, r: Vec3) {
    let i = g.positions.len() as u32;
    let n = (q - p).cross(r - p).normalize();
    for v in [p, q, r] {
        g.positions.push(v.to_array());
        g.normals.push(n.to_array());
        g.uvs.push(v.xz().to_array());
    }
    g.indices.extend([i, i + 1, i + 2]);
}

/// Same physical contour used by geometry, support and placement.
pub(super) fn outline(o: &IndoorObject) -> Vec<Vec2> {
    let program = parameters(o);
    let exponent = program.rug_exponent;
    (0..if program.clipped_corners { 8 } else { 64 })
        .map(|i| {
            let angle = TAU * (i as f32 + if program.clipped_corners { 0.5 } else { 0. })
                / if program.clipped_corners { 8. } else { 64. };
            let f = |v: f32| v.signum() * v.abs().powf(2. / exponent);
            Vec2::new(
                f(angle.cos())
                    * o.size.x
                    * 0.5
                    * (1. - program.rug_taper.abs() + program.rug_taper * f(angle.sin())),
                f(angle.sin()) * o.size.z * 0.5,
            )
        })
        .collect()
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let mut rng = stream(o.seed, 7340);
    // Superellipse exponent spans oval, capsule-like and softly squared forms.
    // Rotation/aspect/edge width are independent continuous parameters.
    let _ = rng.random_range(2.0_f32..7.0);
    let border = rng.random_range(0.025..0.095) / o.size.x.min(o.size.z);
    let inset = (1. - border * 2.).clamp(0.65, 0.97);
    let outline = outline(o);
    let field = Surface::FabricAlt;
    let slot =
        crate::scene::procedural_indoor::materials::rugs::slot(o.finish_seed.unwrap_or(o.seed));
    let rug_label = format!("floormat#rug{slot}");
    let edge_label = format!("floormat#rug_binding{slot}");
    let top = |p: Vec2| Vec3::new(p.x, o.size.y, p.y);
    let bottom = |p: Vec2| Vec3::new(p.x, 0., p.y);
    // Keep Cartesian motif coordinates near the center and ease into the
    // outline-aligned border. One center fan stretches patches into wedges;
    // a bounded four-ring tessellation samples the smooth interior mapping.
    let uv = |p: Vec2, radius: f32| {
        let natural = p / Vec2::new(o.size.x, o.size.z);
        let boundary = natural / natural.abs().max_element() * 0.5;
        (Vec2::splat(0.5) + natural.lerp(boundary, radius * radius) * radius).to_array()
    };
    for i in 0..outline.len() {
        let p = outline[i];
        let q = outline[(i + 1) % outline.len()];
        let g = a.part(field, &rug_label);
        for ring in 0..4 {
            let inner = ring as f32 / 4.;
            let outer = (ring + 1) as f32 / 4.;
            let mut add = |points: [(Vec2, f32); 3]| {
                let start = g.uvs.len();
                let positions = points.map(|(p, r)| top(p * r * inset));
                triangle(g, positions[0], positions[1], positions[2]);
                for (index, (p, r)) in points.into_iter().enumerate() {
                    g.uvs[start + index] = uv(p, r);
                }
            };
            if ring == 0 {
                add([(p, 0.), (q, outer), (p, outer)]);
            } else {
                add([(p, inner), (q, inner), (q, outer)]);
                add([(p, inner), (q, outer), (p, outer)]);
            }
        }
        let g = a.part(field, &edge_label);
        triangle(g, top(p * inset), top(q * inset), top(q));
        triangle(g, top(p * inset), top(q), top(p));
        triangle(g, bottom(p), top(p), top(q));
        triangle(g, bottom(p), top(q), bottom(q));
        triangle(g, Vec3::ZERO, bottom(p), bottom(q));
    }
}
