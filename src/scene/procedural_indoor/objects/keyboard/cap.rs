//! Closed, tapered keycaps with a dished crown. Printing is part of the crown
//! UVs. Separating skirt and crown normals preserves the molded edge highlight.
use super::*;

pub(super) fn build(
    g: &mut Geometry,
    stock: &Caps,
    field: &Field,
    legend: u8,
    center: Vec2,
    size: Vec2,
) {
    let base = size * 0.5;
    let rim = base - stock.taper.min(base.min_element() * 0.24);
    let corner = (rim.min_element() * 0.19).clamp(0.00038, 0.0018);
    let rim_y = field.deck_y + stock.height;
    let bottom = field.deck_y - 0.0003;
    let points = |s: Vec2| {
        [
            Vec2::new(-s.x + corner, -s.y),
            Vec2::new(-s.x, -s.y + corner),
            Vec2::new(-s.x, s.y - corner),
            Vec2::new(-s.x + corner, s.y),
            Vec2::new(s.x - corner, s.y),
            Vec2::new(s.x, s.y - corner),
            Vec2::new(s.x, -s.y + corner),
            Vec2::new(s.x - corner, -s.y),
        ]
    };
    let lower = points(base);
    let upper = points(rim);
    let tf = field.frame;
    let start = g.positions.len() as u32;
    let mut vert = |p: Vec2, y, n: Vec3, uv: Vec2| {
        let local = center + p;
        g.positions.push(
            tf.transform_point(Vec3::new(local.x, y - field.slope * local.y, local.y))
                .to_array(),
        );
        let normal = if field.slope == 0. {
            n
        } else {
            (n + Vec3::Z * (field.slope * n.y)).normalize()
        };
        g.normals.push((tf.rotation * normal).to_array());
        g.uvs.push(uv.to_array());
    };
    for i in 0..8 {
        let outward = (upper[i] - upper[(i + 7) % 8]).normalize()
            + (upper[(i + 1) % 8] - upper[i]).normalize();
        let outward = outward.perp().normalize();
        let n = Vec3::new(outward.x, stock.taper / stock.height, outward.y).normalize();
        vert(
            lower[i],
            bottom,
            n,
            keycaps::uv(b' ', Vec2::new(i as f32 / 8., 0.9)),
        );
        vert(
            upper[i],
            rim_y,
            n,
            keycaps::uv(b' ', Vec2::new(i as f32 / 8., 0.1)),
        );
    }
    for &p in &upper {
        let unit = p / rim;
        let n = Vec3::new(
            -unit.x * stock.dish / rim.x,
            1.,
            -unit.y * stock.dish / rim.y,
        )
        .normalize();
        vert(p, rim_y, n, keycaps::uv(legend, unit * 0.5 + 0.5));
    }
    vert(
        Vec2::ZERO,
        rim_y - stock.dish,
        Vec3::Y,
        keycaps::uv(legend, Vec2::splat(0.5)),
    );
    for &p in &lower {
        vert(
            p,
            bottom,
            Vec3::NEG_Y,
            keycaps::uv(b' ', p / base * 0.5 + 0.5),
        );
    }
    for i in 0..8u32 {
        let j = (i + 1) % 8;
        let lo = start + 2 * i;
        let next = start + 2 * j;
        g.indices.extend([
            lo,
            next,
            next + 1,
            lo,
            next + 1,
            lo + 1,
            start + 16 + i,
            start + 16 + j,
            start + 24,
        ]);
    }
    for i in 1..7u32 {
        g.indices
            .extend([start + 25, start + 25 + i + 1, start + 25 + i]);
    }
}
