//! Printed dial stock and temperature displays. All markings reuse the shared
//! mipmapped panel films. No per-device rasterization or image allocation.
use super::*;

fn corners(rect: Rect, z: f32) -> [Vec3; 4] {
    [
        rect.min,
        Vec2::new(rect.max.x, rect.min.y),
        rect.max,
        Vec2::new(rect.min.x, rect.max.y),
    ]
    .map(|xy| xy.extend(z))
}
pub(super) fn tile(g: &mut Geometry, pos: Vec3, size: Vec2, glyph: Glyph) {
    g.vertex_quad(
        corners(Rect::from_center_size(pos.truncate(), size), pos.z),
        [Vec3::Z; 4],
        glyph.uvs(),
    );
}
/// Sample the middle of a uniform region of the film. The center electrode
/// of eight is energized on both reflective LCD and emissive LED stock.
fn swatch(glyph: Glyph) -> [[f32; 2]; 4] {
    let [a, _, b, _] = glyph.uvs();
    let center = (Vec2::from_array(a) + Vec2::from_array(b)) * 0.5;
    // A subpixel quad inside the uniform swatch preserves a valid UV tangent.
    let span = (Vec2::from_array(b) - Vec2::from_array(a)) * 0.008;
    [
        -Vec2::ONE,
        Vec2::new(1., -1.),
        Vec2::ONE,
        Vec2::new(-1., 1.),
    ]
    .map(|c| (center + c * span).to_array())
}
/// Square physical pixels keep the superscript degree round on every aspect
/// ratio. Tile the face once so foreground and background cannot z-fight.
pub(super) fn unit(g: &mut Geometry, pos: Vec3, size: Vec2, fahrenheit: bool) {
    // Upper-left is the degree ring; the rightmost four columns form C or F.
    let rows = if fahrenheit {
        [
            0b011000000,
            0b100101111,
            0b100101000,
            0b011001000,
            0b000001110,
            0b000001000,
            0b000001000,
        ]
    } else {
        [
            0b011000000,
            0b100100111,
            0b100101000,
            0b011001000,
            0b000001000,
            0b000001000,
            0b000000111,
        ]
    };
    let ink = swatch(Glyph::Number(8));
    let glass = swatch(Glyph::Blank);
    let bounds = Rect::from_center_size(pos.truncate(), size);
    let pitch = (size.x / 10.).min(size.y / 10.5);
    let left = bounds.min.x + 0.04 * size.x;
    let top = bounds.max.y - 0.10 * size.y;
    let right = left + 9. * pitch;
    let bottom = top - 7. * pitch;
    let mut patch = |rect, uv| g.vertex_quad(corners(rect, pos.z), [Vec3::Z; 4], uv);
    patch(
        Rect::new(bounds.min.x, top, bounds.max.x, bounds.max.y),
        glass,
    );
    patch(
        Rect::new(bounds.min.x, bounds.min.y, bounds.max.x, bottom),
        glass,
    );
    patch(Rect::new(bounds.min.x, bottom, left, top), glass);
    patch(Rect::new(right, bottom, bounds.max.x, top), glass);
    for (r, bits) in rows.into_iter().enumerate() {
        let y = top - r as f32 * pitch;
        let marked = |c: usize| bits & (1 << (8 - c)) != 0;
        let mut a = 0;
        while a < 9 {
            let mut b = a + 1;
            while b < 9 && marked(b) == marked(a) {
                b += 1;
            }
            patch(
                Rect::new(
                    left + a as f32 * pitch,
                    y - pitch,
                    left + b as f32 * pitch,
                    y,
                ),
                if marked(a) { ink } else { glass },
            );
            a = b;
        }
    }
}

/// One round, graduated annulus. Tick and stock sectors tile the same plane;
/// sharing the housing and pointer frame keeps the marks registered at any roll.
pub(super) fn dial(g: &mut Geometry, center: Vec2, z: f32, r: f32, fahrenheit: bool) {
    let stock = swatch(Glyph::Blank);
    let ink = swatch(Glyph::Stop);
    let mut sector = |a: f32, b: f32, inner: f32, outer: f32, uv| {
        let direction = |angle: f32| {
            let (s, c) = angle.sin_cos();
            Vec2::new(s, c)
        };
        let p = direction(a) * r;
        let q = direction(b) * r;
        let points = [p * inner, q * inner, q * outer, p * outer].map(|v| (center + v).extend(z));
        g.vertex_quad(points, [Vec3::Z; 4], uv);
    };
    // Measured from twelve o'clock. Eleven graduation centers match the
    // pointer's five-radian adjustment range exactly.
    let first = -2.5 - 0.030;
    let mut prior = first;
    for i in 0..=11 {
        let major = i % 2 == 0 || i == 5;
        let half = if major { 0.030 } else { 0.019 };
        let mid = -2.5 + i as f32 * 0.5;
        let left = if i == 11 { first + TAU } else { mid - half };
        let steps = ((left - prior) / 0.14).ceil() as usize;
        for j in 0..steps {
            let step = |n| prior + (left - prior) * (n as f32 / steps as f32);
            sector(step(j), step(j + 1), 0.86, 1.44, stock);
        }
        if i < 11 {
            let start = if major { 1.078 } else { 1.18 };
            sector(left, mid + half, 0.86, start, stock);
            sector(left, mid + half, start, 1.355, ink);
            sector(left, mid + half, 1.355, 1.44, stock);
            prior = mid + half;
        }
    }
    let limits = if fahrenheit { [59, 86] } else { [15, 30] };
    for (side, n) in [-1., 1.].into_iter().zip(limits) {
        for (j, v) in [n / 10, n % 10].into_iter().enumerate() {
            let xy = center + Vec2::new(side * 0.43 + (j as f32 - 0.5) * 0.22, -1.14) * r;
            // Labels sit inside the unticked bottom arc, fully backed by stock.
            tile(
                g,
                xy.extend(z + 0.00022),
                Vec2::new(0.22, 0.26) * r,
                Glyph::Number(v),
            );
        }
    }
}
