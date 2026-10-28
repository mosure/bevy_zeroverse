//! Paper, backing and binding share a fitted, supported metric envelope.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

#[derive(Clone, Debug, serde::Serialize)]
pub struct PadProgram {
    /// Glued top, top wire, side wire, clipboard, unbound sheets.
    pub binding: u8,
    pub span: Vec2,
    pub height: f32,
    pub backing: f32,
    pub page_inset: Vec2,
    pub page_bias: Vec2,
    pub corner: f32,
    pub ring_pitch: f32,
    pub binding_width: f32,
    pub backing_surface: Surface,
}
pub fn parameters(o: &IndoorObject) -> PadProgram {
    let mut r = stream(o.seed, 0x50414450524f4752);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x50414446494e4953);
    let binding = r.random_range(0..5);
    let span = o.size.xz() * Vec2::new(r.random_range(0.80..0.99), r.random_range(0.80..0.99));
    let height = o.size.y * r.random_range(0.44..0.94);
    let backing = (height * r.random_range(0.08..0.20)).clamp(0.0004, 0.003);
    let inset = span * Vec2::new(r.random_range(0.014..0.065), r.random_range(0.016..0.075));
    PadProgram {
        binding,
        span,
        height,
        backing,
        page_inset: inset,
        page_bias: inset * Vec2::new(r.random_range(-0.45..0.45), r.random_range(-0.45..0.45)),
        corner: span.min_element() * r.random_range(0.008..0.055),
        ring_pitch: r.random_range(0.012..0.026),
        binding_width: r.random_range(0.23..0.61),
        backing_surface: if binding == 3 && f.random_bool(0.6) {
            Surface::Wood
        } else if binding == 4 {
            Surface::Paper
        } else {
            Surface::Plastic
        },
    }
}
fn outline(s: Vec2, corner: f32) -> [Vec2; 8] {
    let h = s * 0.5;
    [
        Vec2::new(-h.x + corner, -h.y),
        Vec2::new(h.x - corner, -h.y),
        Vec2::new(h.x, -h.y + corner),
        Vec2::new(h.x, h.y - corner),
        Vec2::new(h.x - corner, h.y),
        Vec2::new(-h.x + corner, h.y),
        Vec2::new(-h.x, h.y - corner),
        Vec2::new(-h.x, -h.y + corner),
    ]
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let l = o.kind.class_name();
    a.part(p.backing_surface, l).profile_slab(
        &outline(p.span, p.corner),
        p.backing,
        p.backing * 0.15,
        Transform::from_xyz(0., p.backing * 0.5, 0.),
    );
    // Binding hardware is included in the admitted height, including its wire radius.
    let top = p.height
        * if p.binding == 1 || p.binding == 2 {
            0.74
        } else if p.binding == 3 {
            0.62
        } else {
            0.98
        };
    let size = p.span - p.page_inset * 2.;
    let mut page = Geometry::default();
    page.profile_slab(
        &outline(size, p.corner.min(size.min_element() * 0.05)),
        top - p.backing,
        0.0001,
        Transform::from_xyz(p.page_bias.x, (top + p.backing) * 0.5, p.page_bias.y),
    );
    // Move the real top face to a once-wrapped document. No overlay, no coincident
    // faces, and no accidental centimetre-sized crop of a metre-space texture.
    let indices = std::mem::take(&mut page.indices);
    let printed = a.part(Surface::PrintedPaper, l);
    for &tri in indices.as_chunks::<3>().0 {
        if tri.iter().all(|&i| page.normals[i as usize][1] > 0.999) {
            let start = printed.positions.len() as u32;
            for i in tri {
                let v = page.positions[i as usize];
                printed.positions.push(v);
                printed.normals.push([0., 1., 0.]);
                printed.uvs.push([
                    (v[0] - p.page_bias.x) / size.x + 0.5,
                    (v[2] - p.page_bias.y) / size.y + 0.5,
                ]);
            }
            printed.indices.extend([start, start + 1, start + 2]);
        } else {
            page.indices.extend(tri);
        }
    }
    a.part(Surface::Paper, l).append(page);
    if p.binding == 1 || p.binding == 2 {
        let side = p.binding == 2;
        let length = if side { size.y } else { size.x };
        let n = (length / p.ring_pitch).round().clamp(3., 24.) as usize;
        let wire = (o.size.y * 0.025).min(0.00045);
        let radius = p.height * 0.48 - wire;
        let width = (p.span.min_element() * 0.018).min(0.0038);
        for i in 0..n {
            let along = length * ((i as f32 + 0.5) / n as f32 - 0.5) * 0.90;
            let points: [Vec3; 13] = std::array::from_fn(|j| {
                let t = j as f32 * TAU / 12.;
                let cross = width * t.cos();
                if side {
                    Vec3::new(
                        p.page_bias.x - size.x * 0.47 + cross,
                        radius + wire + radius * t.sin(),
                        p.page_bias.y + along,
                    )
                } else {
                    Vec3::new(
                        p.page_bias.x + along,
                        radius + wire + radius * t.sin(),
                        p.page_bias.y - size.y * 0.47 + cross,
                    )
                }
            });
            a.part(Surface::Metal, l).tube(&points, wire, 5);
        }
    } else if p.binding == 3 {
        let w = size.x * p.binding_width;
        let d = size.y * 0.075;
        let h = p.height - top;
        a.box_part(
            Surface::Metal,
            l,
            Vec3::new(p.page_bias.x, top + h * 0.20, p.page_bias.y - size.y * 0.43),
            Vec3::new(w, h * 0.40, d),
            h * 0.08,
        );
        a.box_part(
            Surface::Metal,
            l,
            Vec3::new(p.page_bias.x, top + h * 0.68, p.page_bias.y - size.y * 0.45),
            Vec3::new(w * 0.65, h * 0.55, d * 0.38),
            h * 0.10,
        );
    } else if p.binding == 0 {
        a.box_part(
            p.backing_surface,
            l,
            Vec3::new(
                p.page_bias.x,
                top + p.height * 0.009,
                p.page_bias.y - size.y * 0.47,
            ),
            Vec3::new(size.x, p.height * 0.018, size.y * 0.055),
            0.,
        );
    }
}
