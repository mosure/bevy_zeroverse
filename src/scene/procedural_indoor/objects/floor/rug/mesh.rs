//! Indexed pile field and metric cloth wrapping around the physical edge.
use super::*;
use crate::scene::procedural_indoor::{envelope::polygon, materials::rugs};

/// Rings are indexed once. The full-field dye map stays within [0, 1] and the
/// binding uses a separate cloth chart measured along and across the cut.
fn field(g: &mut Geometry, inside: &[Vec2], size: Vec3, rings: u32, rotate: bool) {
    let base = g.positions.len() as u32;
    let sides = inside.len() as u32;
    let extent = inside.iter().fold(Vec2::ZERO, |hi, p| hi.max(p.abs())) * 2.;
    g.positions.push([0., size.y, 0.]);
    g.normals.push(Vec3::Y.to_array());
    g.uvs.push([0.5, 0.5]);
    for row in 1..=rings {
        let t = row as f32 / rings as f32;
        let offset = base + 1 + (row - 1) * sides;
        for (i, &p) in inside.iter().enumerate() {
            let i = i as u32;
            let next = (i + 1) % sides;
            let natural = p / extent;
            let border = natural / natural.abs().max_element() * 0.5;
            g.positions.push([p.x * t, size.y, p.y * t]);
            g.normals.push(Vec3::Y.to_array());
            let [u, v] = (Vec2::splat(0.5) + natural.lerp(border, t * t) * t).to_array();
            g.uvs.push(if rotate { [v, 1. - u] } else { [u, v] });
            if row == 1 {
                g.indices.extend([base, offset + next, offset + i]);
            } else {
                let prev = offset - sides;
                g.indices.extend([
                    prev + i,
                    prev + next,
                    offset + next,
                    prev + i,
                    offset + next,
                    offset + i,
                ]);
            }
        }
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: RugProgram, rotate: bool) {
    let outside = p.outline(o.size.xz());
    let inset: Vec<_> = (0..outside.len())
        .map(|i| polygon::inset_direction(&outside, i))
        .collect();
    let inside: Vec<_> = outside
        .iter()
        .zip(&inset)
        .map(|(&v, &n)| v + n * p.binding_width_m)
        .collect();
    let slot = rugs::slot(o.finish_seed.unwrap_or(o.seed));
    let rings = if p.cut == Cut::Straight && p.taper == 0. {
        1
    } else {
        4
    };
    field(
        a.part(Surface::FabricAlt, &format!("floormat#rug{slot}")),
        &inside,
        o.size,
        rings,
        rotate,
    );

    let g = a.part(Surface::FabricAlt, &format!("floormat#rug_binding{slot}"));
    let sides = outside.len();
    let h = o.size.y;
    let at = |p: Vec2, y| Vec3::new(p.x, y, p.y);
    let underside = g.positions.len() as u32;
    g.positions.push([0., 0., 0.]);
    g.normals.push(Vec3::NEG_Y.to_array());
    g.uvs.push([0., 0.]);
    for (v, n) in outside.iter().zip(&inset) {
        let foot = v + n * p.turn_under_m;
        g.positions.push(at(foot, 0.).to_array());
        g.normals.push(Vec3::NEG_Y.to_array());
        g.uvs.push(foot.to_array());
    }
    let mut along = 0.;
    let width = p.binding_width_m;
    let drop = h.hypot(p.turn_under_m);
    for i in 0..sides {
        let j = (i + 1) % sides;
        let end = along + outside[i].distance(outside[j]);
        let top = [inside[i], inside[j], outside[j], outside[i]].map(|v| at(v, h));
        g.vertex_quad(
            top,
            [Vec3::Y; 4],
            [[along, 0.], [end, 0.], [end, width], [along, width]],
        );
        let foot = |k| Vec3::from_array(g.positions[underside as usize + 1 + k]);
        let side = [foot(i), top[3], top[2], foot(j)];
        let normal = (side[1] - side[0]).cross(side[2] - side[0]).normalize();
        g.vertex_quad(
            side,
            [normal; 4],
            [
                [along, width + drop],
                [along, width],
                [end, width],
                [end, width + drop],
            ],
        );
        g.indices.extend([
            underside,
            underside + 1 + i as u32,
            underside + 1 + j as u32,
        ]);
        along = end;
    }
}
