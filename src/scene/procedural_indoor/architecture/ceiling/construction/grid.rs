//! Exposed T runners, supported mineral-fiber stock and open-cell grids.
use super::*;
use crate::scene::procedural_indoor::materials::acoustic_tile;

pub(super) enum Infill {
    Open,
    ArchivedTile(f32),
    Mineral { base: f32, stock: CeilingTile },
}

pub(super) fn build(
    a: &mut Assembly,
    f: &Frame,
    aspect: f32,
    infill: Infill,
    anchors_only: bool,
) -> Vec<Vec3> {
    let open = matches!(infill, Infill::Open);
    let (base, stock) = match infill {
        Infill::Open => (0.013, None),
        Infill::ArchivedTile(recess) => (recess + 0.009, None),
        Infill::Mineral { base, stock } => (base, Some(stock)),
    };
    let nx = (f.length / (f.m.pitch_m * aspect)).round().clamp(1., 16.) as usize;
    let nz = (f.width / f.m.pitch_m).round().clamp(1., 12.) as usize;
    let dx = (f.length - 0.028) / nx as f32;
    let dz = (f.width - 0.028) / nz as f32;
    let rail_h = if open { f.height * 0.76 } else { 0.035 };
    let rail_w = stock.map_or(if open { 0.012 } else { 0.023 }, |tile| tile.flange_width_m);
    let mut anchors = Vec::with_capacity(4);
    // Attach hangers to actual outside runners.
    for x in [-f.length * 0.32, f.length * 0.32] {
        for z in [-(nz as f32) * dz * 0.5, nz as f32 * dz * 0.5] {
            anchors.push(Vec3::new(x, base + rail_h - 0.002, z));
        }
    }
    if anchors_only {
        return anchors;
    }
    let material = if open {
        Surface::Metal
    } else {
        Surface::Ceiling
    };
    // Real T extrusions: narrow vertical web and a broad, thin tile ledge.
    // Flat extrusion faces avoid subdividing hidden back edges of every cell.
    let rail = |a: &mut Assembly, p: Vec3, size: Vec3| {
        if open {
            f.block(a, Surface::Metal, p, size, 0.001);
        } else {
            let flange = if stock.is_some() {
                a.part(Surface::Metal, acoustic_tile::GRID_LABEL)
            } else {
                f.part(a, Surface::Metal)
            };
            flange.cuboid(
                size.with_y(0.004),
                0.,
                Transform::from_translation(p.with_y(base + 0.002)),
            );
            let web = if size.x > size.z {
                Vec3::new(size.x, rail_h - 0.003, 0.004)
            } else {
                Vec3::new(0.004, rail_h - 0.003, size.z)
            };
            f.block(
                a,
                Surface::Metal,
                p.with_y(base + 0.003 + web.y * 0.5),
                web,
                0.,
            );
        }
    };
    // Continuous main runners; cross tees end at them with no coplanar faces.
    for j in 0..=nz {
        let z = (j as f32 - nz as f32 * 0.5) * dz;
        rail(
            a,
            Vec3::new(0., base + rail_h * 0.5, z),
            Vec3::new(f.length - 0.014, rail_h, rail_w),
        );
    }
    for i in 0..=nx {
        let x = (i as f32 - nx as f32 * 0.5) * dx;
        for j in 0..nz {
            let z = (j as f32 + 0.5 - nz as f32 * 0.5) * dz;
            rail(
                a,
                Vec3::new(x, base + rail_h * 0.5, z),
                Vec3::new(rail_w, rail_h, dz - rail_w - 0.002),
            );
        }
    }
    if let Some(stock) = stock {
        let g = a.part(Surface::Ceiling, stock.facing.label());
        let mut cuts = stream(f.m.seed, u64::from_le_bytes(*b"TileCuts"));
        for i in 0..nx {
            for j in 0..nz {
                let center = Vec2::new(
                    (i as f32 + 0.5 - nx as f32 * 0.5) * dx,
                    (j as f32 + 0.5 - nz as f32 * 0.5) * dz,
                );
                board(g, stock, center, Vec2::new(dx, dz), base, &mut cuts);
            }
        }
    } else if let Infill::ArchivedTile(recess) = infill {
        for i in 0..nx {
            for j in 0..nz {
                let p = Vec3::new(
                    (i as f32 + 0.5 - nx as f32 * 0.5) * dx,
                    base + 0.004 - recess + (recess + 0.016) * 0.5,
                    (j as f32 + 0.5 - nz as f32 * 0.5) * dz,
                );
                // Tegular tiles: real recessed lower face, narrow perimeter reveal,
                // thicker inner tile. The outer lip rests on the T ledge.
                f.block(
                    a,
                    material,
                    p,
                    Vec3::new(dx - rail_w - 0.006, recess + 0.016, dz - rail_w - 0.006),
                    0.002,
                );
                f.block(
                    a,
                    material,
                    p.with_y(base + 0.008),
                    Vec3::new(dx - 0.010, 0.008, dz - 0.010),
                    0.,
                );
            }
        }
    }
    // Vertical fascia closes the raft edge; the slab is visible through open cells.
    let edge_h = (rail_h + 0.016).min(f.height - base - 0.002);
    for z in [-f.width * 0.5 + 0.004, f.width * 0.5 - 0.004] {
        f.block(
            a,
            material,
            Vec3::new(0., base + edge_h * 0.5, z),
            Vec3::new(f.length, edge_h, 0.008),
            0.001,
        );
    }
    for x in [-f.length * 0.5 + 0.004, f.length * 0.5 - 0.004] {
        f.block(
            a,
            material,
            Vec3::new(x, base + edge_h * 0.5, 0.),
            Vec3::new(0.008, edge_h, f.width - 0.018),
            0.001,
        );
    }
    anchors
}

/// One closed board. The cut shoulder bears on the flange and its upper edge
/// clears the web. There is no second box passing through the lower face.
fn board(
    g: &mut Geometry,
    tile: CeilingTile,
    center: Vec2,
    pitch: Vec2,
    base: f32,
    cuts: &mut impl Rng,
) {
    let start = g.uvs.len();
    let full = pitch * 0.5 - (0.002 + tile.clearance_m);
    // Compress the bearing by 0.35 mm into the flange so contact never leaves
    // duplicate visible planes. The face sits below the tee only on tegular stock.
    let bearing = base + 0.004 - 0.00035;
    let bottom = bearing - tile.drop_m;
    let ring = |half: Vec2, y: f32| {
        let lo = center - half;
        let hi = center + half;
        [
            Vec3::new(lo.x, y, lo.y),
            Vec3::new(hi.x, y, lo.y),
            Vec3::new(hi.x, y, hi.y),
            Vec3::new(lo.x, y, hi.y),
        ]
    };
    let band = |g: &mut Geometry, a: [Vec3; 4], b: [Vec3; 4]| {
        for (i, outward) in [Vec3::NEG_Z, Vec3::X, Vec3::Z, Vec3::NEG_X]
            .into_iter()
            .enumerate()
        {
            let j = (i + 1) % 4;
            g.metric_quad([a[i], b[i], b[j], a[j]], outward - Vec3::Y);
        }
    };
    let shoulder = ring(full, bearing);
    let upper = ring(full, bottom + tile.thickness_m);
    if tile.edge == TileEdge::SquareLayIn {
        g.metric_quad(shoulder, Vec3::NEG_Y);
    } else {
        let neck = pitch * 0.5 - (tile.flange_width_m * 0.5 + tile.clearance_m);
        let bevel = if tile.edge == TileEdge::BeveledTegular {
            tile.drop_m * 0.65
        } else {
            0.
        };
        let face = ring(neck - bevel, bottom);
        let rebate = ring(neck, bearing);
        g.metric_quad(face, Vec3::NEG_Y);
        band(g, face, rebate);
        band(g, rebate, shoulder);
    }
    band(g, shoulder, upper);
    g.metric_quad(upper, Vec3::Y);

    super::super::tile::vary_cut(g, start, cuts);
}

#[cfg(test)]
mod tests;
