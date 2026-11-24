//! One closed roof boundary. The lower face follows the board edges and goes
//! up into equipment pockets. An opaque slab across those openings would hide
//! recessed lenses in RGB and give the wrong surface to the depth renderer.
use super::*;
use crate::scene::procedural_indoor::{
    envelope::construction::triangle, materials::acoustic_tile::GRID_LABEL,
};

use super::super::sheet::{band, ring};

fn bay(
    a: &mut Assembly,
    grid: &CeilingGrid,
    cell: UVec2,
    lo: Vec2,
    hi: Vec2,
    h: f32,
    cuts: &mut impl Rng,
) {
    let inset = grid.tile.flange_width_m * 0.5;
    let at = |x: f32, y: f32| ring(lo + x, hi - x, h + y);
    let metal = a.part(Surface::Metal, GRID_LABEL);
    band(metal, at(0., 0.), at(inset, 0.));
    band(metal, at(inset, 0.), at(inset, BEARING_M));
    let air = grid.air_terminals.iter().find(|a| a.cell == cell);
    if air.is_some() || grid.troffers.iter().any(|m| m.cell == cell) {
        // The housing overlaps the top of the lip by a millimetre. Widen the
        // pocket behind it so its sides do not coincide with the roof mesh.
        let pocket = inset - 0.003;
        band(metal, at(inset, BEARING_M), at(pocket, BEARING_M));
        let g = a.part(Surface::Ink, "ceiling");
        band(g, at(pocket, BEARING_M), at(pocket, POCKET_M));
        g.metric_quad(at(pocket, POCKET_M), Vec3::NEG_Y);
        if let Some(air) = air {
            super::terminals::build(a, air, (lo + hi) * 0.5, hi - lo - 2. * inset, h);
        }
    } else {
        let g = a.part(Surface::Ceiling, grid.tile.facing.label());
        let start = g.uvs.len();
        let seat = at(inset, BEARING_M);
        if grid.tile.edge == TileEdge::SquareLayIn {
            g.metric_quad(seat, Vec3::NEG_Y);
        } else {
            let neck = inset + grid.tile.clearance_m;
            let bevel = if grid.tile.edge == TileEdge::BeveledTegular {
                grid.tile.drop_m * 0.65
            } else {
                0.
            };
            let face = at(neck + bevel, BEARING_M - grid.tile.drop_m);
            band(g, seat, at(neck, BEARING_M));
            band(g, at(neck, BEARING_M), face);
            g.metric_quad(face, Vec3::NEG_Y);
        }
        super::super::tile::vary_cut(g, start, cuts);
    }
}

/// Site-cut border panels terminate flush at the wall. Clip runner and board
/// as disjoint planar regions, including on notched and chamfered envelopes.
fn border(
    a: &mut Assembly,
    grid: &CeilingGrid,
    lo: Vec2,
    hi: Vec2,
    h: f32,
    triangles: &[([Vec2; 3], Vec2, Vec2)],
    cuts: &mut impl Rng,
) {
    let s = grid.tile.flange_width_m * 0.5;
    let inner_lo = lo + s;
    let inner_hi = hi - s;
    let regions = [
        (lo, Vec2::new(inner_lo.x, hi.y)),
        (Vec2::new(inner_hi.x, lo.y), hi),
        (
            Vec2::new(inner_lo.x, lo.y),
            Vec2::new(inner_hi.x, inner_lo.y),
        ),
        (
            Vec2::new(inner_lo.x, inner_hi.y),
            Vec2::new(inner_hi.x, hi.y),
        ),
        (inner_lo, inner_hi),
    ];
    let label = grid.tile.facing.label();
    let start = a.part(Surface::Ceiling, label).uvs.len();
    for &(t, tlo, thi) in triangles {
        if hi.cmple(tlo).any() || lo.cmpge(thi).any() {
            continue;
        }
        for (i, &(l, r)) in regions.iter().enumerate() {
            if r.cmple(tlo).any() || l.cmpge(thi).any() {
                continue;
            }
            let p = polygon::clip_box(&t, l, r);
            let wp = |v: Vec2| Vec3::new(v.x, h, v.y);
            let g = if i == 4 {
                a.part(Surface::Ceiling, label)
            } else {
                a.part(Surface::Metal, GRID_LABEL)
            };
            for j in 1..p.len().saturating_sub(1) {
                triangle(g, [wp(p[0]), wp(p[j]), wp(p[j + 1])], Vec3::NEG_Y);
            }
        }
    }
    let g = a.part(Surface::Ceiling, label);
    if g.uvs.len() > start {
        super::super::tile::vary_cut(g, start, cuts);
    }
}

pub(crate) async fn build<F: std::future::Future<Output = ()>>(
    a: &mut Assembly,
    scene: &IndoorManifest,
    grid: &CeilingGrid,
    yield_if_due: impl Fn() -> F,
) {
    let e = scene.envelope.as_ref().unwrap();
    let size = scene.room_size;
    let h = size.y;
    let triangles: Vec<_> = polygon::triangles(&e.footprint)
        .into_iter()
        .map(|t| (t, t[0].min(t[1]).min(t[2]), t[0].max(t[1]).max(t[2])))
        .collect();
    let roof = a.part(Surface::Ceiling, "ceiling");
    for (t, _, _) in &triangles {
        triangle(roof, t.map(|v| Vec3::new(v.x, h + 0.14, v.y)), Vec3::Y);
    }
    for (u, v) in polygon::edges(&e.footprint) {
        let p = Vec3::new(u.x, h, u.y);
        let q = Vec3::new(v.x, h, v.y);
        roof.metric_quad(
            [p, q, q + Vec3::Y * 0.14, p + Vec3::Y * 0.14],
            Vec3::new(v.y - u.y, 0., u.x - v.x),
        );
    }
    let mut cuts = stream(scene.seed, u64::from_le_bytes(*b"RoomCuts"));
    for z in 0..grid.bays.y {
        for x in 0..grid.bays.x {
            let cell = UVec2::new(x, z);
            let (lo, hi) = grid.cell_bounds(size.xz(), cell);
            if grid.profiled_bay(scene, lo, hi) {
                bay(a, grid, cell, lo, hi, h, &mut cuts);
            } else {
                border(a, grid, lo, hi, h, &triangles, &mut cuts);
            }
        }
        yield_if_due().await;
    }
}
