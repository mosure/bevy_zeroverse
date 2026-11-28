//! Rasterization snaps projected vertices to a subpixel grid. At a T-junction
//! the short edge can round to a different side of the long edge and expose
//! the outside of the hall. Give adjoining floor finishes identical edge segments.
use super::*;
use std::collections::BTreeMap;

#[derive(Default)]
struct GridLines([BTreeMap<u32, Vec<f32>>; 2]);

impl GridLines {
    fn new(a: &Assembly) -> Self {
        let mut lines = Self::default();
        for g in a.parts.values() {
            for &[x, _, z] in &g.positions {
                for (axis, fixed, along) in [(0, x, z), (1, z, x)] {
                    lines.0[axis].entry(key(fixed)).or_default().push(along);
                }
            }
        }
        for offsets in lines.0.iter_mut().flat_map(|axis| axis.values_mut()) {
            offsets.sort_unstable_by(f32::total_cmp);
            offsets.dedup();
        }
        lines
    }

    fn edge(&self, from: Vec2, to: Vec2, boundary: &mut Vec<u32>, g: &mut Geometry) {
        let axis = if from.x == to.x {
            0
        } else if from.y == to.y {
            1
        } else {
            return;
        };
        let (start, end) = (from[1 - axis], to[1 - axis]);
        let line = &self.0[axis][&key(from[axis])];
        let first = line.partition_point(|&v| v <= start.min(end));
        let last = line.partition_point(|&v| v < start.max(end));
        let insert = |offset: f32| {
            let mut p = from;
            p[1 - axis] = offset;
            boundary.push(vertex(g, p));
        };
        if start < end {
            line[first..last].iter().copied().for_each(insert);
        } else if end < start {
            line[first..last].iter().rev().copied().for_each(insert);
        }
    }
}

fn key(v: f32) -> u32 {
    if v == 0. {
        0
    } else {
        v.to_bits()
    }
}

fn vertex(g: &mut Geometry, p: Vec2) -> u32 {
    let i = g.positions.len() as u32;
    g.positions.push([p.x, 0., p.y]);
    g.normals.push([0., 1., 0.]);
    g.uvs.push([p.x, -p.y]);
    i
}

/// The sports floor is the first assembly in the envelope. Its three batches
/// must share junctions even where the primary and practice pigments meet.
pub(super) fn join(a: &mut Assembly) {
    let lines = GridLines::new(a);
    let mut boundary = Vec::new();
    for g in a.parts.values_mut() {
        let original = std::mem::take(&mut g.indices);
        g.indices.reserve(original.len());
        for &tri in original.as_chunks::<3>().0 {
            let p = tri.map(|i| Vec3::from_array(g.positions[i as usize]).xz());
            boundary.clear();
            for e in 0..3 {
                boundary.push(tri[e]);
                lines.edge(p[e], p[(e + 1) % 3], &mut boundary, g);
            }
            if boundary.len() == 3 {
                g.indices.extend(tri);
            } else {
                // A fan anchored on a boundary vertex can drop collinear cuts
                // on its own edge. An interior point preserves all three chains.
                let center = vertex(
                    g,
                    ((p[0].as_dvec2() + p[1].as_dvec2() + p[2].as_dvec2()) / 3.).as_vec2(),
                );
                let mut prev = *boundary.last().unwrap();
                for &next in &boundary {
                    g.indices.extend([center, prev, next]);
                    prev = next;
                }
            }
        }
    }
}
