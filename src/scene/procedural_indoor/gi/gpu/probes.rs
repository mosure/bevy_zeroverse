//! Relocated grid slots use the source probe's position and random sequence.
//! Trace that sequence once and scatter its six completed lobes to each slot.
use bevy::prelude::Vec3;

pub(super) struct ProbeDispatch {
    /// One xyz-bit/next-slot record per grid slot, then four source indices per
    /// record. Both tables share the existing read-only shader binding.
    pub records: Vec<[u32; 4]>,
    pub sources: u32,
    pub relocated: usize,
}

impl ProbeDispatch {
    pub(super) fn new(points: &[Vec3], valid: &[bool]) -> Self {
        debug_assert_eq!(points.len(), valid.len());
        let mut sources: Vec<_> = (0..points.len()).filter(|&i| valid[i]).collect();
        let relocated = points.len() - sources.len();
        let mut records = Vec::with_capacity(points.len() + points.len().div_ceil(4));
        records.extend(
            points
                .iter()
                .map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits(), u32::MAX]),
        );
        if sources.is_empty() {
            // The original fallback traces each slot in place if all are solid.
            sources.extend(0..points.len());
        } else if relocated != 0 {
            for (i, p) in points.iter().enumerate().filter(|(i, _)| !valid[*i]) {
                let source = *sources
                    .iter()
                    .min_by(|&&a, &&b| {
                        points[a]
                            .distance_squared(*p)
                            .total_cmp(&points[b].distance_squared(*p))
                    })
                    .expect("at least one valid source");
                records[i][3] = records[source][3];
                records[source][3] = i as u32;
            }
        }
        for group in sources.chunks(4) {
            let mut indices = [u32::MAX; 4];
            for (dst, &i) in indices.iter_mut().zip(group) {
                *dst = i as u32;
            }
            records.push(indices);
        }
        Self {
            records,
            sources: sources.len() as u32,
            relocated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_replays_every_destination_and_original_rng_index_exactly_once() {
        let points: Vec<_> = (0..8)
            .map(|i| Vec3::new((i % 4) as f32, -0.0, (i / 4) as f32))
            .collect();
        for mask in 0..256 {
            let valid: Vec<_> = (0..8).map(|i| mask & (1 << i) != 0).collect();
            let plan = ProbeDispatch::new(&points, &valid);
            let mut owners = vec![None; points.len()];
            for group in 0..plan.sources as usize {
                let source = plan.records[points.len() + group / 4][group % 4] as usize;
                assert_eq!(
                    plan.records[source][..3],
                    points[source].to_array().map(f32::to_bits)
                );
                let mut slot = source as u32;
                while slot != u32::MAX {
                    assert!(owners[slot as usize].replace(source).is_none());
                    slot = plan.records[slot as usize][3];
                }
            }
            let inside = valid.iter().filter(|&&v| !v).count();
            assert_eq!(plan.relocated, inside);
            assert_eq!(
                plan.sources as usize,
                if inside == points.len() {
                    inside
                } else {
                    points.len() - inside
                }
            );
            for (i, p) in points.iter().enumerate() {
                let source = if valid[i] {
                    i
                } else {
                    (0..points.len())
                        .filter(|&j| valid[j])
                        .min_by(|&a, &b| {
                            points[a]
                                .distance_squared(*p)
                                .total_cmp(&points[b].distance_squared(*p))
                        })
                        .unwrap_or(i)
                };
                assert_eq!(owners[i], Some(source), "mask {mask:#x}, slot {i}");
            }
        }
    }
}
