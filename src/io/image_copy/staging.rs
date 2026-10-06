//! One native view can map all its ordered attachments at once. Copy offsets are
//! already aligned by WebGPU's row pitch. This does not change the transferred
//! bytes, decoded planes, epoch barriers or maximum logical staging residency.

pub(super) fn combined_buffer_size(
    plane_bytes: usize,
    planes: usize,
    device_limit: u64,
) -> Option<usize> {
    if planes < 2 || plane_bytes == 0 {
        return None;
    }
    let bytes = plane_bytes.checked_mul(planes)?;
    (u64::try_from(bytes).ok()? <= device_limit).then_some(bytes)
}

pub(super) fn packed_planes(
    mapped: &[u8],
    row_bytes: usize,
    padded_row_bytes: usize,
    rows: usize,
    planes: usize,
) -> Vec<Vec<u8>> {
    let plane_bytes = padded_row_bytes * rows;
    (0..planes)
        .map(|index| {
            let offset = index * plane_bytes;
            super::super::packed_readback_rows(
                &mapped[offset..offset + plane_bytes],
                row_bytes,
                padded_row_bytes,
                rows,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combined_staging_is_bounded_by_actual_device_buffer_limit() {
        assert_eq!(combined_buffer_size(4096, 6, 24576), Some(24576));
        assert_eq!(combined_buffer_size(4096, 6, 24575), None);
        assert_eq!(combined_buffer_size(0, 6, 24576), None);
        assert_eq!(combined_buffer_size(4096, 1, 24576), None);
        assert_eq!(combined_buffer_size(4096, 0, 24576), None);
        assert_eq!(combined_buffer_size(usize::MAX, 2, u64::MAX), None);
        // Six 4K float attachments must retain separate per-plane buffers on
        // an adapter with a 256MiB max-buffer limit rather than overallocate.
        assert_eq!(combined_buffer_size(4096 * 4096 * 16, 6, 1 << 28), None);
    }

    #[test]
    fn one_mapping_replays_ordered_original_plane_and_row_bytes() {
        for (row, pitch, rows, planes) in [
            (16, 256, 3, 6),
            (256, 256, 7, 5),
            (641 * 16, 10496, 479, 6),
            (512 * 16, 512 * 16, 512, 6),
        ] {
            let stride = pitch * rows;
            let mapped: Vec<_> = (0..stride * planes)
                .map(|i| ((i / stride * 31 + i * 17 + i / pitch) % 251) as u8)
                .collect();
            let expected: Vec<Vec<u8>> = (0..planes)
                .map(|index| {
                    // Independent literal original row-stripping loop.
                    let mut bytes = Vec::with_capacity(row * rows);
                    for y in 0..rows {
                        let offset = index * stride + y * pitch;
                        bytes.extend_from_slice(&mapped[offset..offset + row]);
                    }
                    bytes
                })
                .collect();
            assert_eq!(packed_planes(&mapped, row, pitch, rows, planes), expected);
        }
    }
}
