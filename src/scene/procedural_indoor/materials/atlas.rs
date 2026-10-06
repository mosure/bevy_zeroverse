//! Exact independent texel rows on the existing native preparation workers.
//! Four bands per pass share one map-owned immutable context and disjoint output;
//! no extra full-map buffers, pools or retained per-seed state are introduced.

#[cfg(not(target_arch = "wasm32"))]
const BANDS: usize = 4;

pub(super) fn texels(
    size: usize,
    parallel: bool,
    heights: &mut [f32],
    colors: &mut [u8],
    data: &mut [u8],
    evaluate: impl Fn(usize, &mut [f32], &mut [u8], &mut [u8]) + Sync,
) {
    #[cfg(not(target_arch = "wasm32"))]
    if parallel && size >= 256 {
        let rows = size.div_ceil(BANDS);
        let texels = rows * size;
        super::super::preparation::workers::pool().scope(|scope| {
            for (band, ((heights, colors), data)) in heights
                .chunks_mut(texels)
                .zip(colors.chunks_mut(texels * 4))
                .zip(data.chunks_mut(texels * 4))
                .enumerate()
            {
                let evaluate = &evaluate;
                scope.spawn(async move { evaluate(band * rows, heights, colors, data) });
            }
        });
        return;
    }
    #[cfg(target_arch = "wasm32")]
    let _ = (size, parallel);
    evaluate(0, heights, colors, data);
}

pub(super) fn normals(
    size: usize,
    parallel: bool,
    normals: &mut [u8],
    evaluate: impl Fn(usize, &mut [u8]) + Sync,
) {
    #[cfg(not(target_arch = "wasm32"))]
    if parallel && size >= 256 {
        let rows = size.div_ceil(BANDS);
        super::super::preparation::workers::pool().scope(|scope| {
            for (band, normals) in normals.chunks_mut(rows * size * 4).enumerate() {
                let evaluate = &evaluate;
                scope.spawn(async move { evaluate(band * rows, normals) });
            }
        });
        return;
    }
    #[cfg(target_arch = "wasm32")]
    let _ = (size, parallel);
    evaluate(0, normals);
}

#[cfg(test)]
mod replay_tests;
