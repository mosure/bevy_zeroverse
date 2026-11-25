//! Quantize linear color for sRGB8 mips. Where a linear interval straddles
//! an sRGB8 boundary, use the renderer's scalar transfer for the exact value.
const INTERVALS: usize = 8192;
pub(super) struct ColorTransfer {
    endpoints: &'static [u8; INTERVALS + 1],
    boundary_cache: [(u32, u8); 128],
}
impl ColorTransfer {
    pub(super) fn new() -> Self {
        static ENDPOINTS: std::sync::OnceLock<[u8; INTERVALS + 1]> = std::sync::OnceLock::new();
        Self {
            endpoints: ENDPOINTS
                .get_or_init(|| std::array::from_fn(|i| quantize(i as f32 / INTERVALS as f32))),
            boundary_cache: [(u32::MAX, 0); 128],
        }
    }

    pub(super) fn encode(&mut self, value: f32) -> u8 {
        let index = (value * INTERVALS as f32) as usize;
        if index < INTERVALS {
            let low = self.endpoints[index];
            // The quantized transfer is monotone. Equal endpoints prove the
            // entire interval maps to this byte, including the sRGB toe.
            if low == self.endpoints[index + 1] {
                return low;
            }
        }
        let bits = value.to_bits();
        let slot = &mut self.boundary_cache[(bits.wrapping_mul(0x9e37_79b9) >> 25) as usize];
        if slot.0 != bits {
            *slot = (bits, quantize(value));
        }
        slot.1
    }
}
fn quantize(v: f32) -> u8 {
    (super::linear_to_srgb(v) * 255.) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_edges_and_exceptional_values_match_scalar_transfer() {
        let mut transfer = ColorTransfer::new();
        for i in 0..=INTERVALS {
            let edge = i as f32 / INTERVALS as f32;
            for value in [
                edge.next_down(),
                edge,
                edge.next_up(),
                edge + 0.5 / INTERVALS as f32,
            ] {
                assert_eq!(transfer.encode(value), quantize(value));
            }
        }
        for value in [-1., 2., f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            assert_eq!(transfer.encode(value), quantize(value));
        }
    }

    #[test]
    fn collisions_and_repeated_values_preserve_original_color_quantization() {
        let mut transfer = ColorTransfer::new();
        let linear = super::super::srgb8_table();
        for round in 0..3 {
            for a in 0..256 {
                for b in 0..256 {
                    let mut sum = 0.;
                    for byte in [a, b, (a + b + round) % 256, (a * 7 + b * 13) % 256] {
                        sum += linear[byte];
                    }
                    let value = sum * 0.25;
                    assert_eq!(
                        transfer.encode(value),
                        (super::super::linear_to_srgb(value) * 255.) as u8,
                    );
                }
            }
        }
        for value in [0., -0., 1., f32::MIN_POSITIVE, f32::EPSILON] {
            assert_eq!(
                transfer.encode(value),
                (super::super::linear_to_srgb(value) * 255.) as u8,
            );
        }
    }
}
