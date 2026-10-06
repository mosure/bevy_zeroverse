//! Exact, bounded memoization of the quantized color transfer within one map.
//! Hash collisions replace a slot; they never substitute an approximate value.

const SLOTS: usize = 4096;

#[derive(Clone, Copy)]
struct Entry {
    bits: u32,
    byte: u8,
}

pub(super) struct ColorTransfer {
    entries: Box<[Entry; SLOTS]>,
}

impl ColorTransfer {
    pub(super) fn new() -> Self {
        Self {
            entries: Box::new(
                [Entry {
                    bits: u32::MAX,
                    byte: 0,
                }; SLOTS],
            ),
        }
    }

    pub(super) fn encode(&mut self, value: f32) -> u8 {
        let bits = value.to_bits();
        let index = (bits.wrapping_mul(0x9e37_79b9) >> 20) as usize;
        let entry = &mut self.entries[index];
        if entry.bits != bits {
            *entry = Entry {
                bits,
                byte: (super::linear_to_srgb(value) * 255.) as u8,
            };
        }
        entry.byte
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
