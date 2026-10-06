//! Reuse an exact earlier sRGB decode only when the next encoded input has the
//! same finite f32 bits. Every intervening encode and reflectance operation is
//! retained: an sRGB round trip is never assumed to be an identity.
use super::super::{field, linear_to_srgb, program::PreparedMineral, srgb_to_linear};

pub(in super::super) struct ColorCarry {
    encoded: [f32; 3],
    input: [f32; 3],
    decoded_input: [f32; 3],
    has_input: bool,
}

impl ColorCarry {
    pub(in super::super) fn new(encoded: [f32; 3]) -> Self {
        Self {
            encoded,
            input: [0.; 3],
            decoded_input: [0.; 3],
            has_input: false,
        }
    }

    #[inline]
    fn decoded(&self) -> [f32; 3] {
        std::array::from_fn(|i| {
            if self.has_input && self.encoded[i].to_bits() == self.input[i].to_bits() {
                self.decoded_input[i]
            } else {
                srgb_to_linear(self.encoded[i])
            }
        })
    }

    #[inline]
    fn remember(self, encoded: [f32; 3], decoded: [f32; 3]) -> Self {
        Self {
            encoded,
            input: self.encoded,
            decoded_input: decoded,
            has_input: true,
        }
    }

    #[inline]
    pub(in super::super) fn tinted(self, gain: f32) -> Self {
        // With multiple NaN operands even an unchanged multiply can propagate
        // a different payload after inlining. Retain the original helper for
        // exceptional inputs and do not carry any decode through that path.
        if !gain.is_finite() || self.encoded.iter().any(|c| !c.is_finite()) {
            return Self::new(field::tint(self.encoded, gain));
        }
        let decoded = self.decoded();
        let encoded = decoded.map(|c| linear_to_srgb((c * gain).clamp(0., 1.)));
        self.remember(encoded, decoded)
    }

    #[inline]
    pub(in super::super) fn mixed(self, endpoint: [f32; 3], amount: f32) -> Self {
        if !amount.is_finite() || self.encoded.iter().chain(&endpoint).any(|c| !c.is_finite()) {
            return Self::new(field::mix(self.encoded, endpoint, amount));
        }
        let decoded = self.decoded();
        let encoded = [0, 1, 2].map(|i| {
            linear_to_srgb(decoded[i] * (1. - amount) + srgb_to_linear(endpoint[i]) * amount)
        });
        self.remember(encoded, decoded)
    }

    #[inline]
    pub(in super::super) fn mixed_linear(self, endpoint: [f32; 3], amount: f32) -> Self {
        if !amount.is_finite() || self.encoded.iter().chain(&endpoint).any(|c| !c.is_finite()) {
            return Self::new(PreparedMineral::mixed_linear(
                self.encoded.map(srgb_to_linear),
                endpoint,
                amount,
            ));
        }
        let decoded = self.decoded();
        let encoded =
            [0, 1, 2].map(|i| linear_to_srgb(decoded[i] * (1. - amount) + endpoint[i] * amount));
        self.remember(encoded, decoded)
    }

    #[inline]
    pub(in super::super) fn encoded(&self) -> [f32; 3] {
        self.encoded
    }
}

#[cfg(test)]
mod replay_tests;
