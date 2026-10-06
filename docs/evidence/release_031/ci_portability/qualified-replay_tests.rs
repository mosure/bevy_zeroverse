//! Frozen transfer/mix expressions: never assume the round trip is an identity.
use super::*;

fn decode(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn encode(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

fn tint(color: [f32; 3], gain: f32) -> [f32; 3] {
    color.map(|c| encode((decode(c) * gain).clamp(0., 1.)))
}

fn mix(a: [f32; 3], b: [f32; 3], amount: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| encode(decode(a[i]) * (1. - amount) + decode(b[i]) * amount))
}

fn check(a: [f32; 3], b: [f32; 3]) {
    assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
}

#[track_caller]
fn check_arithmetic(a: [f32; 3], b: [f32; 3], context: std::fmt::Arguments<'_>) {
    // Rust permits arithmetic to choose a different NaN payload and sign,
    // even for equivalent expressions. Every non-NaN result remains exact.
    for (channel, (actual, expected)) in a.into_iter().zip(b).enumerate() {
        assert_eq!(
            actual.is_nan(),
            expected.is_nan(),
            "{context} channel {channel}: actual {:#x}, expected {:#x}",
            actual.to_bits(),
            expected.to_bits(),
        );
        if !expected.is_nan() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{context} channel {channel}",
            );
        }
    }
}

#[test]
fn carried_decode_replays_original_transfers_without_assuming_identity() {
    let mut reused_pow = 0;
    let mut changed_round_trip = 0;
    for i in 1..4096 {
        let input = [i as f32 / 4096., i as f32 / 8192., i as f32 / 6144.];
        let mut expected = input;
        let mut actual = ColorCarry::new(input);
        for _ in 0..5 {
            expected = tint(expected, 1.);
            actual = actual.tinted(1.);
            check(actual.encoded(), expected);
            for channel in 0..3 {
                if actual.encoded[channel].to_bits() == actual.input[channel].to_bits()
                    && actual.encoded[channel] > 0.04045
                {
                    reused_pow += 1;
                }
            }
            changed_round_trip +=
                usize::from(expected.map(f32::to_bits) != input.map(f32::to_bits));
            expected = mix(expected, [0.; 3], 0.);
            actual = actual.mixed([0.; 3], 0.);
            check(actual.encoded(), expected);
        }
    }
    assert!(
        reused_pow > 0,
        "the equality-guarded decode reuse was never exercised"
    );
    assert!(
        changed_round_trip > 0,
        "the corpus missed nonidentity round trips"
    );
}

#[test]
fn carried_decode_preserves_signed_zero_nan_payloads_and_transfer_boundaries() {
    let mut inputs = vec![
        -0.,
        0.,
        -f32::from_bits(1),
        f32::from_bits(1),
        -0.3,
        0.1,
        0.5,
        1.,
        -f32::MAX,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0xffc5_4321),
    ];
    for boundary in [0.04045f32, 0.0031308] {
        inputs.extend((-2i64..=2).map(|d| f32::from_bits((boundary.to_bits() as i64 + d) as u32)));
    }
    for &c in &inputs {
        for endpoint in [[0.; 3], [0.3, 0.7, 0.9], [f32::from_bits(0x7fc9_8765); 3]] {
            for (gain, amount) in [
                (1.0f32, 0.0f32),
                (0., -0.),
                (0.8, 0.23),
                (1.2, 1.),
                (f32::NAN, 0.),
            ] {
                let mut expected = [c, -c, c * 0.7];
                let mut actual = ColorCarry::new(expected);
                // Construction copies bits without arithmetic, including NaN payloads.
                check(actual.encoded(), expected);
                for iteration in 0..3 {
                    let prior = expected;
                    let exceptional = !gain.is_finite() || prior.iter().any(|c| !c.is_finite());
                    let production = super::super::super::field::tint(prior, gain);
                    expected = tint(expected, gain);
                    actual = actual.tinted(gain);
                    assert!(
                        !exceptional || !actual.has_input,
                        "exceptional tint retained a decode"
                    );
                    check_arithmetic(
                        actual.encoded(),
                        expected,
                        format_args!(
                            "tint iteration {iteration} c {:#x} gain {:#x} amount {:#x} prior {:?} endpoint {:?} production {:?}",
                            c.to_bits(),
                            gain.to_bits(),
                            amount.to_bits(),
                            prior.map(f32::to_bits),
                            endpoint.map(f32::to_bits),
                            production.map(f32::to_bits),
                        ),
                    );
                    let prior = expected;
                    let exceptional = !amount.is_finite()
                        || prior.iter().chain(&endpoint).any(|c| !c.is_finite());
                    let production = super::super::super::field::mix(prior, endpoint, amount);
                    expected = mix(expected, endpoint, amount);
                    actual = actual.mixed(endpoint, amount);
                    assert!(
                        !exceptional || !actual.has_input,
                        "exceptional mix retained a decode"
                    );
                    check_arithmetic(
                        actual.encoded(),
                        expected,
                        format_args!(
                            "mix iteration {iteration} c {:#x} gain {:#x} amount {:#x} prior {:?} endpoint {:?} production {:?}",
                            c.to_bits(),
                            gain.to_bits(),
                            amount.to_bits(),
                            prior.map(f32::to_bits),
                            endpoint.map(f32::to_bits),
                            production.map(f32::to_bits),
                        ),
                    );
                    let decoded_endpoint = endpoint.map(decode);
                    let exceptional = !amount.is_finite()
                        || expected
                            .iter()
                            .chain(&decoded_endpoint)
                            .any(|c| !c.is_finite());
                    expected = [0, 1, 2].map(|i| {
                        encode(decode(expected[i]) * (1. - amount) + decoded_endpoint[i] * amount)
                    });
                    actual = actual.mixed_linear(decoded_endpoint, amount);
                    assert!(
                        !exceptional || !actual.has_input,
                        "exceptional linear mix retained a decode"
                    );
                    check_arithmetic(
                        actual.encoded(),
                        expected,
                        format_args!(
                            "linear mix iteration {iteration} c {:#x} gain {:#x} amount {:#x} endpoint {:?}",
                            c.to_bits(),
                            gain.to_bits(),
                            amount.to_bits(),
                            endpoint.map(f32::to_bits),
                        ),
                    );
                }
            }
        }
    }
}
