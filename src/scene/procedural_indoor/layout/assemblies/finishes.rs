//! A coordinated collection is a room-level prior, not a mandatory identity for
//! every chair and computer. Replacement pieces retain their own finish seed.
use super::*;

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct CollectionFinishes {
    pub seat_coherence: f32,
    pub device_coherence: f32,
}
pub fn parameters(seed: u64) -> CollectionFinishes {
    let mut r = stream(seed, 0x434f4c4c454354);
    CollectionFinishes {
        seat_coherence: if r.random_bool(0.24) {
            1.
        } else {
            r.random_range(0.20..0.94)
        },
        device_coherence: r.random_range(0.20..0.96),
    }
}
pub(crate) fn matches(collection: u64, object: u64, probability: f32) -> bool {
    stream(collection ^ object.rotate_left(23), 0x434f4c4c4d415443).random_bool(probability as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn collections_include_matching_and_replacement_furnishings() {
        let mut matching = 0;
        let mut mixed = 0;
        for seed in 0..128 {
            let p = parameters(seed);
            let count = (0..8)
                .filter(|i| matches(seed, seed * 13 + i, p.seat_coherence))
                .count();
            matching += usize::from(count == 8);
            mixed += usize::from((1..8).contains(&count));
        }
        assert!(
            matching > 20 && mixed > 50,
            "matching={matching}, mixed={mixed}"
        );
    }
}
