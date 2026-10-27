use super::*;

/// Species-conditioned envelopes, drawn before support and aisle admission.
/// Width and height are correlated: a bush/fern cannot independently draw the
/// smallest footprint and tallest stem. Upright cane/sword habits retain their
/// characteristic proportions. The minor axis never becomes a thin ribbon.
pub(crate) fn sample_size(variant: u32, tabletop: bool, rng: &mut impl Rng) -> Vec3 {
    let (width, ratio) = match variant % SPECIES {
        0 => (0.62..1.08, 1.20..2.35),
        1 => (0.70..1.16, 1.25..2.20),
        2 => (0.38..0.67, 1.65..3.60),
        3 => (0.68..1.18, 0.85..1.50),
        4 => (0.60..1.05, 0.62..1.10),
        5 => (0.48..0.85, 1.80..3.80),
        6 => (0.70..1.16, 1.25..2.15),
        7 => (0.30..0.58, 0.45..0.85),
        8 => (0.38..0.65, 0.85..2.60),
        9 => (0.40..0.70, 0.80..1.60),
        10 => (0.43..0.76, 2.00..3.80),
        11 => (0.50..0.90, 0.60..2.10),
        12 => (0.65..1.10, 1.05..1.85),
        13 => (0.60..0.98, 0.85..1.45),
        14 => (0.65..1.10, 1.45..2.50),
        15 => (0.65..1.15, 0.95..1.85),
        _ => (0.52..0.92, 0.90..1.50),
    };
    // Log sampling keeps small desk pots common without excluding broad
    // tabletop/sideboard plants. Surface admission fits the whole canopy.
    let width: f32 = if tabletop {
        rng.random_range(0.14_f32.ln()..0.48_f32.ln()).exp()
    } else {
        rng.random_range(width)
    };
    let depth = width * rng.random_range(0.92..1.08);
    let height =
        (width.min(depth) * rng.random_range(ratio)).min(if tabletop { 0.75_f32 } else { 2.12 });
    Vec3::new(width, height, depth)
}

/// Existing floor/tabletop plant proposals also need a compatible growth habit:
/// a two-metre narrow reserve must not turn into a vertically stretched rosette.
pub(crate) fn sample_species(size: Vec3, rng: &mut impl Rng) -> u32 {
    let ratio = size.y / size.x.min(size.z);
    let ranges = [
        (1.1, 6.0),
        (1.2, 5.5),
        (1.3, 8.0),
        (0.7, 3.0),
        (0.65, 1.9),
        (1.6, 7.0),
        (1.3, 6.0),
        (0.30, 1.7),
        (0.75, 6.0),
        (0.70, 4.0),
        (1.8, 8.0),
        (0.70, 4.0),
        (1.2, 5.0),
        (0.70, 3.0),
        (1.20, 5.0),
        (1.00, 4.0),
        (0.65, 3.5),
    ];
    let mut choices = [0u32; SPECIES as usize];
    let mut count = 0;
    for (i, (min, max)) in ranges.into_iter().enumerate() {
        if (min..=max).contains(&ratio) && (i != 7 || size.y <= 0.70 || ratio <= 1.1) {
            choices[count] = i as u32;
            count += 1;
        }
    }
    if count == 0 {
        if ratio < 1. {
            7
        } else {
            2
        }
    } else {
        choices[rng.random_range(0..count)]
    }
}
