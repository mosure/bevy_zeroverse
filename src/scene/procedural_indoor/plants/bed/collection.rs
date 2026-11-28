use super::*;

/// The planting brief is shared with the surrounding containers. A shrub or
/// fan palm can be the main specimen in more than one of these collections.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) enum Collection {
    Woodland,
    Tropical,
    DryGarden,
}

impl Collection {
    pub(crate) fn sample(rng: &mut impl Rng) -> Self {
        match rng.random_range(0..3) {
            0 => Self::Woodland,
            1 => Self::Tropical,
            _ => Self::DryGarden,
        }
    }

    pub(crate) fn specimens(self) -> &'static [u32] {
        match self {
            Self::Woodland => &[6, 13, 10, 4],
            Self::Tropical => &[15, 1, 14, 13, 3],
            Self::DryGarden => &[8, 9, 7, 13, 15],
        }
    }

    pub(super) fn companions(self) -> &'static [u32] {
        match self {
            Self::Woodland => &[13, 4, 16, 12, 0],
            Self::Tropical => &[16, 3, 14, 4, 12],
            Self::DryGarden => &[7, 9, 2, 8],
        }
    }

    pub(crate) fn bed_variant(self, species: u32, version: u32) -> u32 {
        if version < VERSION {
            return species;
        }
        let group = match self {
            Self::Woodland => 1,
            Self::Tropical => 2,
            Self::DryGarden => 3,
        };
        group * SPECIES + species
    }

    pub(super) fn from_bed(variant: u32) -> Self {
        match variant / SPECIES {
            1 => Self::Woodland,
            2 => Self::Tropical,
            3 => Self::DryGarden,
            // A caller can still supply a plain species as the entire variant.
            _ => match variant % SPECIES {
                2 | 7..=9 => Self::DryGarden,
                1 | 3 | 14 | 15 => Self::Tropical,
                _ => Self::Woodland,
            },
        }
    }
}
