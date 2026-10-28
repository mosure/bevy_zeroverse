//! Shared purchasing/use preferences with local belongings. Task membership
//! stays physical; category probabilities no longer repeat uniformly in every room.
use super::*;

pub const CATALOG: [&[ObjectKind]; 6] = {
    use ObjectKind::*;
    [
        &[
            Stapler,
            TapeDispenser,
            DocumentTray,
            DeskOrganizer,
            Pencil,
            Notepad,
            Notebook,
            Phone,
            PenHolder,
        ],
        &[Books, Notebook, Plant, Vase, DeskLamp, Notepad],
        &[
            Mug,
            CoffeeCup,
            WaterBottle,
            SodaCan,
            TissueBox,
            Bowl,
            Kettle,
            CoffeeMaker,
        ],
        &[Speaker, Headphones, DeskFan, Phone, Printer, StorageBox],
        &[TissueBox, Vase, Bowl, Phone, Plant, Bag],
        &[
            DocumentTray,
            StorageBox,
            Books,
            Printer,
            Notebook,
            DeskOrganizer,
        ],
    ]
};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CatalogProgram {
    /// Per-task probabilities, in CATALOG order. Unused tail entries are zero.
    pub probabilities: [[f32; 9]; 6],
    /// Shared room preference versus local surface preference; neither is zero.
    pub room_coherence: f32,
}

pub fn parameters(scene_seed: u64, support_seed: u64) -> CatalogProgram {
    let mut room = stream(scene_seed, 0x434154414c4f4752);
    let mut local = stream(support_seed, 0x434154414c4f474c);
    let coherence = room.random_range(0.25..0.82);
    let probabilities = std::array::from_fn(|task| {
        let mut row = [0.; 9];
        for weight in row.iter_mut().take(CATALOG[task].len()) {
            let shared = room.random::<f32>().powi(2);
            let personal = local.random::<f32>().powi(2);
            // Positive floor retains rare categories. Mixture, not a product,
            // prevents a single object kind from overwhelming a work surface.
            *weight = 0.12 + coherence * shared + (1. - coherence) * personal;
        }
        let sum: f32 = row.iter().sum();
        row.iter_mut().for_each(|w| *w /= sum);
        row
    });
    CatalogProgram {
        probabilities,
        room_coherence: coherence,
    }
}

impl CatalogProgram {
    pub fn sample(&self, task: usize, rng: &mut impl Rng) -> ObjectKind {
        let task = task.min(CATALOG.len() - 1);
        let kinds = CATALOG[task];
        let mut value = rng.random::<f32>();
        let slot = self.probabilities[task][..kinds.len()]
            .iter()
            .position(|w| {
                value -= w;
                value <= 0.
            })
            .unwrap_or(kinds.len() - 1);
        kinds[slot]
    }

    /// Normalized categorical entropy of the proposal prior, not accepted
    /// objects, geometric diversity or independent dataset size.
    pub fn entropy(&self, task: usize) -> f32 {
        -self.probabilities[task]
            .iter()
            .filter(|&&p| p > 0.)
            .map(|p| p * p.ln())
            .sum::<f32>()
            / (CATALOG[task].len() as f32).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    #[test]
    fn catalogs_replay_cover_every_member_and_vary_without_collapsing() {
        let mut dominant: [BTreeSet<usize>; 6] = std::array::from_fn(|_| BTreeSet::new());
        let mut entropy = BTreeSet::new();
        let mut sampled: [BTreeSet<usize>; 6] = std::array::from_fn(|_| BTreeSet::new());
        for seed in 0..256 {
            let p = parameters(seed, seed ^ 91);
            assert_eq!(p, parameters(seed, seed ^ 91));
            assert_ne!(p, parameters(seed, seed ^ 92));
            for task in 0..6 {
                let row = &p.probabilities[task];
                assert!((row.iter().sum::<f32>() - 1.).abs() < 1e-5);
                assert!(row[..CATALOG[task].len()]
                    .iter()
                    .all(|&w| w > 0. && w < 0.66));
                assert!(row[CATALOG[task].len()..].iter().all(|&w| w == 0.));
                assert!((0.65..=1.00001).contains(&p.entropy(task)));
                entropy.insert((p.entropy(task) * 1000.) as u32);
                dominant[task].insert(
                    row.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0,
                );
                let mut r = stream(seed, 31);
                for _ in 0..24 {
                    sampled[task].insert(p.sample(task, &mut r) as usize);
                }
            }
        }
        assert!(entropy.len() > 100);
        for task in 0..6 {
            assert_eq!(dominant[task].len(), CATALOG[task].len());
            assert_eq!(
                sampled[task],
                CATALOG[task].iter().map(|&k| k as usize).collect()
            );
        }
    }
}
