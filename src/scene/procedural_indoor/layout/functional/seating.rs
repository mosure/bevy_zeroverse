//! Shared lecture-tier and seating sampling. Architecture and furniture consume
//! identical row boundaries; local disorder never changes a tier's support.
use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LectureSeating {
    pub seed: u64,
    pub regular: bool,
    pub row_pitch_variation: f32,
    pub seat_pitch_variation: f32,
    pub stagger_metres: f32,
    pub loose_position_metres: f32,
    pub loose_yaw_radians: f32,
    pub linked_row_fraction: f32,
    pub facing_strength: f32,
}
impl Default for LectureSeating {
    fn default() -> Self {
        Self {
            seed: 0,
            regular: true,
            row_pitch_variation: 0.,
            seat_pitch_variation: 0.,
            stagger_metres: 0.,
            loose_position_metres: 0.,
            loose_yaw_radians: 0.,
            linked_row_fraction: 1.,
            facing_strength: 0.5,
        }
    }
}
impl LectureSeating {
    pub(super) fn sample(seed: u64, zone: usize) -> Self {
        let mut r = stream(seed, 0x4c454354555245 + zone as u64);
        if r.random_bool(0.20) {
            return Self {
                seed: r.random(),
                ..Default::default()
            };
        }
        Self {
            seed: r.random(),
            regular: false,
            row_pitch_variation: r.random_range(0.05..0.24),
            seat_pitch_variation: r.random_range(0.06..0.30),
            stagger_metres: r.random_range(0.08..0.38),
            loose_position_metres: r.random_range(0.025..0.13),
            loose_yaw_radians: r.random_range(0.09..0.38),
            linked_row_fraction: r.random_range(0.08..0.55),
            facing_strength: r.random_range(0.25..0.9),
        }
    }
    pub(super) fn rows(&self, start: f32, end: f32, pitch: f32) -> Vec<[f32; 2]> {
        self.rows_with_clearance(start, end, pitch, 0.)
    }
    pub(super) fn rows_with_clearance(
        &self,
        start: f32,
        end: f32,
        pitch: f32,
        minimum: f32,
    ) -> Vec<[f32; 2]> {
        let mut r = stream(self.seed, 0x524f5753);
        let mut front = start;
        let mut rows = Vec::new();
        for _ in 0..7 {
            let depth = minimum
                + (pitch - minimum).max(0.)
                    * (1. + r.random_range(-self.row_pitch_variation..=self.row_pitch_variation));
            let back = front + depth;
            if back > end + 1e-5 {
                break;
            }
            rows.push([front, back]);
            front = back;
        }
        rows
    }
    pub(super) fn bank(&self, row: usize, side: f32) -> SeatingBank {
        let mut r = stream(
            self.seed,
            0x42414e4b + row as u64 * 2 + u64::from(side > 0.),
        );
        let linked = self.regular || r.random_bool(self.linked_row_fraction as f64);
        SeatingBank {
            rng: r,
            linked,
            stagger: self.stagger_metres,
            pitch_variation: if linked {
                0.
            } else {
                self.seat_pitch_variation
            },
            jitter: if linked {
                0.
            } else {
                self.loose_position_metres
            },
            yaw: if linked { 0. } else { self.loose_yaw_radians },
        }
    }
}
pub(super) struct SeatingBank {
    rng: ChaCha8Rng,
    pub linked: bool,
    stagger: f32,
    pitch_variation: f32,
    jitter: f32,
    yaw: f32,
}
impl SeatingBank {
    pub(super) fn shift(&mut self) -> f32 {
        self.rng.random_range(-self.stagger..=self.stagger)
    }
    pub(super) fn increment(&mut self, pitch: f32) -> f32 {
        pitch
            * (1.
                + self
                    .rng
                    .random_range(-self.pitch_variation..=self.pitch_variation))
    }
    pub(super) fn pose(&mut self) -> [f32; 3] {
        [
            self.rng.random_range(-self.jitter..=self.jitter),
            self.rng.random_range(-self.jitter..=self.jitter),
            self.rng.random_range(-self.yaw..=self.yaw),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lecture_rows_are_replayable_contiguous_and_not_usually_uniform() {
        let mut regular = 0;
        let mut varying = 0;
        for seed in 0..512 {
            let p = LectureSeating::sample(seed, 0);
            assert_eq!(p, LectureSeating::sample(seed, 0));
            let rows = p.rows(-3., 9., 1.5);
            assert_eq!(rows, p.rows(-3., 9., 1.5));
            regular += usize::from(p.regular);
            varying += usize::from(
                rows.windows(2)
                    .any(|r| ((r[0][1] - r[0][0]) - (r[1][1] - r[1][0])).abs() > 0.04),
            );
            for (i, [lo, hi]) in rows.iter().copied().enumerate() {
                assert!(hi > lo && hi <= 9.00001);
                if i > 0 {
                    assert_eq!(rows[i - 1][1], lo);
                }
            }
        }
        assert!((60..160).contains(&regular));
        assert!(varying > 300);
    }

    #[test]
    fn lecture_seating_has_pose_spacing_variance_and_preserves_support_and_routes() {
        let mut rooms = Vec::new();
        let mut loose = 0;
        let mut fixed = 0;
        let mut varied_pose_rooms = 0;
        let mut varied_gap_rooms = 0;
        for seed in 0..64 {
            let scene = IndoorManifest::generate_with_humans(
                seed,
                IndoorLayout::LectureHall,
                0.65,
                0,
                0.25,
            )
            .unwrap();
            scene.validate_public_programs().unwrap();
            super::super::super::super::validation::validate_layout(&scene).unwrap();
            let chairs: Vec<_> = scene
                .objects
                .iter()
                .filter(|o| o.kind == ObjectKind::Chair && !o.neighbor)
                .collect();
            assert!(chairs.len() >= 8, "seed {seed}: {} chairs", chairs.len());
            let program = &scene.functional_programs()[0];
            let p = &program.parameters.lecture_seating;
            let tiers = program
                .parameters
                .lecture_rows(program.min.y + 3., program.max.y - 1.);
            let center = (program.min.x + program.max.x) * 0.5;
            let mut residuals = Vec::new();
            let mut varied_gap = false;
            for &[front, back] in &tiers {
                let seats: Vec<_> = chairs
                    .iter()
                    .filter(|o| o.position.z >= front && o.position.z < back)
                    .collect();
                for o in &seats {
                    let facing = ((o.position.x - center) / (front - program.min.y + 1.)).atan()
                        * p.facing_strength;
                    residuals.push(o.yaw - facing);
                }
                for side in [-1., 1.] {
                    let mut positions: Vec<_> = seats
                        .iter()
                        .filter(|o| (o.position.x - center) * side > 0.)
                        .map(|o| o.position.x)
                        .collect();
                    positions.sort_by(f32::total_cmp);
                    varied_gap |= positions.windows(2).any(|w| {
                        let n = (w[1] - w[0]) / program.parameters.seat_pitch_metres;
                        (n - n.round()).abs() > 0.07
                    });
                }
            }
            if !p.regular {
                let mean = residuals.iter().sum::<f32>() / residuals.len().max(1) as f32;
                let std = (residuals.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                    / residuals.len().max(1) as f32)
                    .sqrt();
                varied_pose_rooms += usize::from(std > 0.02);
                varied_gap_rooms += usize::from(varied_gap);
            }
            loose += usize::from(!p.regular);
            fixed += usize::from(p.regular);
            for o in &chairs {
                assert!((o.position.y - scene.floor_height(o.position.xz())).abs() < 0.015);
            }
            if seed < 4 {
                let replay = IndoorManifest::generate_with_humans(
                    seed,
                    IndoorLayout::LectureHall,
                    0.65,
                    0,
                    0.25,
                )
                .unwrap();
                assert_eq!(scene, replay);
            }
            rooms.push(serde_json::json!({"seed":seed,"policy":p,"chairs":chairs.iter().map(|o|serde_json::json!({"position":o.position,"size":o.size,"yaw":o.yaw,"variant":o.variant})).collect::<Vec<_>>()}));
        }
        assert!(loose >= 40 && fixed >= 5);
        assert!(
            varied_pose_rooms >= 30,
            "only {varied_pose_rooms} halls retained yaw diversity"
        );
        assert!(
            varied_gap_rooms >= 30,
            "only {varied_gap_rooms} halls retained spacing diversity"
        );
        std::fs::create_dir_all("out/content_diversity_10x").unwrap();
        std::fs::write(
            "out/content_diversity_10x/lecture_seating.json",
            serde_json::to_vec_pretty(&rooms).unwrap(),
        )
        .unwrap();
    }
}
