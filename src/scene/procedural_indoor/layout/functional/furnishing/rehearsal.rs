//! Fit the seat and the instrument or reading desk as one workstation. Layouts
//! range from almost straight ranks to curved ensembles on either side of an aisle.
use super::*;
pub(in super::super) mod percussion;

pub(super) fn across_x(seed: u64, min: Vec2, max: Vec2) -> bool {
    let size = max - min;
    size.x > size.y * stream(seed, u64::from_le_bytes(*b"MuscAxis")).random_range(0.74..1.37)
}

fn oriented(v: Vec2, across_x: bool) -> Vec2 {
    if across_x {
        Vec2::new(v.y, v.x)
    } else {
        v
    }
}

struct Ensemble {
    across_x: bool,
    index: usize,
    structure: PlacementStructure,
    chair_size: Vec3,
    focus: Vec2,
    curvature: f32,
}

impl Ensemble {
    fn pair(
        &self,
        scene: &mut IndoorManifest,
        point: Vec2,
        direction: Vec2,
        kind: ObjectKind,
        r: &mut ChaCha8Rng,
    ) -> bool {
        let point = oriented(point, self.across_x);
        let direction = oriented(direction, self.across_x);
        let yaw = (-direction.x).atan2(-direction.y);
        let pos = |p: Vec2| Vec3::new(p.x, 0., p.y);
        let n = scene.objects.len();
        let chair = FloorProposal {
            kind: ObjectKind::Chair,
            position: pos(point),
            size: self.chair_size * r.random_range(0.97..1.03),
            yaw: yaw + r.random_range(-0.105..0.105),
        };
        let Some(player) = scene.public_fit_at_offsets(
            self.index,
            chair,
            r,
            &self.structure,
            &[Vec2::ZERO],
            |o| {
                o.variant = crate::scene::procedural_indoor::objects::chairs::ENSEMBLE_CHAIR;
                true
            },
        ) else {
            return false;
        };
        let size = geometry::dimensions(kind, r.random());
        let reach = if kind == ObjectKind::StagePiano {
            size.z * 0.44 + r.random_range(0.48..0.56)
        } else {
            r.random_range(0.77..0.99)
        };
        let offset = direction.perp() * r.random_range(-0.09..0.09);
        let fixture = FloorProposal {
            kind,
            size,
            position: pos(point + direction * reach + offset),
            yaw: yaw + r.random_range(-0.08..0.08),
        };
        if scene
            .public_fit_at_offsets(
                self.index,
                fixture,
                r,
                &self.structure,
                &[Vec2::ZERO],
                |o| {
                    o.interaction_target = Some(player);
                    true
                },
            )
            .is_some()
        {
            return true;
        }
        // These two kinds carry no separate service or circulation reservations.
        scene.objects.truncate(n);
        false
    }

    fn musician(&self, scene: &mut IndoorManifest, point: Vec2, r: &mut ChaCha8Rng) -> bool {
        let to_front = self.focus - point;
        let direction = Vec2::new(self.curvature * to_front.x, to_front.y).normalize_or_zero();
        direction.length_squared() > 0.5
            && self.pair(scene, point, direction, ObjectKind::MusicStand, r)
    }
}

impl IndoorManifest {
    pub(super) fn rehearsal_program(&mut self, index: usize, r: &mut ChaCha8Rng) {
        let program = &mut self.assembly_diagnostics.programs[index];
        let across_x = across_x(self.seed, program.min, program.max);
        let (min, max) = (
            oriented(program.min, across_x),
            oriented(program.max, across_x),
        );
        let aisle = &program.reservations[0];
        let banks = [
            (min.x, oriented(aisle.min, across_x).x),
            (oriented(aisle.max, across_x).x, max.x),
        ];
        let depth = max.y - min.y;
        let sign = if r.random_bool(0.5) { 1. } else { -1. };
        let z_at = |d: f32| if sign > 0. { min.y + d } else { max.y - d };
        let focus = Vec2::new(
            (min.x + max.x) * 0.5 + (max.x - min.x) * r.random_range(-0.13..0.13),
            z_at(r.random_range(0.48..1.05)),
        );
        let leader = oriented(focus, across_x);
        program.focus = Vec3::new(leader.x, 1.25, leader.y);
        program.required.push(ObjectKind::MusicStand);
        let ensemble = Ensemble {
            across_x,
            index,
            structure: PlacementStructure::new(),
            chair_size: Vec3::new(
                r.random_range(0.49..0.59),
                r.random_range(0.79..0.96),
                r.random_range(0.48..0.58),
            ),
            focus,
            curvature: r.random_range(0.20..0.94),
        };
        let piano = r.random_bool(0.61);
        let first_bank = usize::from(r.random_bool(0.5));
        if piano {
            for (l, h) in [banks[first_bank], banks[1 - first_bank]] {
                let point = Vec2::new(
                    l + (h - l) * r.random_range(0.41..0.59),
                    z_at(r.random_range(0.56..0.88)),
                );
                let direction = Vec2::new(r.random_range(-0.31..0.31), sign).normalize();
                if ensemble.pair(self, point, direction, ObjectKind::StagePiano, r) {
                    break;
                }
            }
        }
        let first_row = if piano { 2.68 } else { 1.95 } + r.random_range(0.0..0.49);
        let row_pitch = r.random_range(1.65..2.18);
        let seat_pitch = r.random_range(1.08..1.62);
        let rows = (((depth - first_row - 0.45) / row_pitch).floor() as i32 + 1).clamp(1, 5);
        let target =
            (4. + self.density * 9. + (max.x - min.x) * depth / 14. + r.random_range(-3.0..3.0))
                .round()
                .clamp(4., 24.) as usize;
        self.assembly_diagnostics.programs[index].requested_capacity = target + usize::from(piano);
        let mut players = 0;
        // Proceed row by row, alternating the preferred bank; short ensembles still
        // occupy both sides. The rows curve towards the conductor at their ends.
        for row in 0..rows {
            for bank in [
                ((row as usize) + first_bank) % 2,
                ((row as usize) + first_bank + 1) % 2,
            ] {
                let (l, h) = banks[bank];
                let seats = ((h - l - 0.45) / seat_pitch).floor().clamp(1., 5.) as usize;
                let row_phase = r.random_range(-0.16..0.16);
                for i in 0..seats {
                    if players >= target {
                        return;
                    }
                    let x = l
                        + (h - l) * (i as f32 + 0.5 + row_phase + r.random_range(-0.08..0.08))
                            / (seats as f32);
                    let bend = ((x - focus.x) / (max.x - min.x)).powi(2)
                        * ensemble.curvature
                        * depth
                        * 0.63;
                    let p = Vec2::new(
                        x,
                        z_at(
                            first_row + row as f32 * row_pitch - bend + r.random_range(-0.14..0.14),
                        ),
                    );
                    players += usize::from(ensemble.musician(self, p, r));
                }
            }
        }
        if players < 2 {
            for uv in variation::recovery_points(5, r) {
                let p = min + (max - min) * uv;
                if (p.y - focus.y) * sign > 1.1 {
                    players += usize::from(ensemble.musician(self, p, r));
                }
                if players >= 2 {
                    break;
                }
            }
        }
    }
}
