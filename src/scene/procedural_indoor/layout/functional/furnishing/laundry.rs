//! Vended laundries use level runs of plumbed machines and cross-aisles. Stock
//! and flange alignment are shared within a bank. Loading and laundry vary per unit.
use super::*;
use rand::seq::SliceRandom;

#[derive(Clone, Copy)]
struct Wall {
    axis: usize,
    sign: f32,
}

/// A stock run fixes cabinet pitch, product and service depth before individual
/// stations are admitted. Its remaining sites can be filled after the anchors.
struct BankSite {
    proposal: FloorProposal,
    stock: u64,
}

fn sites(
    p: &ProgramAssembly,
    kind: ObjectKind,
    requested: usize,
    walls: &[Wall],
    r: &mut ChaCha8Rng,
) -> Vec<BankSite> {
    let (lo, hi) = (p.min, p.max);
    let mut sites = Vec::new();
    for wall in walls {
        let stock = r.random();
        let size = geometry::laundry::dimensions(kind, stock);
        let axis = wall.axis;
        let along = 1 - axis;
        let sign = wall.sign;
        let yaw = if axis == 0 {
            sign * FRAC_PI_2
        } else if sign > 0. {
            0.
        } else {
            PI
        };
        let gap = if kind == ObjectKind::LaundryFoldingTable {
            r.random_range(0.16..0.44)
        } else {
            r.random_range(0.065..0.142)
        };
        let rear = r.random_range(0.065..0.36);
        let perpendicular = if sign > 0. {
            lo[axis] + size.z * 0.5 + rear
        } else {
            hi[axis] - size.z * 0.5 - rear
        };
        let cross = &p.reservations[along];
        let mut segments = [(lo[along], cross.min[along]), (cross.max[along], hi[along])];
        if r.random() {
            segments.reverse();
        }
        for (a, b) in segments {
            let width = b - a - 0.14;
            let count = (((width + gap) / (size.x + gap)).floor().max(0.) as usize).min(requested);
            if count == 0 {
                continue;
            }
            let span = count as f32 * (size.x + gap) - gap;
            let start = a + 0.07 + (width - span).max(0.) * r.random_range(0.08..0.92);
            for i in 0..count {
                let mut q = Vec2::ZERO;
                q[axis] = perpendicular;
                q[along] = start + size.x * 0.5 + i as f32 * (size.x + gap);
                sites.push(BankSite {
                    proposal: FloorProposal {
                        kind,
                        position: Vec3::new(q.x, 0., q.y),
                        size,
                        yaw,
                    },
                    stock,
                });
            }
        }
    }
    sites
}

impl IndoorManifest {
    pub(super) fn laundry_program(&mut self, index: usize, r: &mut ChaCha8Rng) {
        let mut walls = [
            Wall { axis: 0, sign: 1. },
            Wall { axis: 0, sign: -1. },
            Wall { axis: 1, sign: 1. },
            Wall { axis: 1, sign: -1. },
        ];
        walls.shuffle(r);
        let mut kinds = [ObjectKind::WashingMachine, ObjectKind::TumbleDryer];
        kinds.shuffle(r);
        let structure = PlacementStructure::new();
        let p = &mut self.assembly_diagnostics.programs[index];
        p.required.extend(kinds);
        let area = (p.max - p.min).element_product();
        let mut banks = kinds.map(|kind| {
            let target = (area * r.random_range(0.079..0.125) * (0.64 + 0.65 * self.density))
                .round()
                .clamp(2., 9.) as usize;
            p.requested_capacity += target;
            let candidates = sites(p, kind, target, &walls, r).into_iter();
            walls.rotate_left(1);
            (target, candidates)
        });
        // Establish both washing and drying before optional furniture takes
        // a loading front, then fill the original runs at the same stock pitch.
        for (target, candidates) in &mut banks {
            *target -= self.install_laundry_run(index, candidates, 1, &structure, r);
        }
        if r.random_bool(0.90) {
            let count = 1 + usize::from(area > 53. && r.random_bool(0.55));
            let p = &self.functional_programs()[index];
            let mut stations =
                sites(p, ObjectKind::LaundryFoldingTable, count, &walls, r).into_iter();
            self.install_laundry_run(index, &mut stations, count, &structure, r);
        }
        for (target, mut candidates) in banks {
            self.install_laundry_run(index, &mut candidates, target, &structure, r);
        }
    }

    fn install_laundry_run(
        &mut self,
        index: usize,
        candidates: &mut impl Iterator<Item = BankSite>,
        count: usize,
        structure: &PlacementStructure,
        r: &mut ChaCha8Rng,
    ) -> usize {
        let mut installed = 0;
        for BankSite { proposal, stock } in candidates {
            // Other sites on the service runs provide bounded recovery.
            // Moving an individual plumbed cabinet sideways breaks the bank.
            let Some(id) =
                self.public_fit_at_offsets(index, proposal, r, structure, &[Vec2::ZERO], |o| {
                    o.finish_seed = Some(stock);
                    true
                })
            else {
                continue;
            };
            if self.objects[id].kind == ObjectKind::LaundryFoldingTable {
                let table = self.objects[id].clone();
                for side in [-1., 1.] {
                    if r.random_bool(0.53) {
                        let size = Vec3::new(
                            r.random_range(0.245..0.36),
                            r.random_range(0.022..0.073),
                            r.random_range(0.275..0.41),
                        );
                        let offset = Vec3::new(
                            table.size.x * side * r.random_range(0.17..0.27),
                            0.,
                            table.size.z * r.random_range(-0.14..0.14),
                        );
                        self.program_prop(
                            &table,
                            ObjectKind::KitchenTowel,
                            offset,
                            size,
                            r.random_range(-0.38..0.38),
                            r,
                        );
                    }
                }
            }
            installed += 1;
            if installed == count {
                break;
            }
        }
        installed
    }
}

pub(in super::super) fn front_is_served(p: &ProgramAssembly, o: &IndoorObject) -> bool {
    let Some(depth) = super::frontage::access_depth(o.kind) else {
        return false;
    };
    let tf = o.transform();
    let along = usize::from(o.yaw.sin().abs() > o.yaw.cos().abs());
    let Some(spine) = p.reservations.get(along) else {
        return false;
    };
    // Both near and far corners of the loading space must be served by a
    // single working aisle that reaches the zone's corresponding cross-aisle.
    p.reservations.iter().any(|r| {
        r.role == ReservationRole::EquipmentAccess
            && r.min[along] <= spine.max[along] + 0.001
            && r.max[along] >= spine.min[along] - 0.001
            && [-1., 1.].into_iter().all(|side| {
                [0.003, depth - 0.003].into_iter().all(|d| {
                    let q = tf
                        .transform_point(Vec3::new(
                            side * (o.size.x * 0.5 - 0.003),
                            0.,
                            o.size.z * 0.5 + d,
                        ))
                        .xz();
                    q.cmpge(r.min).all() && q.cmple(r.max).all()
                })
            })
    })
}
