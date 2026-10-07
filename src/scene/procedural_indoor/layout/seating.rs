//! Joint chair/person placement, after furniture, props and static poses exist.
use super::super::{
    footprint::Footprint,
    humans,
    objects::{chairs, tables},
};
use super::*;

pub(crate) fn furniture_overlap(a: &IndoorObject, b: &IndoorObject, margin: f32) -> bool {
    let (lo, hi) = a.bounds();
    let (bl, bh) = b.bounds();
    if lo.y >= bh.y || hi.y <= bl.y || !Footprint::object(a).overlaps(Footprint::object(b), margin)
    {
        return false;
    }
    let pair = if a.kind == ObjectKind::Chair {
        Some((a, b))
    } else if b.kind == ObjectKind::Chair {
        Some((b, a))
    } else {
        None
    };
    if let Some((chair, table)) = pair {
        if chair.interaction_target == Some(table.id)
            && matches!(table.kind, ObjectKind::Table | ObjectKind::Desk)
        {
            let solid = tables::Clearance::new(table);
            let tf =
                table.transform().compute_affine().inverse() * chair.transform().compute_affine();
            return chairs::clearance_boxes(chair).into_iter().any(|(lo, hi)| {
                let mut a = Vec3::splat(f32::INFINITY);
                let mut b = Vec3::splat(f32::NEG_INFINITY);
                for x in [lo.x, hi.x] {
                    for y in [lo.y, hi.y] {
                        for z in [lo.z, hi.z] {
                            let p = tf.transform_point3(Vec3::new(x, y, z));
                            a = a.min(p);
                            b = b.max(p);
                        }
                    }
                }
                solid.hits_box(a, b, margin.max(0.008))
            });
        }
    }
    true
}

impl IndoorManifest {
    pub(super) fn settle_seating(&mut self) {
        let chairs: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor && o.kind == ObjectKind::Chair && o.interaction_target.is_some()
            })
            .map(|o| o.id)
            .collect();
        for id in chairs {
            let chair = self.objects[id].clone();
            let table = self.objects[chair.interaction_target.unwrap()].clone();
            if !matches!(table.kind, ObjectKind::Table | ObjectKind::Desk) {
                continue;
            }
            let mut rng = stream(chair.seed, 905);
            // Retain pulled-out chairs as part of the distribution.
            if rng.random_bool(0.18) {
                continue;
            }
            let p = table
                .transform()
                .compute_affine()
                .inverse()
                .transform_point3(chair.position)
                .xz();
            let outline = tables::outline(&table);
            let nearest = outline
                .iter()
                .enumerate()
                .map(|(i, &a)| {
                    let edge = outline[(i + 1) % outline.len()] - a;
                    a + edge * ((p - a).dot(edge) / edge.length_squared().max(1e-12)).clamp(0., 1.)
                })
                .min_by(|a, b| a.distance_squared(p).total_cmp(&b.distance_squared(p)))
                .unwrap();
            let delta =
                table.transform().rotation * Vec3::new(nearest.x - p.x, 0., nearest.y - p.y);
            let distance = delta.length();
            if distance < 0.01 {
                continue;
            }
            let occupant = self.humans.iter().position(|h| h.chair == Some(id));
            // Occupied workstations aim toward a usable reach; empty chairs also
            // sample partial insertion. Clearance, not the prior, sets the limit.
            let depth = if occupant.is_some() {
                rng.random_range(0.85..1.10)
            } else {
                rng.random_range(0.35..1.05)
            };
            let movement = delta / distance * (distance + 0.15) * depth;
            let desired_yaw = (-delta.x).atan2(-delta.z);
            let turn = (desired_yaw - chair.yaw)
                .sin()
                .atan2((desired_yaw - chair.yaw).cos())
                * rng.random_range(0.45..0.95);
            let screens: Vec<_> = self
                .objects
                .iter()
                .filter(|o| {
                    o.interaction_target == Some(id)
                        && matches!(o.kind, ObjectKind::Laptop | ObjectKind::Monitor)
                })
                .cloned()
                .collect();
            // Bounded backtracking retains a sampled insertion depth, never
            // forces a collision or changes the person's body dimensions to make it fit.
            for step in (1..=24).rev() {
                let mut moved = chair.clone();
                moved.position += movement * (step as f32 / 24.);
                moved.yaw += turn;
                if !self.placement_clear(&moved, 0.014) {
                    continue;
                }
                self.objects[id] = moved.clone();
                for screen in &screens {
                    let delta = moved.position - screen.position;
                    self.objects[screen.id].yaw = delta.x.atan2(delta.z);
                }
                let clear_screens = screens.iter().all(|s| {
                    self.prop_clear(
                        &self.objects[s.id],
                        &self.objects[s.support.unwrap()],
                        0.012,
                    )
                });
                // Empty chairs and rotating screens can also enter a standing
                // person's space or another occupied seat. Check affected people
                // before accepting the whole placement transaction.
                let others_clear = self
                    .humans
                    .iter()
                    .filter(|h| Some(h.id) != occupant.map(|i| self.humans[i].id))
                    .all(|h| {
                        let (lo, hi) = h.bounds();
                        let affected = std::iter::once(&self.objects[id])
                            .chain(screens.iter().map(|s| &self.objects[s.id]))
                            .any(|o| {
                                let (a, b) = o.bounds();
                                lo.cmplt(b).all() && hi.cmpgt(a).all()
                            });
                        !affected || humans::placement_clear(self, h)
                    });
                let moved_person = occupant.and_then(|i| {
                    if !clear_screens || !others_clear {
                        return None;
                    }
                    let proposals = if self.humans[i].pose == humans::HumanPoseKind::SeatedWorking {
                        humans::WORKTOP_POSE_ATTEMPTS
                    } else {
                        1
                    };
                    (0..proposals).find_map(|proposal| {
                        let mut h = self.humans[i].clone();
                        h.position = moved.position;
                        h.yaw = moved.yaw;
                        humans::fit_worktop(&mut h, &table, proposal);
                        humans::placement_clear(self, &h).then_some(h)
                    })
                });
                if clear_screens && others_clear && (occupant.is_none() || moved_person.is_some()) {
                    if let (Some(i), Some(h)) = (occupant, moved_person) {
                        self.humans[i] = h;
                    }
                    break;
                }
                self.objects[id] = chair.clone();
                for screen in &screens {
                    self.objects[screen.id] = screen.clone();
                }
            }
        }
    }
}
