//! Installed TVs face a viewing area selected from the furnishings before any
//! cameras are sampled. A failed mount never forces a TV onto glass or into an aisle.
use super::*;
mod audience;
use crate::scene::procedural_indoor::envelope::polygon;
use audience::Audience;
mod partitions;
mod projection;
use partitions::PartitionFace;
#[cfg(test)]
mod tests;

type Bounds = (Vec3, Vec3);
const COLUMN_BITE: f32 = 0.006; // The narrow fixing passes through a sleeve joint into the shaft.
const WALL_TRIALS: usize = 24;
const COLUMN_TRIALS: usize = 8;

/// Physical backing reconstructed from the archived room and display pose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TelevisionMount {
    PerimeterWall,
    Partition(usize),
    Pillar(usize),
}

struct MountSpace {
    structure: Vec<Bounds>,
    objects: Vec<(usize, Bounds)>,
}

impl MountSpace {
    fn sample(scene: &IndoorManifest) -> Self {
        Self {
            structure: scene.camera_obstacles(),
            objects: scene
                .objects
                .iter()
                .flat_map(|o| o.obstacle_bounds().map(move |bounds| (o.id, bounds)))
                .collect(),
        }
    }

    fn clear(&self, o: &IndoorObject, host: Option<Bounds>) -> bool {
        let (lo, hi) = o.bounds();
        let intersects = |(a, b): Bounds| {
            lo.cmplt(b + Vec3::splat(0.035)).all() && hi.cmpgt(a - Vec3::splat(0.035)).all()
        };
        !self.objects.iter().any(|&(_, bounds)| intersects(bounds))
            && !self
                .structure
                .iter()
                .any(|&b| Some(b) != host && intersects(b))
    }
}

impl IndoorManifest {
    pub(in crate::scene::procedural_indoor::layout) fn display_demand(&self) -> usize {
        use IndoorLayout::*;
        let (chance, repeater) = match self.layout {
            Conference => (0.89, 0.47),
            LectureHall | Training => (0.92, 0.70),
            Classroom => (0.82, 0.38),
            Coworking | OpenOffice => (0.80, 0.42),
            Reception | Showroom | Breakroom => (0.64, 0.24),
            Cafe | Cafeteria | PublicSuite | FitnessRoom | CyclingStudio | Gymnasium => {
                (0.56, 0.36)
            }
            Lounge | Studio | ElevatorLanding => (0.46, 0.20),
            Workshop | Library | Mixed => (0.34, 0.14),
            GroceryStore | ClothingStore | Hallway | StairHall | Courtyard => (0.30, 0.18),
            Gallery | Kitchenette | Kitchen | Restaurant => (0.15, 0.08),
        };
        let mut rng = sampling::Domain::Displays.rng(self.seed);
        if !rng.random_bool(chance) {
            return 0;
        }
        let (lo, hi) = self.primary_room_bounds();
        let area = (hi - lo).element_product();
        let extra = ((area - 30.) / 115.).clamp(0., 1.) * (0.55 + 0.45 * self.density) * repeater;
        if !rng.random_bool(extra.into()) {
            return 1;
        }
        // Larger venues can serve a distant seating section with a third set.
        let distant_section = ((area - 95.) / 140.).clamp(0., 1.) * repeater * 0.55;
        2 + usize::from(rng.random_bool(distant_section.into()))
    }

    /// Identify the physical backing of a mounted TV, including in a deserialized manifest.
    pub fn television_mount(&self, o: &IndoorObject) -> Option<TelevisionMount> {
        if o.kind != ObjectKind::Display || o.neighbor || o.support.is_some() {
            return None;
        }
        if self.perimeter_attachment_clear(o) {
            return Some(TelevisionMount::PerimeterWall);
        }
        if let Some(face) = self.partition_faces().find(|face| face.supports(self, o)) {
            return Some(TelevisionMount::Partition(face.index));
        }
        self.display_pillar(o)
            .filter(|&i| self.display_pillar_clear(o, i))
            .map(TelevisionMount::Pillar)
    }

    fn display_pillar(&self, o: &IndoorObject) -> Option<usize> {
        let e = self.envelope.as_ref()?;
        let n = (o.transform().rotation * Vec3::Z).xz();
        let back = o.position.xz() - n * (o.size.z * 0.5 - COLUMN_BITE);
        let y = o.position.y + o.size.y * 0.5;
        e.pillars.iter().position(|p| {
            back.distance_squared(p.center) <= (p.radius + 0.002).powi(2)
                && (0..4).any(|side| {
                    p.mounting_face(e, self.room_size, side, y)
                        .is_some_and(|(face, outward)| {
                            face.distance_squared(back) < 1e-6 && outward.dot(n) > 0.99999
                        })
                })
        })
    }

    fn display_pillar_clear(&self, o: &IndoorObject, index: usize) -> bool {
        let e = self.envelope.as_ref().unwrap();
        let (lo, hi) = o.bounds();
        let (a, b) = (lo.xz(), hi.xz());
        let region = self.primary_region();
        let headroom = 2.02;
        // A narrow tread, high rug, soffit or mezzanine edge can pass between
        // point samples. Reserve walking clearance over the whole TV footprint.
        region.min.cmple(a).all()
            && region.max.cmpge(b).all()
            && polygon::box_inside(&e.footprint, a, b, 0.02)
            && hi.y + 0.10 < self.minimum_ceiling_height(a, b)
            && !self.covers_cooktop_splashback(o)
            && lo.y >= self.floor_height(e.pillars[index].center) + headroom
            && [a, b, Vec2::new(a.x, b.y), Vec2::new(b.x, a.y)]
                .into_iter()
                .all(|p| lo.y >= self.floor_height(p) + headroom)
            && !e
                .floor_patches
                .iter()
                .any(|p| p.overlaps(a, b) && lo.y < p.height + headroom)
            && !self
                .rug_supports
                .iter()
                .any(|r| a.cmplt(r.max).all() && b.cmpgt(r.min).all() && lo.y < r.top + headroom)
            && !e.mezzanine.as_ref().is_some_and(|m| {
                (a.cmplt(m.stair_max).all() && b.cmpgt(m.stair_min).all())
                    || (m.deck.overlaps(a, b)
                        && hi.y >= m.deck.height - m.thickness
                        && lo.y < m.deck.height + headroom)
            })
    }

    fn display_size(&self, rng: &mut ChaCha8Rng, pillar: bool) -> Vec3 {
        use IndoorLayout::*;
        let signage = matches!(
            self.layout,
            Reception | ElevatorLanding | Cafe | Cafeteria | Showroom | Gallery | PublicSuite
        );
        let (lo, hi): (f32, f32) = if pillar {
            (0.66, 1.43)
        } else if matches!(self.layout, LectureHall | Conference | Training | Classroom) {
            (1.0, 2.25)
        } else {
            (0.70, 1.65)
        };
        let long = rng.random_range(lo.ln()..hi.ln()).exp();
        let aspect = if rng.random_bool(0.81) {
            16. / 9.
        } else {
            rng.random_range(1.50..2.18)
        };
        let (x, y) = if signage && rng.random_bool(0.24) {
            (long / aspect, long)
        } else {
            (long, long / aspect)
        };
        Vec3::new(x, y, rng.random_range(0.075..0.17))
    }

    fn wall_tv(
        &self,
        size: Vec3,
        teaching: bool,
        tf: Transform,
        span: f32,
        rng: &mut ChaCha8Rng,
    ) -> Option<IndoorObject> {
        let extent = (span - size.x) * 0.5 - 0.12;
        if extent <= 0. {
            return None;
        }
        let along = rng.random_range(-extent..extent);

        let center = if teaching {
            rng.random_range(1.62..2.35)
        } else {
            rng.random_range(1.28..2.02)
        };
        let mut o = self.candidate(
            ObjectKind::Display,
            tf.transform_point(Vec3::X * along),
            size,
            tf.rotation.to_euler(EulerRot::YXZ).0,
            rng,
        );
        o.position += tf.rotation
            * Vec3::Z
            * crate::scene::procedural_indoor::objects::wall::mount_offset(&o)?;
        o.position.y = self.floor_height(o.position.xz()) + center - size.y * 0.5;
        o.solid = false;
        (self
            .primary_region()
            .contains_projection(o.position.xz(), 0.)
            && self.attachment_clear(tf, span, &o))
        .then_some(o)
    }

    fn pillar_tv(&self, size: Vec3, rng: &mut ChaCha8Rng) -> Option<(IndoorObject, Bounds)> {
        let e = self.envelope.as_ref()?;
        let index = rng.random_range(0..e.pillars.len());
        let p = &e.pillars[index];
        let bottom = self.floor_height(p.center) + rng.random_range(2.04..2.29);
        let (face, n) = p.mounting_face(
            e,
            self.room_size,
            rng.random_range(0..4),
            bottom + size.y * 0.5,
        )?;
        let pos = face + n * (size.z * 0.5 - COLUMN_BITE);
        let mut o = self.candidate(
            ObjectKind::Display,
            Vec3::new(pos.x, bottom, pos.y),
            size,
            n.x.atan2(n.y),
            rng,
        );
        o.solid = false;
        self.display_pillar_clear(&o, index)
            .then(|| (o, p.obstacle_bounds(e, self.room_size)))
    }

    pub(in crate::scene::procedural_indoor::layout) fn furnish_displays(&mut self) {
        let demand = self.display_demand();
        if demand == 0 {
            return;
        }
        let audience = Audience::sample(self);
        if audience.eyes.is_empty() {
            return;
        }
        let mut space = MountSpace::sample(self);
        let region = self.primary_region();
        let existing: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor
                    && o.kind == ObjectKind::Display
                    && region.contains_projection(o.position.xz(), 0.)
            })
            .collect();
        // A decorative TV in a different room, or behind a teaching audience,
        // does not equip the selected viewing area. Preserve space between sets.
        let equipped = existing
            .iter()
            .filter(|o| audience.score(self, o, &space, None).is_some())
            .count();
        let mut installed: Vec<_> = existing.into_iter().map(|o| o.position.xz()).collect();
        let partitions: Vec<PartitionFace> = self.partition_faces().collect();
        for ordinal in equipped..demand {
            let mut rng = sampling::Domain::Displays.proposal(self.seed, ordinal);
            let wall = self.display_size(&mut rng, false);
            let column = self.display_size(&mut rng, true);
            let prefer_column = rng.random_bool(0.33);
            let mut best: Option<(f32, IndoorObject)> = None;
            let pillars = self.envelope.as_ref().map_or(0, |e| e.pillars.len());
            for trial in 0..(WALL_TRIALS + if pillars == 0 { 0 } else { COLUMN_TRIALS }) {
                let proposal = if trial < WALL_TRIALS {
                    let (tf, span, host) = if !partitions.is_empty() && trial % 3 == 0 {
                        let face = &partitions[rng.random_range(0..partitions.len())];
                        (face.tf, face.span, Some(face.bounds))
                    } else {
                        let (tf, span) = self.attachment_wall(&mut rng);
                        (tf, span, None)
                    };
                    self.wall_tv(wall, audience.teaching_front.is_some(), tf, span, &mut rng)
                        .map(|o| (o, host))
                } else {
                    self.pillar_tv(column, &mut rng).map(|(o, b)| (o, Some(b)))
                };
                let Some((o, host)) = proposal else {
                    continue;
                };
                if installed
                    .iter()
                    .any(|p| p.distance_squared(o.position.xz()) < (2.4 + o.size.x * 0.5).powi(2))
                    || !space.clear(&o, host)
                {
                    continue;
                }
                let Some(sight) = audience.score(self, &o, &space, host) else {
                    continue;
                };
                let score = sight
                    + rng.random_range(0.0..0.55)
                    + if prefer_column && trial >= WALL_TRIALS {
                        1.05
                    } else {
                        0.
                    };
                if best.as_ref().is_none_or(|(s, _)| score > *s) {
                    best = Some((score, o));
                }
            }
            if let Some((_, o)) = best {
                space.objects.push((o.id, o.bounds()));
                installed.push(o.position.xz());
                self.objects.push(o);
            }
        }
    }
}
