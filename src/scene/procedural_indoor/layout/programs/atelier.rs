//! Private painting stations and groups working from a physical still life.
use super::*;
use crate::scene::procedural_indoor::objects::public::{
    self,
    easel::{Canvas, Easel},
};

pub(super) const INTRODUCED: u32 = public::easel::VERSION;

impl IndoorManifest {
    pub(super) fn furnish_artist_studio(&mut self) {
        let mut r = stream(self.seed, u64::from_le_bytes(*b"Atelier "));
        let (lo, hi) = self.primary_room_bounds();
        let extent = hi - lo;
        let area = extent.element_product();
        if area < 10. || r.random_bool(0.25) {
            return;
        }
        let capacity = ((area / r.random_range(6.8..11.5)) * (0.58 + 0.5 * self.density))
            .floor()
            .clamp(1., 4.) as usize;
        let count = r.random_range(1..=capacity);
        let kit = r.random();
        let structure = PlacementStructure::new();
        let start = self.objects.len();
        let subject = if count >= 2 && r.random_bool(0.60) {
            self.place_studio_subject(lo, hi, &mut r, &structure)
        } else {
            None
        };
        let daylight = self
            .exterior
            .iter()
            .flat_map(|e| &e.facades)
            .flat_map(|f| {
                let tf = f.side.transform(self.room_size);
                f.openings.iter().map(move |opening| {
                    tf.transform_point(Vec3::X * ((opening.min.x + opening.max.x) * 0.5))
                        .xz()
                })
            })
            .filter(|p| p.distance_squared(p.clamp(lo, hi)) < 0.30)
            .collect::<Vec<_>>();
        let side = if r.random_bool(0.5) { -1. } else { 1. };
        let room_yaw = r.random_range(-std::f32::consts::PI..std::f32::consts::PI);
        let match_kit = r.random_bool(0.57);
        let mut made = 0;
        for _ in 0..count {
            let size = public::dimensions(ObjectKind::ArtistEasel, r.random());
            let mut stand =
                self.floor_candidate(ObjectKind::ArtistEasel, Vec3::ZERO, size, 0., &mut r);
            if match_kit && r.random_bool(0.8) {
                stand.finish_seed = Some(kit);
            }
            for trial in 0..20 {
                let site =
                    lo + extent * Vec2::new(r.random_range(0.06..0.94), r.random_range(0.06..0.94));
                let use_subject = subject.filter(|_| trial < 12);
                let yaw = if let Some(id) = use_subject {
                    let away = site - self.objects[id].position.xz();
                    if !(1.25..5.2).contains(&away.length()) {
                        continue;
                    }
                    let side = if r.random_bool(0.5) { 1. } else { -1. };
                    away.x.atan2(away.y) + side * r.random_range(0.55..1.16)
                } else if !daylight.is_empty() && r.random_bool(0.65) {
                    let window = daylight
                        .iter()
                        .min_by(|a, b| {
                            a.distance_squared(site)
                                .total_cmp(&b.distance_squared(site))
                        })
                        .unwrap();
                    let lateral = *window - site;
                    lateral.x.atan2(lateral.y) + side * r.random_range(0.86..2.04)
                } else {
                    room_yaw + r.random_range(-1.12..1.12)
                };
                stand.yaw = yaw;
                stand.position = Vec3::new(site.x, self.floor_height(site), site.y);
                let (min, max) = super::access::envelope(&stand).unwrap().bounds();
                if min.xz().cmplt(lo + 0.04).any()
                    || max.xz().cmpgt(hi - 0.04).any()
                    || !self.placement_clear_with_structure(&stand, 0.045, false, &structure)
                {
                    continue;
                }
                if let Some(id) = use_subject {
                    if !self.studio_subject_visible(
                        &stand,
                        &self.objects[id],
                        structure.get_or_init(|| self.columns()),
                    ) {
                        continue;
                    }
                    stand.interaction_target = Some(id);
                }
                self.objects.push(stand);
                made += 1;
                break;
            }
        }
        if made == 0 {
            self.objects.truncate(start);
        }
    }

    fn place_studio_subject(
        &mut self,
        lo: Vec2,
        hi: Vec2,
        r: &mut ChaCha8Rng,
        structure: &PlacementStructure,
    ) -> Option<usize> {
        let width = r.random_range(0.53..0.78);
        let size = Vec3::new(
            width,
            r.random_range(0.53..0.74),
            width * r.random_range(0.87..1.16),
        );
        for _ in 0..8 {
            let site =
                lo + (hi - lo) * Vec2::new(r.random_range(0.28..0.72), r.random_range(0.28..0.72));
            let yaw = r.random_range(-std::f32::consts::PI..std::f32::consts::PI);
            let table = self.floor_candidate(
                ObjectKind::SideTable,
                Vec3::new(site.x, 0., site.y),
                size,
                yaw,
                r,
            );
            if !self.placement_clear_with_structure(&table, 0.06, false, structure) {
                continue;
            }
            let vase = r.random_bool(0.65);
            let diameter = if vase {
                r.random_range(0.115..0.205)
            } else {
                r.random_range(0.18..0.262)
            };
            let height = if vase {
                diameter * r.random_range(1.2..2.15)
            } else {
                diameter * r.random_range(0.34..0.60)
            };
            let location = table.transform().transform_point(Vec3::new(
                r.random_range(-0.025..0.025),
                size.y,
                r.random_range(-0.025..0.025),
            ));
            let rotation = r.random_range(-std::f32::consts::PI..std::f32::consts::PI);
            self.objects.push(table.clone());
            let subject = self.candidate(
                if vase {
                    ObjectKind::Vase
                } else {
                    ObjectKind::Bowl
                },
                location,
                Vec3::new(diameter, height, diameter),
                rotation,
                r,
            );
            let id = subject.id;
            if self.admit_supported_prop(&table, subject) {
                return Some(id);
            }
            self.objects.pop();
        }
        None
    }

    pub(in crate::scene::procedural_indoor) fn valid_easel_subject(
        &self,
        easel: &IndoorObject,
        subject: &IndoorObject,
    ) -> bool {
        matches!(subject.kind, ObjectKind::Vase | ObjectKind::Bowl)
            && subject.id < easel.id
            && subject.neighbor == easel.neighbor
            && subject
                .support
                .and_then(|i| self.objects.get(i))
                .is_some_and(|t| t.kind == ObjectKind::SideTable && t.neighbor == easel.neighbor)
    }

    fn studio_subject_visible(
        &self,
        easel: &IndoorObject,
        subject: &IndoorObject,
        structure: &[(Vec3, Vec3)],
    ) -> bool {
        let target = subject.position + Vec3::Y * (subject.size.y * 0.70);
        let tf = easel.transform();
        let model = Easel::sample(easel);
        let panel = tf * model.rail_frame(easel);
        let local = |p| panel.rotation.inverse() * (p - panel.translation);
        let other = local(target);
        [-0.23, 0.23].into_iter().any(|side| {
            let eye = tf.transform_point(Vec3::new(side, 1.52, easel.size.z * 0.5 + 0.38));
            if model.canvas != Canvas::Empty
                && segment_hits_box(
                    local(eye),
                    other,
                    Vec3::new(
                        -model.canvas_width_m * 0.5,
                        model.tray_position_m,
                        model.canvas_back_z(),
                    ),
                    Vec3::new(
                        model.canvas_width_m * 0.5,
                        model.tray_position_m + model.canvas_height_m,
                        model.canvas_back_z() + model.canvas_depth_m,
                    ),
                )
            {
                return false;
            }
            if self.envelope.as_ref().is_some_and(|e| {
                !crate::scene::procedural_indoor::envelope::polygon::segment_inside(
                    &e.footprint,
                    eye.xz(),
                    target.xz(),
                    0.,
                )
            }) || structure
                .iter()
                .any(|(lo, hi)| segment_hits_box(eye, target, *lo, *hi))
            {
                return false;
            }
            !self.objects.iter().any(|o| {
                if o.id == easel.id
                    || o.id == subject.id
                    || Some(o.id) == subject.support
                    || o.neighbor
                    || (!o.solid && o.size.y < 0.15)
                {
                    return false;
                }
                let inv = o.transform().rotation.inverse();
                let half = o.size * 0.5;
                segment_hits_box(
                    inv * (eye - o.position),
                    inv * (target - o.position),
                    Vec3::new(-half.x, 0., -half.z),
                    Vec3::new(half.x, o.size.y, half.z),
                )
            })
        })
    }

    /// Furnishing can add a tall bookcase between artist and subject. Retain
    /// the furniture and record the relationship only at stations with a view.
    pub(in crate::scene::procedural_indoor::layout) fn settle_studio_sightlines(&mut self) {
        let structure = PlacementStructure::new();
        for i in 0..self.objects.len() {
            let o = &self.objects[i];
            if o.kind != ObjectKind::ArtistEasel {
                continue;
            }
            let Some(target) = o.interaction_target else {
                continue;
            };
            if !self.studio_subject_visible(
                o,
                &self.objects[target],
                structure.get_or_init(|| self.columns()),
            ) {
                self.objects[i].interaction_target = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::validation::validate_layout;

    #[test]
    fn studio_stations_preserve_standing_space_and_visible_supported_subjects() {
        let mut independent = 0;
        let mut subjects = 0;
        let mut multiple = 0;
        let mut no_easel = 0;
        let mut verified_occlusion = false;
        for seed in 0..96 {
            let mut scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Studio, 0.65, 0, 0.25)
                    .unwrap();
            validate_layout(&scene).unwrap();
            if seed < 3 {
                assert_eq!(
                    scene,
                    IndoorManifest::generate_with_humans(seed, IndoorLayout::Studio, 0.65, 0, 0.25)
                        .unwrap()
                );
            }
            let stations: Vec<_> = scene
                .objects
                .iter()
                .filter(|o| o.kind == ObjectKind::ArtistEasel)
                .map(|o| o.id)
                .collect();
            multiple += usize::from(stations.len() > 1);
            no_easel += usize::from(stations.is_empty());
            let mut occluder = None;
            for id in stations {
                let station = &scene.objects[id];
                assert!(
                    scene.placement_clear(station, 0.),
                    "seed {seed}, easel {id}"
                );
                assert!(!scene.objects.iter().any(|o| o.id != id
                    && !o.neighbor
                    && o.solid
                    && o.support.is_none()
                    && super::super::access::blocks(station, o)));
                let Some(index) = station.interaction_target else {
                    independent += 1;
                    continue;
                };
                let target = &scene.objects[index];
                assert!(scene.valid_easel_subject(station, target));
                assert!(scene.studio_subject_visible(station, target, &scene.columns()));
                assert!(scene.prop_clear(target, &scene.objects[target.support.unwrap()], 0.));
                subjects += 1;
                if !verified_occlusion {
                    let mut wall = station.clone();
                    wall.kind = ObjectKind::RoomDivider;
                    wall.id = scene.objects.len();
                    let eye = station.transform().transform_point(Vec3::new(
                        0.,
                        1.52,
                        station.size.z * 0.5 + 0.38,
                    ));
                    wall.position = eye
                        .lerp(target.position, 0.55)
                        .with_y(target.position.y - 0.35);
                    wall.size = Vec3::new(1.4, 2., 1.4);
                    wall.interaction_target = None;
                    occluder = Some((id, wall));
                }
            }
            if let Some((id, wall)) = occluder {
                scene.objects.push(wall);
                scene.settle_studio_sightlines();
                assert_eq!(scene.objects[id].interaction_target, None);
                verified_occlusion = true;
            }
        }
        assert!(independent>20 && subjects>3 && multiple>5 && no_easel>8 && verified_occlusion,
            "independent {independent}, still life {subjects}, groups {multiple}, without {no_easel}");
    }
}
