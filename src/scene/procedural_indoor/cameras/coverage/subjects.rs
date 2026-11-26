use super::Coverage;
use crate::scene::procedural_indoor::layout::{IndoorManifest, IndoorObject};
use rand::Rng;

impl Coverage {
    /// Index zero is the unrestricted room. All lists retain manifest order,
    /// including objects that belong to more than one zone. Build on the first
    /// object-directed proposal; person framing and ceiling-only rigs skip it.
    fn subject_ids(&self, scene: &IndoorManifest, zone: Option<usize>) -> &[usize] {
        let lists = self.subjects.get_or_init(|| {
            let zones = scene.program.as_ref().map_or(&[][..], |p| &p.zones[..]);
            let mut lists = vec![Vec::new(); zones.len() + 1];
            for (i, o) in scene
                .objects
                .iter()
                .enumerate()
                .filter(|(_, o)| o.solid && !o.neighbor)
            {
                lists[0].push(i);
                for (j, z) in zones.iter().enumerate() {
                    if o.position.x > z.min.x
                        && o.position.x < z.max.x
                        && o.position.z > z.min.y
                        && o.position.z < z.max.y
                    {
                        lists[j + 1].push(i);
                    }
                }
            }
            lists
        });
        &lists[zone.map_or(0, |index| index + 1)]
    }

    pub(in super::super) fn sample_subject<'a>(
        &self,
        scene: &'a IndoorManifest,
        zone: Option<usize>,
        rng: &mut impl Rng,
    ) -> Option<&'a IndoorObject> {
        let ids = self.subject_ids(scene, zone);
        if ids.is_empty() || !rng.random_bool(0.72) {
            return None;
        }
        Some(&scene.objects[ids[rng.random_range(0..ids.len())]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::{stream, IndoorLayout};
    use bevy::prelude::*;
    #[test]
    fn camera_subject_cache_preserves_zone_membership_order_and_rng_replay() {
        let mut scene =
            IndoorManifest::generate_with_humans(18, IndoorLayout::PublicSuite, 0.7, 0, 0.)
                .unwrap();
        let zones = &scene.program.as_ref().unwrap().zones;
        // Cover boundary exclusion and an unoccupied zone as well as the two
        // real zones and unrestricted proposals.
        let mut empty = zones[0].clone();
        empty.min = Vec2::splat(10000.);
        empty.max = empty.min + Vec2::ONE;
        scene.program.as_mut().unwrap().zones.push(empty);
        let coverage = Coverage::new(&scene);
        assert!(coverage.subjects.get().is_none());
        let mut cached = stream(2222, 7);
        let mut scalar = cached.clone();
        let zones = &scene.program.as_ref().unwrap().zones;
        for zone in std::iter::once(None).chain((0..zones.len()).map(Some)) {
            let reference: Vec<_> = scene
                .objects
                .iter()
                .enumerate()
                .filter(|(_, o)| {
                    o.solid
                        && !o.neighbor
                        && zone.is_none_or(|index| {
                            let p = o.position.xz();
                            let z = &zones[index];
                            p.cmpgt(z.min).all() && p.cmplt(z.max).all()
                        })
                })
                .map(|(i, _)| i)
                .collect();
            assert_eq!(coverage.subject_ids(&scene, zone), reference);
            for _ in 0..64 {
                let expected = (!reference.is_empty() && scalar.random_bool(0.72))
                    .then(|| scene.objects[reference[scalar.random_range(0..reference.len())]].id);
                assert_eq!(
                    coverage
                        .sample_subject(&scene, zone, &mut cached)
                        .map(|o| o.id),
                    expected
                );
            }
        }
        assert_eq!(cached.random::<u64>(), scalar.random::<u64>());
    }
}
