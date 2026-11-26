//! The rectangular irradiance texture encloses two rooms. A point in open air
//! above a sloping roof is a valid ray origin, but its sunlit radiance must
//! never be interpolated onto the underside of the roof.
use super::*;
use crate::scene::procedural_indoor::{
    envelope::{polygon, EnvelopeProgram},
    layout::AdjoiningSpace,
};

pub(super) struct ProbeDomain {
    pub(super) filter_irradiance: bool,
    size: Vec3,
    envelope: EnvelopeProgram,
    adjoining: AdjoiningSpace,
}

impl ProbeDomain {
    pub(super) fn from_manifest(scene: &IndoorManifest) -> Option<Box<Self>> {
        if scene.generator_version < 128 {
            return None;
        }
        Some(Box::new(Self {
            filter_irradiance: scene.generator_version >= 289,
            size: scene.room_size,
            envelope: scene.envelope.as_ref()?.clone(),
            adjoining: scene.adjoining_space(),
        }))
    }

    fn vertical_interval(&self, p: Vec2) -> Option<(f32, f32)> {
        let floor = if p.y > self.size.z * 0.5 {
            if !self.adjoining.contains_bounds(self.size, p, p, 0.) {
                return None;
            }
            0.
        } else {
            if !polygon::contains(&self.envelope.footprint, p, 0.) {
                return None;
            }
            self.envelope.floor_height(p)
        };
        Some((floor, self.envelope.ceiling_height(self.size, p)))
    }
}

impl BakeScene {
    pub(super) fn probe_grid(&self, resolution: UVec3) -> (Vec<Vec3>, Vec<bool>) {
        let point = |xyz: UVec3| {
            self.bounds_min
                + (xyz.as_vec3() + Vec3::splat(0.5)) / resolution.as_vec3()
                    * (self.bounds_max - self.bounds_min)
        };
        let count = resolution.element_product() as usize;
        let mut points = Vec::with_capacity(count);
        let mut valid = Vec::with_capacity(count);
        let mut intervals = Vec::with_capacity(resolution.x as usize);
        for z in 0..resolution.z {
            intervals.clear();
            if let Some(domain) = &self.probe_domain {
                intervals.extend(
                    (0..resolution.x)
                        .map(|x| domain.vertical_interval(point(UVec3::new(x, 0, z)).xz())),
                );
            }
            for y in 0..resolution.y {
                for x in 0..resolution.x {
                    let p = point(UVec3::new(x, y, z));
                    let in_room = intervals.is_empty()
                        || intervals[x as usize]
                            .is_some_and(|(floor, roof)| p.y > floor + 0.002 && p.y < roof - 0.002);
                    points.push(p);
                    valid.push(in_room && !self.inside_solid(p));
                }
            }
        }
        (points, valid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::IndoorLayout;

    #[test]
    fn exterior_rooftop_probes_cannot_light_the_cafe_ceiling() {
        let manifest = IndoorManifest::generate(44218, IndoorLayout::Mixed, 0.65, 0).unwrap();
        let mut scene = super::super::tests::empty();
        let size = manifest.room_size;
        scene.bounds_min = Vec3::new(-size.x * 0.5 - 0.1, -0.1, -size.z * 0.5 - 0.1);
        scene.bounds_max = size * Vec3::new(0.5, 1., 0.5)
            + Vec3::splat(0.1)
            + Vec3::Z * manifest.adjoining_space().max_depth();
        let resolution = ((scene.bounds_max - scene.bounds_min) / 0.85)
            .ceil()
            .as_uvec3();
        scene.probe_domain = ProbeDomain::from_manifest(&manifest);
        let (points, valid) = scene.probe_grid(resolution);
        let mut rooftop = 0;
        let mut adjoining = 0;
        for (p, valid) in points.into_iter().zip(valid) {
            if p.y > manifest.ceiling_height(p.xz()) + 0.14 {
                rooftop += 1;
                assert!(!valid);
            }
            if valid {
                assert!(p.y < manifest.ceiling_height(p.xz()));
                if p.z > size.z * 0.5 {
                    assert!(manifest
                        .adjoining_space()
                        .contains_bounds(size, p.xz(), p.xz(), 0.));
                    adjoining += 1;
                } else {
                    assert!(p.y > manifest.floor_height(p.xz()));
                }
            }
        }
        assert!(rooftop > 0 && adjoining > 0);
    }
}
