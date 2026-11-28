use super::*;

#[derive(Clone, Copy)]
struct Side {
    inward: Vec2,
    offset: f32,
    inv_crest_distance: f32,
}

pub(super) struct GroundPoint {
    pub(super) height_m: f32,
    pub(super) clearance_m: f32,
    gradient: Vec2,
}

#[derive(serde::Serialize)]
pub(super) struct Ground {
    pub(super) radii_m: Vec2,
    pub(super) edge_height_m: f32,
    crest_m: Vec3,
    medium: soil::Medium,
    cover: soil::garden::Cover,
    texture_axis: Vec2,
    texture_origin_m: Vec2,
    #[serde(skip)]
    sides: [Side; SIDES],
    #[serde(skip)]
    pub(super) boundary: [Vec2; SIDES],
}

impl Ground {
    pub(super) fn sample(o: &IndoorObject, edging: &Edging, collection: Collection) -> Self {
        let mut rng = stream(o.seed, u64::from_le_bytes(*b"YardFill"));
        let radii_m = o.size.xz() * 0.5 - Vec2::splat(edging.width_m);
        let profile = RevolvedOutline::superellipse(SIDES as u32, edging.exponent, radii_m);
        let boundary = std::array::from_fn(|i| profile.perimeter_point(i as f32 / SIDES as f32));
        let recess = rng
            .random_range(0.034..0.096_f32)
            .min(edging.height_m * 0.40);
        let edge_height_m = edging.height_m - recess;
        let crest =
            Vec2::from_angle(rng.random_range(-PI..PI)) * rng.random_range(0.0..0.32) * radii_m;
        let crest_m = at_height(crest, edge_height_m + rng.random_range(0.08..0.76) * recess);
        let medium = match rng.random_range(0..10) {
            0..=6 if collection == Collection::DryGarden => soil::Medium::MineralGravel,
            0..=3 => soil::Medium::BarkChips,
            4 => soil::Medium::MineralGravel,
            _ => soil::Medium::Loam,
        };
        let sides = std::array::from_fn(|i| {
            let a = boundary[i];
            let inward = (boundary[(i + 1) % SIDES] - a).perp().normalize();
            let offset = inward.dot(a);
            Side {
                inward,
                offset,
                inv_crest_distance: (inward.dot(crest) - offset).recip(),
            }
        });
        Self {
            radii_m,
            edge_height_m,
            crest_m,
            medium,
            boundary,
            sides,
            cover: soil::garden::Cover::sample(o.seed, medium),
            texture_axis: Vec2::from_angle(rng.random_range(0.0..TAU)),
            texture_origin_m: Vec2::new(rng.random_range(0.0..2.), rng.random_range(0.0..2.)),
        }
    }

    /// The limiting edge identifies the triangle of the soil fan below this
    /// root. Its signed distance ratio is the barycentric weight of the crest.
    pub(super) fn at(&self, p: Vec2) -> GroundPoint {
        let mut weight = f32::INFINITY;
        let mut gradient = Vec2::ZERO;
        let mut clearance = f32::INFINITY;
        for side in &self.sides {
            let distance = side.inward.dot(p) - side.offset;
            let w = distance * side.inv_crest_distance;
            if w < weight {
                weight = w;
                gradient = side.inward * side.inv_crest_distance;
            }
            clearance = clearance.min(distance);
        }
        let relief = self.crest_m.y - self.edge_height_m;
        GroundPoint {
            height_m: self.edge_height_m + weight * relief,
            clearance_m: clearance,
            gradient: gradient * relief,
        }
    }

    pub(super) fn mesh(&self, a: &mut Assembly) {
        let g = a.part(Surface::Soil, &self.medium.label(soil::Layer::Bed));
        let first = g.positions.len() as u32;
        let uv = |point| (self.texture_axis.rotate(point) + self.texture_origin_m).to_array();
        g.positions.push(self.crest_m.to_array());
        g.uvs.push(uv(self.crest_m.xz()));
        let mut faces = [Vec3::ZERO; SIDES];
        for (i, normal) in faces.iter_mut().enumerate() {
            let p = at_height(self.boundary[i], self.edge_height_m);
            let q = at_height(self.boundary[(i + 1) % SIDES], self.edge_height_m);
            *normal = (q - self.crest_m).cross(p - self.crest_m);
            g.positions.push(p.to_array());
            g.uvs.push(uv(p.xz()));
            g.indices.extend([
                first,
                first + 1 + ((i + 1) % SIDES) as u32,
                first + 1 + i as u32,
            ]);
        }
        g.normals
            .push(faces.iter().sum::<Vec3>().normalize().to_array());
        for i in 0..SIDES {
            g.normals.push(
                (faces[(i + SIDES - 1) % SIDES] + faces[i])
                    .normalize()
                    .to_array(),
            );
        }
        self.cover.mesh(a, self.medium, self.radii_m, |p| {
            let at = self.at(p);
            (at.height_m, at.gradient)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn courtyard_gravel_and_bark_have_seated_skirts_visible_tops_and_one_batch_per_finish() {
        let mut covered = 0;
        for seed in 0..64 {
            let o = super::super::tests::fixture(seed);
            let edging = Edging::sample(&o);
            let ground = Ground::sample(&o, &edging, Collection::from_bed(o.variant));
            let mut a = Assembly::default();
            ground.mesh(&mut a);
            if ground.cover.count == 0 {
                assert_eq!(a.parts.len(), 1);
                continue;
            }
            covered += 1;
            assert_eq!(a.parts.len(), 2);
            let chips = &a.parts[&(
                Surface::Soil,
                ground.medium.label(soil::Layer::Grain).into_owned(),
            )];
            let sides = if ground.medium == soil::Medium::BarkChips {
                5
            } else {
                8
            };
            let stride = 2 * sides + 1;
            assert_eq!(
                chips.positions.len(),
                usize::from(ground.cover.count) * stride
            );
            for grain in chips.positions.chunks_exact(stride) {
                for (i, p) in grain.iter().enumerate() {
                    let xyz = Vec3::from_array(*p);
                    let at = ground.at(xyz.xz());
                    assert!(at.clearance_m > 0. && xyz.y > 0.);
                    if i < 2 * sides && i % 2 == 0 {
                        assert!(xyz.y < at.height_m);
                    } else {
                        assert!(xyz.y > at.height_m);
                    }
                }
            }
        }
        assert!(covered > 20);
    }
}
