//! Seeded, mipmapped fibre arcs used by the coiled grooms in a room. A smooth
//! local rotation bends fibres through a lock; colour, relief and gloss follow
//! that same path. The finite kernel is periodic across both atlas boundaries.
use super::*;
use rand::Rng;

const SIZE: usize = 128;
const CELLS: i32 = 4;

struct Whorl {
    offset: Vec2,
    turn: Vec2,
    radius_sq: f32,
}

struct Fibres {
    locks: [Whorl; (CELLS * CELLS) as usize],
    broad: field::PreparedPeriodic,
    fine: field::PreparedPeriodic,
    pigment: field::PreparedPeriodic,
    period: f32,
}
impl Fibres {
    fn new(seed: u64) -> Self {
        let mut r = crate::scene::procedural_indoor::layout::stream(seed, 0x434f_494c_4649_4252);
        let locks = std::array::from_fn(|_| {
            let angle = r.random_range(0.95..2.48) * if r.random_bool(0.5) { 1. } else { -1. };
            Whorl {
                offset: Vec2::new(r.random_range(0.24..0.76), r.random_range(0.24..0.76)),
                turn: Vec2::from_angle(angle) - Vec2::X,
                radius_sq: r.random_range(0.54..0.90),
            }
        });
        let mut budget = 4096;
        Self {
            locks,
            broad: field::PreparedPeriodic::new(
                [r.random_range(24..39), 7],
                r.random(),
                &mut budget,
            ),
            fine: field::PreparedPeriodic::new(
                [r.random_range(48..78), 13],
                r.random(),
                &mut budget,
            ),
            pigment: field::PreparedPeriodic::new([8, 7], r.random(), &mut budget),
            period: r.random_range(0.047..0.071),
        }
    }

    fn strand(&self, uv: Vec2) -> (f32, f32) {
        let p = uv * CELLS as f32;
        let base = p.floor().as_ivec2();
        let mut warped = uv;
        for y in base.y - 1..=base.y + 1 {
            for x in base.x - 1..=base.x + 1 {
                let whorl =
                    &self.locks[(x.rem_euclid(CELLS) + CELLS * y.rem_euclid(CELLS)) as usize];
                let d = p - Vec2::new(x as f32, y as f32) - whorl.offset;
                let weight = 1. - d.length_squared() / whorl.radius_sq;
                if weight > 0. {
                    warped += whorl.turn.rotate(d) * (weight * weight / CELLS as f32);
                }
            }
        }
        let broad = self.broad.sample(warped.x, warped.y);
        let fine = self.fine.sample(warped.x + 0.008 * (broad - 0.5), warped.y);
        let pigment = self.pigment.sample(warped.x, warped.y);
        (0.66 * broad + 0.34 * fine, pigment)
    }
}

pub(in super::super) fn material(
    seed: u64,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
) -> Handle<StandardMaterial> {
    let f = Fibres::new(seed);
    let mut samples = Vec::with_capacity(SIZE * SIZE);
    for y in 0..SIZE {
        for x in 0..SIZE {
            samples.push(f.strand(Vec2::new(x as f32, y as f32) / SIZE as f32));
        }
    }
    let mut maps: [Vec<u8>; 3] = std::array::from_fn(|_| Vec::with_capacity(SIZE * SIZE * 4));
    let slope = 0.00016 * SIZE as f32 / (2. * f.period);
    let byte = |n: f32| (255. * n.clamp(0., 1.)).round() as u8;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let (bundle, pigment) = samples[y * SIZE + x];
            let value = |x: usize, y: usize| samples[y * SIZE + x].0;
            let dx = value((x + 1) % SIZE, y) - value((x + SIZE - 1) % SIZE, y);
            let dy = value(x, (y + 1) % SIZE) - value(x, (y + SIZE - 1) % SIZE);
            let normal = Vec3::new(-dx * slope, -dy * slope, 1.).normalize();
            let tone = byte(0.87 + 0.39 * (bundle - 0.5) + 0.24 * (pigment - 0.5));
            maps[0].extend([tone, tone, tone, 255]);
            maps[1].extend([
                byte(normal.x * 0.5 + 0.5),
                byte(normal.y * 0.5 + 0.5),
                byte(normal.z * 0.5 + 0.5),
                255,
            ]);
            maps[2].extend([
                byte(0.88 + 0.12 * bundle),
                byte(0.80 + 0.16 * (1. - bundle) + 0.03 * pigment),
                0,
                255,
            ]);
        }
    }
    let [color, normal, packed] = maps;
    let packed = images.add(mip_image(packed, SIZE as u32, MapType::Data));
    materials.add(StandardMaterial {
        base_color_texture: Some(images.add(mip_image(color, SIZE as u32, MapType::Color))),
        normal_map_texture: Some(images.add(mip_image(normal, SIZE as u32, MapType::Normal))),
        metallic_roughness_texture: Some(packed.clone()),
        occlusion_texture: Some(packed),
        uv_transform: bevy::math::Affine2::from_scale(Vec2::splat(f.period.recip())),
        anisotropy_rotation: std::f32::consts::FRAC_PI_2,
        ..default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fibres_wrap_and_mips_remain_dielectric_and_normalized() {
        let mut stock = std::collections::BTreeSet::new();
        for seed in 10..18 {
            let f = Fibres::new(seed);
            let mut lo = 1_f32;
            let mut hi = 0_f32;
            for i in 0..173 {
                let uv = Vec2::new(i as f32 / 127. - 0.27, i as f32 / 91. - 0.72);
                let (v, p) = f.strand(uv);
                lo = lo.min(v);
                hi = hi.max(v);
                for shift in [Vec2::X, Vec2::Y, Vec2::new(-1., 2.)] {
                    let (w, q) = f.strand(uv + shift);
                    assert!((v - w).abs() < 0.0001 && (p - q).abs() < 0.0001);
                }
            }
            assert!(hi - lo > 0.4);
            stock.insert(f.strand(Vec2::new(0.21, 0.48)).0.to_bits());
        }
        assert_eq!(stock.len(), 8);
        let mut images = Assets::<Image>::default();
        let mut materials = Assets::<StandardMaterial>::default();
        let handle = material(81, &mut images, &mut materials);
        let m = materials.get(&handle).unwrap();
        for handle in [
            &m.base_color_texture,
            &m.normal_map_texture,
            &m.metallic_roughness_texture,
        ] {
            let im = images.get(handle.as_ref().unwrap()).unwrap();
            assert_eq!(im.texture_descriptor.mip_level_count, 8);
        }
        let orm = images.get(m.occlusion_texture.as_ref().unwrap()).unwrap();
        for p in orm.data.as_ref().unwrap().as_chunks::<4>().0 {
            assert!(p[0] > 210 && p[1] > 195 && p[2] == 0 && p[3] == 255);
        }
        let normal = images.get(m.normal_map_texture.as_ref().unwrap()).unwrap();
        for p in normal.data.as_ref().unwrap().as_chunks::<4>().0 {
            let n = Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) / 127.5 - Vec3::ONE;
            assert!((n.length() - 1.).abs() < 0.012);
        }
    }
}
