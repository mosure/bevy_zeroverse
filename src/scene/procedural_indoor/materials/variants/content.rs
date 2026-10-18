//! Owned content synthesis jobs. Only immutable images cross worker boundaries;
//! asset handles are allocated in the original surface/slot order on binding.
use super::*;

enum Prepared {
    Screen(Box<screens::PreparedScreen>),
    Board(Box<boards::PreparedBoard>),
    Art(u64, Box<[Image; 3]>),
}

impl Prepared {
    fn new(surface: Surface, seed: u64) -> Self {
        match surface {
            Surface::Screen | Surface::PhoneScreen | Surface::Television => {
                let kind = match surface {
                    Surface::Screen => screens::DisplayKind::Desktop,
                    Surface::PhoneScreen => screens::DisplayKind::Phone,
                    _ => screens::DisplayKind::Television,
                };
                Self::Screen(Box::new(screens::prepare(seed, kind)))
            }
            Surface::Whiteboard => Self::Board(Box::new(boards::prepare(seed))),
            Surface::Art => Self::Art(seed, Box::new(art::maps(seed))),
            _ => unreachable!("content jobs only synthesize supported surfaces"),
        }
    }

    fn apply(self, mat: &mut StandardMaterial, images: &mut impl AssetStore<Image>) {
        match self {
            Self::Screen(prepared) => prepared.apply(mat, images),
            Self::Board(prepared) => prepared.apply(mat, images),
            Self::Art(seed, maps) => art::apply_prepared(seed, *maps, mat, images),
        }
    }
}

pub(super) struct Job {
    #[cfg(not(target_arch = "wasm32"))]
    prepared: Prepared,
    #[cfg(target_arch = "wasm32")]
    input: (Surface, u64),
}

impl Job {
    pub(super) async fn apply(
        self,
        mat: &mut StandardMaterial,
        images: &mut impl AssetStore<Image>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        let prepared = self.prepared;
        #[cfg(target_arch = "wasm32")]
        let prepared = Prepared::new(self.input.0, self.input.1);
        prepared.apply(mat, images);
    }
}

pub(super) fn queue(
    seed: u64,
    used: Option<&std::collections::BTreeSet<(Surface, usize)>>,
) -> BTreeMap<(Surface, usize), Job> {
    let inputs: Vec<_> = [
        Surface::Screen,
        Surface::PhoneScreen,
        Surface::Television,
        Surface::Whiteboard,
        Surface::Art,
    ]
    .into_iter()
    .flat_map(|surface| {
        (0..COUNT)
            .filter(move |&slot| used.is_none_or(|keys| keys.contains(&(surface, slot))))
            .map(move |slot| ((surface, slot), screen_seed(seed, slot)))
    })
    .collect();
    #[cfg(not(target_arch = "wasm32"))]
    {
        // A scoped join also drives queued work when the synchronous material
        // API runs inside the pool, including a one-worker budget. Blocking on
        // individually spawned Tasks here would deadlock that supported case.
        crate::scene::procedural_indoor::preparation::workers::pool()
            .scope(|scope| {
                for (key, seed) in inputs {
                    scope.spawn(async move {
                        (
                            key,
                            Job {
                                prepared: Prepared::new(key.0, seed),
                            },
                        )
                    });
                }
            })
            .into_iter()
            .collect()
    }
    #[cfg(target_arch = "wasm32")]
    inputs
        .into_iter()
        .map(|(key, seed)| {
            (
                key,
                Job {
                    input: (key.0, seed),
                },
            )
        })
        .collect()
}

#[cfg(test)]
pub(in super::super) fn assert_materials_equal(
    a: &StandardMaterial,
    images_a: &Assets<Image>,
    b: &StandardMaterial,
    images_b: &Assets<Image>,
) {
    assert_eq!(images_a.len(), images_b.len());
    let mut a = a.clone();
    let mut b = b.clone();
    let maps = [
        (a.base_color_texture.take(), b.base_color_texture.take()),
        (a.normal_map_texture.take(), b.normal_map_texture.take()),
        (
            a.metallic_roughness_texture.take(),
            b.metallic_roughness_texture.take(),
        ),
        (a.occlusion_texture.take(), b.occlusion_texture.take()),
        (a.emissive_texture.take(), b.emissive_texture.take()),
    ];
    for (ah, bh) in &maps {
        assert_eq!(ah.is_some(), bh.is_some());
        if let (Some(ah), Some(bh)) = (ah, bh) {
            let ai = images_a.get(ah).unwrap();
            let bi = images_b.get(bh).unwrap();
            assert_eq!(ai.texture_descriptor, bi.texture_descriptor);
            assert_eq!(ai.data, bi.data);
            assert_eq!(ai.asset_usage, bi.asset_usage);
            assert_eq!(format!("{:?}", ai.sampler), format!("{:?}", bi.sampler));
        }
        for (other_a, other_b) in &maps {
            assert_eq!(ah == other_a, bh == other_b, "texture sharing changed");
        }
    }
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queued_content_preserves_selection_seed_binding_and_all_pbr_values() {
        for seed in [202, 600, u64::MAX] {
            let surfaces = [
                Surface::Screen,
                Surface::PhoneScreen,
                Surface::Television,
                Surface::Whiteboard,
                Surface::Art,
            ];
            let used = surfaces
                .into_iter()
                .enumerate()
                .map(|(i, s)| (s, i))
                .collect();
            let mut jobs = queue(seed, Some(&used));
            assert_eq!(jobs.len(), surfaces.len());
            for (surface, slot) in used {
                let content_seed = screen_seed(seed, slot);
                let mut a = StandardMaterial {
                    base_color: Color::srgb(0.2, 0.4, 0.6),
                    metallic: 0.7,
                    diffuse_transmission: 0.16,
                    ..default()
                };
                let mut b = a.clone();
                let mut ia = Assets::<Image>::default();
                let mut ib = Assets::<Image>::default();
                match surface {
                    Surface::Screen => screens::apply(content_seed, false, &mut a, &mut ia),
                    Surface::PhoneScreen => screens::apply(content_seed, true, &mut a, &mut ia),
                    Surface::Television => screens::apply_tv(content_seed, &mut a, &mut ia),
                    Surface::Whiteboard => boards::apply(content_seed, &mut a, &mut ia),
                    Surface::Art => art::apply(content_seed, &mut a, &mut ia),
                    _ => unreachable!(),
                }
                bevy::tasks::block_on(
                    jobs.remove(&(surface, slot))
                        .unwrap()
                        .apply(&mut b, &mut ib),
                );
                assert_materials_equal(&a, &ia, &b, &ib);
            }
            assert!(jobs.is_empty());
            assert!(queue(seed, Some(&Default::default())).is_empty());
        }
    }
}
