use super::super::super::{program, Surface};
use super::*;

fn carpet(seed: u64) -> program::MaterialRecipe {
    program::sample_with_floor(seed, 1)
        .unwrap()
        .remove(Surface::Floor as usize)
}

#[test]
fn production_sampling_covers_continuous_carpet_and_buildable_installations() {
    let mut counts = [0; 5];
    let mut phase_bins = [0; 8];
    for seed in 0..1024 {
        let r = carpet(seed);
        r.validate().unwrap();
        let f = r.flooring.unwrap();
        if let Some(tiles) = f.carpet_tiles {
            assert!(f.tile_definition > 0.);
            counts[tiles.direction.gpu_code() as usize] += 1;
            phase_bins[(tiles.registration * 8.) as usize] += 1;
        } else {
            assert_eq!(f.tile_definition, 0.);
            counts[0] += 1;
        }
    }
    assert!(counts.into_iter().all(|n| n > 35), "{counts:?}");
    assert!(phase_bins.into_iter().all(|n| n > 20), "{phase_bins:?}");
}

#[test]
fn tile_frames_preserve_yarn_scale_periodicity_and_installation_direction() {
    use PileDirection::*;
    let r = carpet(21);
    let mut f = r.flooring.unwrap();
    f.layout = FloorLayout::Straight;
    f.tile_definition = 0.6;
    for cells in [[4, 4], [4, 7]] {
        f.cells = cells;
        for direction in [Parallel, OpposingCourses, QuarterTurn, NonDirectional] {
            let tiles = CarpetTiles {
                direction,
                registration: 0.83,
            };
            if cells[0] != cells[1] && direction == QuarterTurn {
                assert!(tiles.validate(&f).is_err());
                continue;
            }
            tiles.validate(&f).unwrap();
            let mut origins = std::collections::HashSet::new();
            for x in 0..cells[0] {
                for y in 0..cells[1] {
                    let yarn = |uv| tiles.yarn_uv(uv, [x, y], &f, u64::MAX - 5);
                    let uv = [0.125, 0.75];
                    let p = yarn(uv);
                    assert_eq!(p, yarn([uv[0] - 1., uv[1] + 2.]));
                    origins.insert(p.map(f32::to_bits));
                    let step = 1. / 1024.;
                    let delta = |q: [f32; 2]| {
                        std::array::from_fn::<_, 2, _>(|i| {
                            ((q[i] - p[i] + 0.5).rem_euclid(1.) - 0.5) / step
                        })
                    };
                    let dx = delta(yarn([uv[0] + step, uv[1]]));
                    let dy = delta(yarn([uv[0], uv[1] + step]));
                    assert_eq!(dx[0] * dx[0] + dx[1] * dx[1], 1.);
                    assert_eq!(dy, [-dx[1], dx[0]]);
                    match direction {
                        Parallel => assert_eq!(dx, [1., 0.]),
                        OpposingCourses => assert_eq!(dx, [if x % 2 == 0 { 1. } else { -1. }, 0.]),
                        QuarterTurn => {
                            assert_eq!(dx, if (x + y) % 2 == 0 { [1., 0.] } else { [0., 1.] })
                        }
                        NonDirectional if cells[0] != cells[1] => assert_eq!(dx[1], 0.),
                        NonDirectional => (),
                    }
                }
            }
            assert_eq!(origins.len(), (cells[0] * cells[1]) as usize);
        }
    }
}

#[test]
fn one_tile_frame_drives_color_relief_roughness_and_occlusion_and_older_recipes_replay() {
    let mut r = carpet(43);
    let f = r.flooring.as_mut().unwrap();
    f.layout = FloorLayout::Straight;
    f.cells = [4; 2];
    f.tile_definition = 0.75;
    f.piece_tone = 0.;
    f.piece_warmth = 0.;
    f.piece_roughness = 0.;
    f.mottling = 0.;
    f.accent_fraction = 0.;
    f.carpet_tiles = Some(CarpetTiles {
        direction: PileDirection::QuarterTurn,
        registration: 0.8,
    });
    let raw = program::MaterialRecipe {
        flooring: None,
        ..r.clone()
    };
    let mut archive = serde_json::to_value(&r).unwrap();
    archive["flooring"]
        .as_object_mut()
        .unwrap()
        .remove("carpet_tiles");
    let global: program::MaterialRecipe = serde_json::from_value(archive).unwrap();
    let dye = r.carpet.as_ref().unwrap();
    for x in 0..4 {
        for y in 0..4 {
            let uv = [(x as f32 + 0.37) / 4., (y as f32 + 0.54) / 4.];
            let f = r.flooring.as_ref().unwrap();
            let cut = f.cut(&r, uv);
            assert_eq!(cut.id, [x, y]);
            assert_eq!(f.seam(&r, &cut, 1), 0.);
            let yarn = f.carpet_tiles.unwrap().yarn_uv(uv, cut.id, f, r.seed);
            for (actual, field_uv) in [(&r, yarn), (&global, uv)] {
                let a = dye.evaluate(actual, uv);
                let b = dye.evaluate(&raw, field_uv);
                assert_eq!(
                    (a.color, a.height, a.roughness, a.occlusion),
                    (b.color, b.height, b.roughness, b.occlusion)
                );
            }
        }
    }
}
