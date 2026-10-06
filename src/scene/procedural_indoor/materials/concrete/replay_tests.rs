//! Independent original casting loop, including its scalar hash/noise evaluations.
use super::*;

pub(in super::super) fn reference(
    c: &ConcreteFinish,
    r: &MaterialRecipe,
    uv: [f32; 2],
    polish: f32,
    t: &mut Texel,
) {
    let [u, v] = uv;
    let boards = frequency(r.period_m, c.board_width_m);
    let warped =
        v * boards as f32 + (periodic_noise(u, v, 3, 5, r.seed.wrapping_add(607)) - 0.5) * 0.025;
    let edge = (warped.rem_euclid(1.) - 0.5).abs();
    let seam = band(
        0.5 - edge,
        c.seam_width_m / r.period_m * boards as f32,
        boards as f32 / r.map_size(0) as f32,
    ) * c.formwork;
    let imprint = periodic_noise(u, v, 4, boards * 19, r.seed.wrapping_add(613)) - 0.5;
    let board_id = (warped.floor() as i32).rem_euclid(boards as i32) as u32;
    let board_tone =
        (super::super::hash(0, board_id, r.seed.wrapping_add(615)) - 0.5) * c.formwork * 0.12;
    let pass_n = frequency(r.period_m, c.trowel_scale_m);
    let passes = deposit(uv, [pass_n, pass_n], 0.16, r.seed.wrapping_add(617)) - 0.5;
    let cure = deposit(uv, [3, 5], 0.17, r.seed.wrapping_add(619)) - 0.5;
    let bleed = (periodic_noise(u, v, 21, 3, r.seed.wrapping_add(631)) - 0.45).max(0.) * c.bleed;
    let sand = periodic_noise(u, v, 173, 167, r.seed.wrapping_add(641)) - 0.5;
    let void_n = frequency(r.period_m, c.bughole_radius_m * 14.);
    let cell = cellular(uv, [void_n, void_n], r.seed.wrapping_add(643));
    let radius = c.bughole_radius_m / r.period_m * void_n as f32 * (0.6 + cell.dye);
    let void = disk(cell.radius, radius, void_n as f32 / r.map_size(0) as f32)
        * smooth((c.bughole_density - cell.dye) * 15.);
    // Grinding removes board impressions and closes the paste's fine relief,
    // while a few casting voids remain. No geological veins are added here.
    let open = 1. - polish * 0.92;
    t.color = tint(
        t.color,
        1. + cure * c.cure_variation + passes * c.trowel * 0.06 + board_tone
            - bleed
            - seam * 0.065
            - void * 0.28 * (1. - polish * 0.5),
    );
    t.height += open
        * (imprint * c.formwork * c.form_relief_m
            + passes * c.trowel * 0.000055
            + sand * c.sand_exposure * 0.000035
            - seam * c.form_relief_m)
        - void * c.bughole_depth_m * (1. - polish * 0.65);
    t.roughness = (t.roughness + cure * 0.05 - passes * c.trowel * 0.12
        + sand * c.sand_exposure * 0.035
        + void * 0.12
        + seam * 0.035)
        .clamp(0.18, 0.99);
    t.occlusion = (t.occlusion * (1. - void * 0.10 - seam * 0.008)).max(0.80);
}

#[test]
fn prepared_casting_replays_scalar_channels_with_shared_budget_fallbacks() {
    for seed in [0, 202, 43_084_482, u64::MAX] {
        let recipes = super::super::program::sample(seed);
        let r = &recipes[super::super::Surface::Concrete as usize];
        let c = r.mineral.as_ref().unwrap().casting.as_ref().unwrap();
        for budget in [0, 32, 1024 * 1024] {
            let mut remaining = budget;
            let p = PreparedConcrete::new(c, r, &mut remaining);
            assert!(remaining <= budget);
            assert_eq!(p.bytes() + remaining, budget);
            for size in [32, 256] {
                let samples = (0..size).flat_map(|y| {
                    (0..size).map(move |x| [x as f32 / size as f32, y as f32 / size as f32])
                });
                for uv in samples.chain([[-1. / 256., 1.], [1., -1. / 256.], [2.3, -0.4]]) {
                    let initial = || Texel {
                        color: [0.13, 0.41, 0.87],
                        height: 0.0017,
                        roughness: 0.47,
                        occlusion: 0.91,
                    };
                    let mut expected = initial();
                    let mut actual = initial();
                    reference(c, r, uv, 0.37, &mut expected);
                    c.apply_prepared(r, uv, 0.37, &mut actual, Some(&p));
                    let bits = |t: Texel| {
                        [
                            t.color[0],
                            t.color[1],
                            t.color[2],
                            t.height,
                            t.roughness,
                            t.occlusion,
                        ]
                        .map(f32::to_bits)
                    };
                    assert_eq!(
                        bits(expected),
                        bits(actual),
                        "seed {seed} budget {budget} uv {uv:?}"
                    );
                }
            }
        }
    }
}
