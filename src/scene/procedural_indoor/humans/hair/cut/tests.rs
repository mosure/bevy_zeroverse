use super::*;
use crate::scene::procedural_indoor::humans::{
    hair::{finish_material, HairProgram},
    sample_person, HumanPoseKind, HumanSurface,
};

#[test]
fn seeded_cuts_span_real_lengths_without_assigning_parts_to_every_style() {
    let cuts: Vec<_> = (0..1024).map(ScalpCut::sample).collect();
    assert!(cuts.iter().all(|c| c.valid()));
    assert_eq!(cuts[46], ScalpCut::sample(46));
    let mut pixie = 0;
    let mut bob = 0;
    for c in &cuts {
        assert_eq!(c.part_width(HairStyle::SidePart), c.part_width_m);
        for s in [
            HairStyle::Bald,
            HairStyle::Buzz,
            HairStyle::Afro,
            HairStyle::Ponytail,
        ] {
            assert_eq!(c.part_width(s), 0.);
        }
        pixie += usize::from(c.part_width(HairStyle::Pixie) > 0.);
        bob += usize::from(c.part_width(HairStyle::Bob) > 0.);
        assert!((0.1..0.95).contains(&c.buzz_coverage()));
    }
    assert!((330..475).contains(&pixie) && (750..880).contains(&bob));
    let observed = |f: fn(&ScalpCut) -> f32| {
        let mut v: Vec<_> = cuts.iter().map(f).collect();
        v.sort_by(f32::total_cmp);
        [v[51], v[512], v[972]]
    };
    let q = [
        observed(|c| c.clip_length_m * 1000.),
        observed(|c| c.temple_recession_m * 1000.),
        observed(|c| c.part_width_m * 1000.),
        observed(|c| c.buzz_coverage()),
    ];
    assert!(q[0][0] < 1.2 && q[0][2] > 6.4 && q[1][0] < 3. && q[1][2] > 32.);
    assert!(q[2][0] < 1.9 && q[2][2] > 5.1 && q[3][2] - q[3][0] > 0.33);
    println!("1024 scalp seeds; p05/p50/p95 (guard mm, temple mm, optional part mm, buzz coverage): {q:?}; parts pixie={pixie} bob={bob}");
    let legacy: HairProgram = serde_json::from_str("{}").unwrap();
    assert!(legacy.scalp_cut.is_none() && legacy.validate().is_ok());
    let mut invalid = HairProgram::sample(3);
    invalid.scalp_cut.as_mut().unwrap().part_width_m = f32::NAN;
    assert!(invalid.validate().is_err());
}

#[test]
fn part_tapers_out_at_the_crown_and_below_the_temple() {
    let part = Part {
        width: 0.004,
        crown: 0.45,
        start_x: 0.22,
        curve: -0.35,
    };
    let p = |y, z| part.coordinates(Vec3::new(0., y, z), 0.09);
    assert_eq!(p(0.8, -0.8).y, 0.002);
    assert_eq!(p(0.8, 0.46).y, 0.);
    assert_eq!(p(0.01, -0.8).y, 0.);
    assert!(p(0.7, 0.28).y > p(0.7, 0.37).y);
    assert!(p(0.8, -0.8).x < -0.01 && p(0.8, -0.8).x != p(0.8, 0.20).x);
}

#[test]
fn close_clips_show_person_specific_scalp_colour_and_reduce_fibre_alignment() {
    let mut h = sample_person(
        27,
        0,
        Vec3::ZERO,
        0.,
        HumanPoseKind::StandingRelaxed,
        None,
        false,
    );
    h.hairstyle = HairStyle::Buzz as u8;
    let mut finishes = Vec::new();
    for guard in [0.0009, 0.0078] {
        let a = h.appearance.as_mut().unwrap();
        a.hair_program.scalp_cut.as_mut().unwrap().clip_length_m = guard;
        let coverage = a.hair_program.scalp_cut.unwrap().buzz_coverage();
        let pigment = a.color(HumanSurface::Hair).unwrap().to_linear();
        let skin = a.color(HumanSurface::Skin).unwrap().to_linear();
        let brow = a.color(HumanSurface::Brow).unwrap();
        assert_eq!(
            h.material_color(HumanSurface::Hair).to_linear(),
            skin.mix(&pigment, coverage)
        );
        assert_eq!(h.material_color(HumanSurface::Brow), brow);
        let mut m = StandardMaterial::default();
        finish_material(&mut m, &h);
        assert_eq!(m.alpha_mode, AlphaMode::Opaque);
        assert_eq!(m.metallic, 0.);
        finishes.push((coverage, m));
    }
    let (short, a) = &finishes[0];
    let (long, b) = &finishes[1];
    assert!(
        short < long
            && a.anisotropy_strength < b.anisotropy_strength
            && a.perceptual_roughness > b.perceptual_roughness
    );
    assert_eq!(a.uv_transform.matrix2.x_axis, b.uv_transform.matrix2.x_axis);
    assert!(a.uv_transform.matrix2.y_axis.y > b.uv_transform.matrix2.y_axis.y * 6.);
    let pigment = h
        .appearance
        .as_ref()
        .unwrap()
        .color(HumanSurface::Hair)
        .unwrap();
    h.appearance.as_mut().unwrap().hair_program.scalp_cut = None;
    assert_eq!(h.material_color(HumanSurface::Hair), pigment);
}
