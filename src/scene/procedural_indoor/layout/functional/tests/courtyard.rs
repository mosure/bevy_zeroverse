use super::*;
use crate::scene::procedural_indoor::{
    architecture::{fixture_half_extents, fixture_positions},
    materials::Surface,
    objects::services::bench,
    validation::validate_layout,
};
use std::collections::BTreeSet;

#[test]
fn courtyard_has_clear_entrance_weatherproof_furniture_paving_and_supported_lights() {
    let mut species = BTreeSet::new();
    let mut layouts = BTreeSet::new();
    let mut spans = Vec::new();
    let mut planted_beds = 0;
    for seed in 0..32 {
        let density = [0., 0.65, 1.][seed as usize % 3];
        let s = IndoorManifest::generate_with_humans(
            seed,
            IndoorLayout::Courtyard,
            density,
            if seed < 6 { 3 } else { 0 },
            0.25,
        )
        .unwrap();
        validate_layout(&s).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        let e = s.envelope.as_ref().unwrap();
        let roof = e.courtyard.unwrap();
        assert!(e
            .walls
            .iter()
            .all(|wall| wall.facade.as_ref().is_none_or(|f| f.shade
                == crate::scene::procedural_indoor::architecture::facade::Shade::None
                && f.treatment.is_none())));
        assert!(e.walls.iter().all(|wall| wall.lining.as_ref().is_none_or(|s|
            s.stock != crate::scene::procedural_indoor::envelope::lining::LiningStock::AcousticFabric)));
        let area = roof.open_area_m2() / s.room_size.xz().element_product();
        spans.push(area);
        assert!(e.pillars.len() >= 4);
        for light in fixture_positions(&s)
            .into_iter()
            .filter(|p| p.z < s.room_size.z * 0.5)
        {
            let half = fixture_half_extents(&s, light).xz();
            assert!(!roof.overlaps(light.xz() - half, light.xz() + half));
        }
        let floor = &s.program.as_ref().unwrap().materials[Surface::Floor as usize];
        floor.validate().unwrap();
        assert!(floor.roughness >= 0.65 && floor.carpet.is_none());
        layouts.insert(format!("{:?}", floor.flooring.as_ref().unwrap().layout));
        let p = &s.functional_programs()[0];
        assert!(p.members.iter().any(|&i| matches!(
            s.objects[i].kind,
            ObjectKind::Plant | ObjectKind::PlantingBed
        )));
        for o in s.objects.iter().filter(|o| !o.neighbor) {
            match o.kind {
                ObjectKind::Bench => {
                    assert_eq!(o.variant, bench::OUTDOOR);
                    assert!(matches!(
                        bench::parameters(o).seat,
                        Surface::Wood | Surface::Concrete | Surface::Metal
                    ));
                }
                ObjectKind::Plant => {
                    species.insert(o.variant);
                }
                ObjectKind::PlantingBed => {
                    let (lo, hi) = o.bounds();
                    assert!(lo.xz().cmpge(roof.min).all() && hi.xz().cmple(roof.max).all());
                    planted_beds += 1;
                }
                _ => {}
            }
        }
        if seed == 0 {
            let replay: IndoorManifest =
                serde_json::from_slice(&serde_json::to_vec(&s).unwrap()).unwrap();
            assert_eq!(s, replay);
            assert_eq!(
                s,
                IndoorManifest::generate_with_humans(
                    seed,
                    IndoorLayout::Courtyard,
                    density,
                    3,
                    0.25
                )
                .unwrap()
            );
            let mut invalid = s.clone();
            invalid.envelope.as_mut().unwrap().courtyard = None;
            assert!(validate_layout(&invalid).is_err());
            let mut unsupported = s.clone();
            unsupported.envelope.as_mut().unwrap().pillars.clear();
            assert!(validate_layout(&unsupported).is_err());
            let mut obstruction = s.clone();
            let col = &mut obstruction.envelope.as_mut().unwrap().pillars[0];
            let r = &p.reservations[0];
            col.center = (r.max + r.min) * 0.5;
            assert!(validate_layout(&obstruction).is_err());
        }
    }
    assert!(species.len() >= 8 && layouts.len() >= 3);
    assert!((10..32).contains(&planted_beds), "{planted_beds}");
    assert!(spans.iter().copied().fold(0., f32::max) - spans.into_iter().fold(1., f32::min) > 0.12);
}

#[test]
fn courtyard_bedded_vegetation_satisfies_sparse_garden_program() {
    let scene =
        IndoorManifest::generate_with_humans(53250, IndoorLayout::Courtyard, 0., 3, 0.).unwrap();
    validate_layout(&scene).unwrap();
    assert_eq!(
        scene.functional_programs()[0].required,
        [ObjectKind::PlantingBed]
    );
}

#[test]
fn courtyard_low_foliage_stays_above_the_bed_base() {
    let scene =
        IndoorManifest::generate_with_humans(53735, IndoorLayout::Courtyard, 0.35, 4, 0.4).unwrap();
    assert!(scene
        .objects
        .iter()
        .any(|o| o.kind == ObjectKind::PlantingBed));
    crate::scene::procedural_indoor::validation::validate_geometry(&scene).unwrap();
}
