use super::*;
use crate::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest},
    validation,
};
use std::collections::BTreeSet;

#[test]
fn wall_depth_is_not_correlated_with_interior_window_sills() {
    let mut sums = [0_f64; 5];
    for seed in 0..1024 {
        let x = ((wall_thickness(seed) / 0.15).ln() / (0.72_f32 / 0.15).ln()) as f64;
        let sill=crate::scene::procedural_indoor::architecture::interior_glazing::InteriorGlazing::sample(seed).sill_m;
        let y = ((sill - 0.028) / 1.25).sqrt() as f64;
        for (sum, value) in sums.iter_mut().zip([x, y, x * x, y * y, x * y]) {
            *sum += value;
        }
    }
    let [sx, sy, sxx, syy, sxy] = sums;
    let correlation =
        (1024. * sxy - sx * sy) / ((1024. * sxx - sx * sx) * (1024. * syy - sy * sy)).sqrt();
    assert!(
        correlation.abs() < 0.12,
        "unintended parameter coupling: {correlation}"
    );
}

#[test]
fn inconsistent_wall_sections_and_intruding_surrounds_are_rejected() {
    let (p, recess) = RevealProgram::sample(3, 0.4, 0.08, 0.06);
    assert!(p.validate(f32::NAN, recess).is_err());
    assert!(p.validate(0.08, f32::NAN).is_err());
    assert!(p.validate(0.08, 0.02).is_err());
    assert!(p.validate(0.08, 0.39).is_err());
    let mut scene =
        IndoorManifest::generate_with_humans(3, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let e = scene.envelope.as_mut().unwrap();
    let f = e.walls.iter_mut().find_map(|w| w.facade.as_mut()).unwrap();
    f.joinery
        .as_mut()
        .unwrap()
        .reveal
        .as_mut()
        .unwrap()
        .wall_thickness_m += 0.01;
    assert!(scene.envelope.as_ref().unwrap().validate(&scene).is_err());
    let mut ext = ExteriorProgram::sample(7, Vec3::new(6., 3., 6.), 0.3);
    let f = &mut ext.facades[0];
    let mut s = crate::scene::procedural_indoor::architecture::trim::TrimProgram::sample(3)
        .base
        .unwrap();
    s.height_m = f.frame_width + 0.001;
    s.depth_m = 0.01;
    f.joinery
        .as_mut()
        .unwrap()
        .reveal
        .as_mut()
        .unwrap()
        .surround = Some(s);
    assert!(ext.validate(Vec3::new(6., 3., 6.)).is_err());
}

#[test]
fn wall_and_opening_sections_replay_with_independent_metric_scales() {
    let mut depths = Vec::new();
    let mut returns = Vec::new();
    let mut profiles = BTreeSet::new();
    let mut finishes = BTreeSet::new();
    let mut absent = 0;
    for seed in 0..1024 {
        let t = wall_thickness(seed);
        let (p, recess) = RevealProgram::sample(seed, t, 0.08, 0.06);
        p.validate(0.08, recess).unwrap();
        assert_eq!(
            (p.clone(), recess),
            RevealProgram::sample(seed, t, 0.08, 0.06)
        );
        assert_eq!(
            p,
            serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap()
        );
        depths.push(t);
        returns.push(recess - 0.04);
        finishes.insert(format!("{:?}", p.lining_surface));
        if let Some(s) = p.surround {
            profiles.insert(format!("{:?}", s.profile));
        } else {
            absent += 1;
        }
    }
    depths.sort_by(f32::total_cmp);
    returns.sort_by(f32::total_cmp);
    assert!(depths[51] < 0.18 && depths[972] > 0.62);
    assert!(returns[51] < 0.035 && returns[972] > 0.30);
    assert_eq!(profiles.len(), 5);
    assert_eq!(finishes.len(), 5);
    assert!((200..400).contains(&absent));
}

#[test]
fn deep_returns_have_finite_outward_meshes_and_preserve_clear_pane_centres() {
    for seed in 0..96 {
        let mut f = ExteriorProgram::sample(seed, Vec3::new(6., 3., 6.), 0.24)
            .facades
            .remove(0);
        f.shade = Shade::None;
        f.treatment = None;
        for o in &mut f.openings {
            o.glazing = None;
            o.operable = false;
        }
        for o in &f.openings {
            let mut a = Assembly::default();
            build(&mut a, Transform::IDENTITY, &f, o);
            for ((_, label), g) in &a.parts {
                assert!(label.starts_with("wall#") || label.starts_with("window#"));
                for t in g.indices.as_chunks::<3>().0 {
                    let v = t.map(|i| Vec3::from(g.positions[i as usize]));
                    let n = (v[1] - v[0]).cross(v[2] - v[0]);
                    assert!(
                        n.is_finite() && n.length_squared() > 1e-14,
                        "{seed}/{label}"
                    );
                    assert!(n.dot(Vec3::from(g.normals[t[0] as usize])) > 0.);
                    for p in v {
                        assert!(
                            p.x >= o.min.x - 1e-5
                                && p.x <= o.max.x + 1e-5
                                && p.y >= o.min.y - 1e-5
                                && p.y <= o.max.y + 1e-5
                        );
                    }
                }
            }
            // Every pane centre has a clear normal ray through the return.
            for (lo, hi) in f.pane_rectangles(o) {
                let c = (lo + hi) * 0.5;
                let r = f.joinery.as_ref().unwrap().reveal.as_ref().unwrap();
                let border = r.surround.as_ref().map_or(r.lining_thickness_m, |s| {
                    s.height_m.max(r.lining_thickness_m)
                });
                assert!(
                    c.cmpgt(o.min + Vec2::splat(border)).all()
                        && c.cmplt(o.max - Vec2::splat(border)).all()
                );
            }
        }
    }
}

#[test]
fn legacy_sections_keep_their_original_depth_and_new_walls_expand_crop_bounds() {
    let mut scene =
        IndoorManifest::generate_with_humans(3, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let e = scene.envelope.as_mut().unwrap();
    e.wall_thickness_m = Some(0.70);
    for w in &mut e.walls {
        if let Some(f) = &mut w.facade {
            let (p, recess) = RevealProgram::sample(scene.seed, 0.70, f.frame_depth, f.frame_width);
            f.recess = recess;
            f.joinery.as_mut().unwrap().reveal = Some(p);
        }
    }
    scene.envelope.as_ref().unwrap().validate(&scene).unwrap();
    let region = scene.primary_region();
    let (lo, hi) = region.shell_bounds();
    assert!(lo.x <= region.min.x - 0.719 && hi.z >= region.max.y + 0.719);
    let mut json = serde_json::to_value(&scene).unwrap();
    json["envelope"]
        .as_object_mut()
        .unwrap()
        .remove("wall_thickness_m");
    for wall in json["envelope"]["walls"].as_array_mut().unwrap() {
        if let Some(f) = wall["facade"].as_object_mut() {
            f.insert("recess".into(), serde_json::json!(0.08));
            f.get_mut("joinery")
                .unwrap()
                .as_object_mut()
                .unwrap()
                .remove("reveal");
        }
    }
    let old: IndoorManifest = serde_json::from_value(json).unwrap();
    assert_eq!(old.envelope.as_ref().unwrap().wall_thickness(), 0.18);
    old.envelope.as_ref().unwrap().validate(&old).unwrap();
    validation::validate_geometry(&old).unwrap();
}
