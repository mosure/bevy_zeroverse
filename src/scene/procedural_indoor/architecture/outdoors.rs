//! Compact exterior context: separated parcels, varied buildings and courtyards.
//! The finite sky closure remains behind these metric, shadow-casting meshes.
use super::super::{layout::IndoorObject, plants};
use super::*;
mod details;
mod streets;

#[derive(Clone, Debug)]
struct Parcel {
    center: Vec2,
    size: Vec2,
    yaw: f32,
    kind: u8,
    seed: u64,
}
pub(crate) fn counts(scene: &IndoorManifest) -> [usize; 4] {
    let mut counts = [0; 4];
    for p in parcels(scene) {
        counts[p.kind as usize] += 1;
    }
    counts
}
/// Window origins and outward normals are measured from the built envelope,
/// including angled/recessed edges, rather than its enclosing rectangle.
fn window_rays(scene: &IndoorManifest) -> Vec<(Vec2, Vec2)> {
    let mut rays = Vec::new();
    if let Some(e) = &scene.envelope {
        for wall in &e.walls {
            if let Some(f) = &wall.facade {
                let tf = e.wall_transform(wall.edge);
                for w in &f.openings {
                    rays.push((
                        tf.transform_point(Vec3::new((w.min.x + w.max.x) * 0.5, 0., 0.))
                            .xz(),
                        (tf.rotation * Vec3::NEG_Z).xz(),
                    ));
                }
            }
        }
    } else if let Some(exterior) = &scene.exterior {
        for f in &exterior.facades {
            let tf = f.side.transform(scene.room_size);
            for w in &f.openings {
                rays.push((
                    tf.transform_point(Vec3::new((w.min.x + w.max.x) * 0.5, 0., 0.))
                        .xz(),
                    (tf.rotation * Vec3::NEG_Z).xz(),
                ));
            }
        }
    }
    rays
}

pub(crate) fn window_context_fraction(scene: &IndoorManifest) -> Option<f32> {
    let rays = window_rays(scene);
    let sites = parcels(scene);
    if rays.is_empty() {
        return None;
    }
    Some(
        rays.iter()
            .filter(|(p, n)| {
                sites.iter().any(|s| {
                    let delta = s.center - *p;
                    delta.normalize_or_zero().dot(*n) > 0.85 && delta.length() < 15.
                })
            })
            .count() as f32
            / rays.len() as f32,
    )
}

fn parcels(scene: &IndoorManifest) -> Vec<Parcel> {
    let mut rng = stream(scene.seed, 6340);
    let mut parcels: Vec<Parcel> = vec![];
    let half = scene.room_size.xz() * 0.5;
    let density = rng.random_range(0.35..0.85);
    let windows = window_rays(scene);
    let directed = (windows.len() * 6).min(96);
    for attempt in 0..directed + 96 {
        if parcels.len() >= 18 {
            break;
        }
        let size = Vec2::new(rng.random_range(2.2..4.8), rng.random_range(2.0..4.0));
        let radius = size.length() * 0.5;
        let (center, yaw) = if attempt < directed {
            let (origin, normal) = windows[attempt % windows.len()];
            let tangent = Vec2::new(-normal.y, normal.x);
            let center = origin
                + normal * rng.random_range(3.2..10.0)
                + tangent * rng.random_range(-1.4..1.4);
            (
                center,
                (-normal.x).atan2(-normal.y) + rng.random_range(-0.10..0.10),
            )
        } else {
            let angle = rng.random_range(0.0..std::f32::consts::TAU);
            (
                Vec2::new(angle.cos(), angle.sin()) * (half.length() + rng.random_range(3.0..10.0)),
                rng.random_range(-0.4..0.4),
            )
        };
        let lo = Vec2::new(-half.x - 0.8, -half.y - 0.8);
        let hi = Vec2::new(half.x + 0.8, half.y + NEIGHBOR_DEPTH + 0.8);
        if center.distance(center.clamp(lo, hi)) < radius + 0.4
            || center.length() + radius > super::background::radius(scene) - 1.0
            || parcels
                .iter()
                .any(|p| center.distance(p.center) < radius + p.size.length() * 0.5 + 0.6)
        {
            continue;
        }
        let kind = if rng.random_bool(density) {
            0
        } else {
            rng.random_range(1..=3)
        };
        parcels.push(Parcel {
            center,
            size,
            yaw,
            kind,
            seed: rng.random(),
        });
    }
    parcels
}
fn box_at(
    a: &mut Assembly,
    p: &Parcel,
    surface: Surface,
    label: &str,
    center: Vec3,
    size: Vec3,
    bevel: f32,
) {
    let rotation = Quat::from_rotation_y(p.yaw);
    a.part(surface, label).cuboid(
        size,
        bevel,
        Transform::from_translation(Vec3::new(p.center.x, 0., p.center.y) + rotation * center)
            .with_rotation(rotation),
    );
}
fn building(a: &mut Assembly, p: &Parcel, sky_radius: f32) {
    let mut rng = stream(p.seed, 1);
    let requested_levels = rng.random_range(1..=4);
    let pitch = rng.random_range(2.5_f32..3.2);
    // Leave roof equipment below the finite sky: a distant tall building must
    // not be sliced by the annotation closure's near-range sphere.
    let horizontal = p.center.length() + p.size.length() * 0.5 + 0.10;
    let available_height = (sky_radius.powi(2) - horizontal.powi(2)).max(0.).sqrt() - 0.8;
    let levels = requested_levels.min((available_height / pitch).floor().max(1.) as usize);
    let setback = rng.random_range(0.04..0.16);
    let finish = [Surface::Concrete, Surface::Paint, Surface::WoodEdge][rng.random_range(0..3)];
    for level in 0..levels {
        let size = p.size * (1. - level as f32 * setback);
        let bottom = level as f32 * pitch - 0.08;
        box_at(
            a,
            p,
            finish,
            "other_structure",
            Vec3::new(0., bottom + pitch * 0.5, 0.),
            Vec3::new(size.x, pitch, size.y),
            0.012,
        );
        details::facade(a, p, size, bottom, pitch, level);
        // Roof slabs/cornices are separated from facade glazing, with variable
        // aperture spacing and aspect instead of the same two windows everywhere.
        box_at(
            a,
            p,
            Surface::Concrete,
            "other_structure",
            Vec3::new(0., bottom + pitch + 0.035, 0.),
            Vec3::new(size.x + 0.10, 0.07, size.y + 0.10),
            0.01,
        );
        for side in 0..4 {
            let across = if side % 2 == 0 { size.x } else { size.y };
            let columns = (across / rng.random_range(0.75..1.5)).floor().max(1.) as usize;
            let width = across / columns as f32 * rng.random_range(0.45..0.78);
            let height = pitch * rng.random_range(0.44..0.70);
            for column in 0..columns {
                let u = (column as f32 + 0.5) * across / columns as f32 - across * 0.5;
                let sign = if side < 2 { 1. } else { -1. };
                let (pos, dims) = if side % 2 == 0 {
                    (
                        Vec3::new(u, bottom + pitch * 0.53, sign * (size.y * 0.5 + 0.009)),
                        Vec3::new(width, height, 0.012),
                    )
                } else {
                    (
                        Vec3::new(sign * (size.x * 0.5 + 0.009), bottom + pitch * 0.53, u),
                        Vec3::new(0.012, height, width),
                    )
                };
                box_at(
                    a,
                    p,
                    Surface::Metal,
                    "window",
                    pos,
                    dims + Vec3::new(
                        if side % 2 == 0 { 0.06 } else { 0. },
                        0.06,
                        if side % 2 == 0 { 0. } else { 0.06 },
                    ),
                    0.,
                );
                let normal = if side % 2 == 0 {
                    Vec3::Z * sign
                } else {
                    Vec3::X * sign
                };
                box_at(
                    a,
                    p,
                    Surface::Glass,
                    "window",
                    pos + normal * 0.009,
                    dims,
                    0.,
                );
            }
        }
    }
    // Rooftop service equipment: low, bounded profiles, no added light sources.
    if rng.random_bool(0.6) {
        let roof = levels as f32 * pitch - 0.01;
        box_at(
            a,
            p,
            Surface::Metal,
            "other_structure",
            Vec3::new(0., roof + 0.20, 0.),
            Vec3::new(p.size.x * 0.24, 0.40, p.size.y * 0.27),
            0.012,
        );
    }
}
fn garden(a: &mut Assembly, scene: &IndoorManifest, p: &Parcel) {
    let Some(template) = scene.objects.first() else {
        return;
    };
    let mut rng = stream(p.seed, 2);
    let count = if p.kind == 1 { 3 } else { 2 };
    for i in 0..count {
        let mut plant: IndoorObject = template.clone();
        plant.kind = ObjectKind::Plant;
        plant.seed = rng.random();
        plant.size = Vec3::new(
            p.size.x * 0.26,
            rng.random_range(1.0..if p.kind == 3 { 2.45 } else { 3.4 }),
            p.size.y * 0.40,
        );
        plant.variant = plants::sample_species(plant.size, &mut rng);
        plant.position = Vec3::new(p.center.x, 0.03, p.center.y)
            + Quat::from_rotation_y(p.yaw)
                * Vec3::X
                * ((i as f32 - (count - 1) as f32 * 0.5) * p.size.x * 0.30);
        plant.yaw = p.yaw;
        let tf = plant.transform();
        let mut parts = Assembly::default();
        plants::build(&mut parts, &plant);
        for ((surface, label), mut g) in parts.parts {
            for v in &mut g.positions {
                *v = tf.transform_point(Vec3::from_array(*v)).to_array();
            }
            for n in &mut g.normals {
                *n = (tf.rotation * Vec3::from_array(*n)).to_array();
            }
            a.part(surface, &label).append(g);
        }
    }
}
pub(crate) fn build(a: &mut Assembly, scene: &IndoorManifest) {
    let half = scene.room_size.xz() * 0.5;
    let (left, right, back, front) = (
        -half.x - 0.5,
        half.x + 0.5,
        -half.y - 0.5,
        half.y + NEIGHBOR_DEPTH + 0.5,
    );
    for (lo, hi) in [
        (
            Vec2::new(left - 18., back - 18.),
            Vec2::new(left, front + 18.),
        ),
        (
            Vec2::new(right, back - 18.),
            Vec2::new(right + 18., front + 18.),
        ),
        (Vec2::new(left, back - 18.), Vec2::new(right, back)),
        (Vec2::new(left, front), Vec2::new(right, front + 18.)),
    ] {
        a.box_part(
            Surface::Concrete,
            "floor",
            Vec3::new((lo.x + hi.x) * 0.5, -0.17, (lo.y + hi.y) * 0.5),
            Vec3::new(hi.x - lo.x, 0.20, hi.y - lo.y),
            0.,
        );
    }
    streets::build(a, scene);
    for p in parcels(scene) {
        if p.kind == 0 {
            building(a, &p, super::background::radius(scene));
            continue;
        }
        box_at(
            a,
            &p,
            Surface::Concrete,
            "floor",
            Vec3::new(0., -0.015, 0.),
            Vec3::new(p.size.x, 0.09, p.size.y),
            0.012,
        );
        garden(a, scene, &p);
        if p.kind == 2 {
            // Courtyard bench, backrest and two legs, kept inside its parcel.
            for z in [-0.22, 0., 0.22] {
                box_at(
                    a,
                    &p,
                    Surface::Wood,
                    "chair",
                    Vec3::new(0., 0.47, p.size.y * 0.32 + z * 0.35),
                    Vec3::new(p.size.x * 0.72, 0.045, 0.064),
                    0.004,
                );
            }
            for x in [-0.26, 0.26] {
                box_at(
                    a,
                    &p,
                    Surface::Metal,
                    "chair",
                    Vec3::new(p.size.x * x, 0.25, p.size.y * 0.32),
                    Vec3::new(0.055, 0.44, 0.32),
                    0.006,
                );
            }
            box_at(
                a,
                &p,
                Surface::Wood,
                "chair",
                Vec3::new(0., 0.76, p.size.y * 0.39),
                Vec3::new(p.size.x * 0.72, 0.27, 0.055),
                0.005,
            );
        } else if p.kind == 3 {
            details::pergola(a, &p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn window_facing_context_is_broadly_available() {
        let mut fractions = Vec::new();
        for seed in 0..128 {
            let scene = IndoorManifest::generate_with_humans(
                seed,
                super::super::super::layout::IndoorLayout::Mixed,
                0.65,
                0,
                0.,
            )
            .unwrap();
            if !window_rays(&scene).is_empty() {
                fractions.push(window_context_fraction(&scene).unwrap());
            }
        }
        let mean = fractions.iter().sum::<f32>() / fractions.len() as f32;
        assert!(mean > 0.85, "window-normal near-field coverage {mean:.3}");
        assert!(
            fractions.iter().filter(|&&x| x < 0.5).count() <= 3,
            "many empty window corridors: {fractions:?}"
        );
    }
    #[test]
    fn parcels_are_replayable_disjoint_and_external() {
        let mut kinds = std::collections::BTreeSet::new();
        for seed in 0..128 {
            let scene = IndoorManifest::generate_with_humans(
                seed,
                super::super::super::layout::IndoorLayout::Mixed,
                0.65,
                0,
                0.,
            )
            .unwrap();
            let sites = parcels(&scene);
            assert_eq!(format!("{sites:?}"), format!("{:?}", parcels(&scene)));
            assert!(sites.len() >= 4 && sites.len() <= 18);
            for (i, p) in sites.iter().enumerate() {
                kinds.insert(p.kind);
                for q in &sites[i + 1..] {
                    assert!(
                        p.center.distance(q.center)
                            > p.size.length() * 0.5 + q.size.length() * 0.5 + 0.59
                    );
                }
            }
            let mut a = Assembly::default();
            build(&mut a, &scene);
            let h = scene.room_size.xz() * 0.5;
            for ((_, label), g) in &a.parts {
                for point in &g.positions {
                    let v = Vec3::from_array(*point);
                    assert!(v.is_finite());
                    if label != "floor" {
                        assert!(
                            v.length() < super::super::background::radius(&scene),
                            "seed {seed}: {label} crosses sky at {v}"
                        );
                    }
                    assert!(
                        v.x.abs() > h.x + 0.3
                            || v.z < -h.y - 0.3
                            || v.z > h.y + NEIGHBOR_DEPTH + 0.3,
                        "seed {seed}: exterior enters room at {v}"
                    );
                }
            }
        }
        assert_eq!(kinds.len(), 4);
    }
}
