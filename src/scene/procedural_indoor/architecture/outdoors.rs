//! Compact exterior context: separated parcels, varied buildings and courtyards.
//! The finite sky closure remains behind these metric, shadow-casting meshes.
use super::super::{layout::IndoorObject, plants};
use super::*;
mod buildings;
mod details;
mod program;
#[cfg(test)]
mod qualification;
use program::{Building, District};
mod streets;

#[derive(Clone, Debug, serde::Serialize)]
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

/// Realized exterior construction controls for dataset distribution reports.
pub(crate) fn measurements(scene: &IndoorManifest) -> Vec<(&'static str, f32)> {
    let d = District::sample(scene.seed);
    let mut out = vec![
        ("exterior_building_probability", d.building_fraction),
        ("exterior_parcel_scale", d.parcel_scale),
        ("exterior_walkway_width_m", d.paving_width),
    ];
    for p in parcels(scene).into_iter().filter(|p| p.kind == 0) {
        let b = Building::sample(&p, super::background::radius(scene));
        out.extend([
            ("exterior_building_width_m", p.size.x),
            ("exterior_building_depth_m", p.size.y),
            ("exterior_storeys", b.levels as f32),
            ("exterior_storey_height_m", b.pitch),
            ("exterior_roof_type", b.roof as f32),
            (
                "exterior_building_wings",
                if b.split < 1. { 2. } else { 1. },
            ),
            ("exterior_window_bay_pitch_m", b.bay_pitch),
            (
                "exterior_window_requested_width_fraction",
                b.window_fraction,
            ),
            ("exterior_window_height_fraction", b.window_height),
            ("exterior_building_distance_m", p.center.length()),
        ]);
    }
    out
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
    let district = District::sample(scene.seed);
    let windows = window_rays(scene);
    let directed = (windows.len() * 9).min(128);
    for attempt in 0..directed + 192 {
        if parcels.len() >= 18 {
            break;
        }
        // Larger continuous footprints break the repeated narrow tower silhouette.
        // Late proposals use smaller plots to fill genuine available gaps.
        let scale = if attempt < directed + 64 {
            district.parcel_scale
        } else {
            1.
        };
        let size = Vec2::new(
            rng.random_range(2.6..4.8) * scale,
            rng.random_range(2.2..4.4) * scale,
        );
        let radius = size.length() * 0.5;
        let (center, yaw) = if attempt < directed {
            let (origin, normal) = windows[attempt % windows.len()];
            let tangent = Vec2::new(-normal.y, normal.x);
            let center = origin
                + normal * rng.random_range(3.2..12.5)
                + tangent * rng.random_range(-3.4..3.4);
            (
                center,
                (-normal.x).atan2(-normal.y) + rng.random_range(-0.10..0.10),
            )
        } else {
            let angle = rng.random_range(0.0..std::f32::consts::TAU);
            (
                Vec2::new(angle.cos(), angle.sin()) * (half.length() + rng.random_range(3.0..10.0)),
                district.street_yaw + rng.random_range(-0.12..0.12),
            )
        };
        let border = 0.65 + district.paving_width;
        let lo = Vec2::new(-half.x - border, -half.y - border);
        let hi = Vec2::new(half.x + border, half.y + NEIGHBOR_DEPTH + border);
        if center.distance(center.clamp(lo, hi)) < radius + 0.5
            || center.length() + radius > super::background::radius(scene) - 1.0
            || parcels
                .iter()
                .any(|p| center.distance(p.center) < radius + p.size.length() * 0.5 + 0.6)
        {
            continue;
        }
        let kind = if rng.random_bool(district.building_fraction as f64) {
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
    a.part(
        surface,
        &super::super::materials::exterior::label(surface, label, p.seed),
    )
    .cuboid(
        size,
        bevel,
        Transform::from_translation(Vec3::new(p.center.x, 0., p.center.y) + rotation * center)
            .with_rotation(rotation),
    );
}
fn garden(a: &mut Assembly, scene: &IndoorManifest, p: &Parcel) {
    let Some(template) = scene.objects.first() else {
        return;
    };
    let mut rng = stream(p.seed, 2);
    let count = if p.kind == 3 {
        2
    } else {
        rng.random_range(1..=4)
    };
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
        if p.kind != 3 {
            let width = (p.size.x / count as f32 * rng.random_range(0.70..0.95))
                .min(p.size.y * if p.kind == 2 { 0.40 } else { 0.70 });
            plant.size = Vec3::new(width, rng.random_range(1.1..4.5), width);
            // Temperate branching trees/shrubs and bamboo, instead of a row of
            // identical tropical indoor pots. Each keeps the existing leaf mesh.
            plant.variant = [6, 10, 13][rng.random_range(0..3)];
            if plant.size.y / width > 5.5 {
                plant.variant = 10;
            }
        }
        plant.position = Vec3::new(p.center.x, 0.03, p.center.y)
            + Quat::from_rotation_y(p.yaw)
                * Vec3::X
                * ((i as f32 - (count - 1) as f32 * 0.5) * p.size.x * 0.30);
        if p.kind != 3 {
            let x = (i as f32 + rng.random_range(0.25..0.75)) / count as f32 - 0.5;
            let z = if p.kind == 2 {
                rng.random_range(-0.22..-0.12)
            } else {
                rng.random_range(-0.15..0.05)
            };
            plant.position = Vec3::new(p.center.x, 0.025, p.center.y)
                + Quat::from_rotation_y(p.yaw) * Vec3::new(x * p.size.x * 0.8, 0., z * p.size.y);
        }
        plant.yaw = p.yaw
            + if p.kind == 3 {
                0.
            } else {
                rng.random_range(-std::f32::consts::PI..std::f32::consts::PI)
            };
        let tf = plant.transform();
        let mut parts = Assembly::default();
        plants::build(&mut parts, &plant);
        let rooted = if p.kind != 3 {
            parts.parts.retain(|(s, _), _| {
                matches!(
                    s,
                    Surface::Bark | Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated
                )
            });
            parts
                .parts
                .iter()
                .filter(|((s, _), _)| *s == Surface::Bark)
                .flat_map(|(_, g)| g.positions.iter().map(|v| v[1]))
                .fold(f32::INFINITY, f32::min)
        } else {
            0.
        };
        for ((surface, _label), mut g) in parts.parts {
            for v in &mut g.positions {
                *v = tf
                    .transform_point(Vec3::from_array(*v) - Vec3::Y * rooted)
                    .to_array();
            }
            for n in &mut g.normals {
                *n = (tf.rotation * Vec3::from_array(*n)).to_array();
            }
            a.part(surface, "plant").append(g);
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
    let mut ground_rng = stream(scene.seed, 6389);
    let ground_surface = if ground_rng.random_bool(0.45) {
        Surface::Soil
    } else {
        Surface::Concrete
    };
    let ground_label =
        super::super::materials::exterior::label(ground_surface, "floor", scene.seed);
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
            ground_surface,
            &ground_label,
            Vec3::new((lo.x + hi.x) * 0.5, -0.17, (lo.y + hi.y) * 0.5),
            Vec3::new(hi.x - lo.x, 0.20, hi.y - lo.y),
            0.,
        );
    }
    streets::build(a, scene);
    for p in parcels(scene) {
        if p.kind == 0 {
            buildings::build(
                a,
                &p,
                &Building::sample(&p, super::background::radius(scene)),
            );
            continue;
        }
        box_at(
            a,
            &p,
            if p.kind == 1 {
                Surface::Soil
            } else {
                Surface::Concrete
            },
            "floor",
            Vec3::new(0., -0.025, 0.),
            Vec3::new(p.size.x, 0.11, p.size.y),
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
            let supports = (p.size.x * 0.60 / 1.6).ceil().max(1.) as usize;
            for i in 0..=supports {
                let x = (i as f32 / supports as f32 - 0.5) * 0.60;
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
                Vec3::new(0., 0.76, p.size.y * 0.32 + 0.12),
                Vec3::new(p.size.x * 0.72, 0.27, 0.055),
                0.005,
            );
            let supports = (p.size.x * 0.60 / 1.6).ceil().max(1.) as usize;
            for i in 0..=supports {
                let x = (i as f32 / supports as f32 - 0.5) * 0.60;
                box_at(
                    a,
                    &p,
                    Surface::Metal,
                    "chair",
                    Vec3::new(p.size.x * x, 0.60, p.size.y * 0.32 + 0.12),
                    Vec3::new(0.045, 0.60, 0.045),
                    0.004,
                );
            }
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
                    if label.split('#').next() != Some("floor") {
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
