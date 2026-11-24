//! Compact exterior context: separated parcels, varied buildings and courtyards.
//! The finite sky closure remains behind these metric, shadow-casting meshes.
use super::super::{layout::IndoorObject, plants};
use super::*;
mod buildings;
mod frontage_layout;
mod frontages;
mod ground;
mod midground;
mod pergola;
mod plinth;
mod program;
#[cfg(test)]
mod qualification;
use program::{Building, District};
mod streets;

const MAX_PARCELS: usize = 18;
const _: () = assert!(MAX_PARCELS <= u32::BITS as usize);

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
struct Parcel {
    center: Vec2,
    size: Vec2,
    yaw: f32,
    kind: u8,
    seed: u64,
}
fn counts(sites: &[Parcel]) -> [usize; 4] {
    let mut counts = [0; 4];
    for p in sites {
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
    let ExteriorPlan {
        windows,
        sites: parcels,
        rows: terraces,
        frontage,
    } = ExteriorPlan::new(scene, &d);
    for (name, count) in [
        "exterior_building_parcels",
        "exterior_garden_parcels",
        "exterior_courtyard_parcels",
        "exterior_pergola_parcels",
    ]
    .into_iter()
    .zip(counts(&parcels))
    {
        out.push((name, count as f32));
    }
    if let Some(fraction) = window_context_fraction(&windows, &parcels) {
        out.push(("exterior_window_context_fraction", fraction));
    }
    if let Some(fraction) = frontage {
        out.push(("exterior_ground_storey_frontage_fraction", fraction));
    }
    for p in parcels.iter().filter(|p| p.kind == 3) {
        pergola::Program::sample(p).measure(&mut out);
    }
    let sides = frontages::viewed_sides(&parcels, &windows);
    out.push((
        "exterior_viewed_frontages",
        sides.iter().flatten().count() as f32,
    ));
    out.push(("exterior_terrace_rows", terraces.len() as f32));
    for row in &terraces {
        out.push(("exterior_terrace_frontage_m", row.size.x));
        out.push(("exterior_terrace_roof_clearance_m", row.eaves_limit));
    }
    let mut entrances = Vec::new();
    for (i, p) in parcels.iter().enumerate().filter(|(_, p)| p.kind == 0) {
        let b = Building::sample(p, super::background::radius(scene), scene.generator_version);
        if let Some(base) = &b.plinth {
            base.measure(&b, p.size, &mut out);
        }
        if let Some(entry) = buildings::main_entrance(p, &b) {
            entrances.push((i, entry));
        }
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
            (
                "exterior_window_nominal_recess_m",
                b.window_installation.recess_m,
            ),
            (
                "exterior_window_sash_rebate_m",
                b.window_installation.rebate_m,
            ),
            ("exterior_building_distance_m", p.center.length()),
        ]);
    }
    ground::measure(scene, &windows, &d, &parcels, &mut out);
    streets::measure(scene, &parcels, &terraces, &entrances, &mut out);
    out
}
/// Permanent facade apertures in the room frame. Landscape placement is
/// independent of the requested cameras, trajectory and world yaw.
#[derive(Clone, Copy)]
struct WindowView {
    center: Vec2,
    normal: Vec2,
    width: f32,
    sill: f32,
    head: f32,
}

fn window_openings(scene: &IndoorManifest) -> Vec<WindowView> {
    let mut windows = Vec::new();
    let mut collect = |tf: Transform, facade: &super::facade::Facade| {
        let normal = (tf.rotation * Vec3::NEG_Z).xz();
        for opening in &facade.openings {
            let x = (opening.min.x + opening.max.x) * 0.5;
            windows.push(WindowView {
                center: tf.transform_point(Vec3::new(x, 0., 0.)).xz(),
                normal,
                width: opening.max.x - opening.min.x,
                sill: opening.min.y,
                head: opening.max.y,
            });
        }
    };
    if let Some(e) = &scene.envelope {
        for wall in &e.walls {
            if let Some(f) = &wall.facade {
                collect(e.wall_transform(wall.edge), f);
            }
        }
    } else if let Some(exterior) = &scene.exterior {
        for f in &exterior.facades {
            collect(f.side.transform(scene.room_size), f);
        }
    }
    if let Some(w) = scene
        .adjoining_elevation
        .as_ref()
        .and_then(|p| p.windows.as_ref())
    {
        if let Some((a, b)) = w.endpoints(scene) {
            collect(super::super::envelope::polygon::wall_frame(a, b), &w.facade);
        }
    }
    windows
}

fn window_context_fraction(rays: &[WindowView], sites: &[Parcel]) -> Option<f32> {
    if rays.is_empty() {
        return None;
    }
    Some(
        rays.iter()
            .filter(|w| {
                sites.iter().any(|s| {
                    let delta = s.center - w.center;
                    delta.normalize_or_zero().dot(w.normal) > 0.85 && delta.length() < 15.
                })
            })
            .count() as f32
            / rays.len() as f32,
    )
}

struct ExteriorPlan {
    windows: Vec<WindowView>,
    sites: Vec<Parcel>,
    rows: Vec<midground::Terrace>,
    frontage: Option<f32>,
}

impl ExteriorPlan {
    fn new(scene: &IndoorManifest, district: &District) -> Self {
        let windows = window_openings(scene);
        let mut sites = propose_parcels(scene, &windows, district);
        // The footprint intersections stay valid when a parcel changes use.
        let horizon = frontage_layout::Horizon::new(&sites, &windows);
        horizon.assign(&mut sites, scene.seed, district.building_fraction);
        let frontage = horizon.coverage(&sites);
        let rows = midground::plan(scene, district, &horizon, &sites);
        Self {
            windows,
            sites,
            rows,
            frontage,
        }
    }
}

fn propose_parcels(
    scene: &IndoorManifest,
    windows: &[WindowView],
    district: &District,
) -> Vec<Parcel> {
    let mut rng = stream(scene.seed, 6340);
    let mut parcels: Vec<Parcel> = vec![];
    let half = scene.room_size.xz() * 0.5;
    let directed = (windows.len() * 9).min(128);
    for attempt in 0..directed + 192 {
        if parcels.len() >= MAX_PARCELS {
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
            let WindowView {
                center: origin,
                normal,
                ..
            } = windows[attempt % windows.len()];
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
        let hi = Vec2::new(
            half.x + border,
            half.y + scene.adjoining_space().max_depth() + border,
        );
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
fn garden_plants(scene: &IndoorManifest, p: &Parcel) -> Vec<IndoorObject> {
    let Some(template) = scene.objects.first() else {
        return Vec::new();
    };
    let mut rng = stream(p.seed, 2);
    let count: i32 = if p.kind == 3 {
        2
    } else {
        rng.random_range(1..=4)
    };
    let pergola_plant_height = (p.kind == 3).then(|| pergola::Program::sample(p).headroom() - 0.12);
    let mut plants = Vec::with_capacity(count as usize);
    for i in 0..count {
        let mut plant: IndoorObject = template.clone();
        plant.kind = ObjectKind::Plant;
        plant.seed = rng.random();
        plant.size = Vec3::new(
            p.size.x * 0.26,
            rng.random_range(1.0..pergola_plant_height.unwrap_or(3.4)),
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
        plants.push(plant);
    }
    plants
}

/// Potted exterior vegetation for callers building a palette without a list of mesh parts.
pub(crate) fn pergola_plants(scene: &IndoorManifest) -> impl Iterator<Item = IndoorObject> + '_ {
    ExteriorPlan::new(scene, &District::sample(scene.seed))
        .sites
        .into_iter()
        .filter(|p| p.kind == 3)
        .flat_map(move |p| garden_plants(scene, &p))
}

fn garden(a: &mut Assembly, p: &Parcel, vegetation: &[IndoorObject], version: u32) {
    for plant in vegetation {
        let tf = plant.transform();
        let mut parts = Assembly::default();
        let rooted = if p.kind != 3 {
            plants::build_in_ground(&mut parts, plant, version);
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
            plants::build_at_version(&mut parts, plant, version);
            0.
        };
        for ((surface, label), mut g) in parts.parts {
            for v in &mut g.positions {
                *v = tf
                    .transform_point(Vec3::from_array(*v) - Vec3::Y * rooted)
                    .to_array();
            }
            for n in &mut g.normals {
                *n = (tf.rotation * Vec3::from_array(*n)).to_array();
            }
            // All parts share the plant class. Keep the finish selected by the
            // planter when this parcel is merged into the exterior architecture.
            a.part(
                surface,
                if label.starts_with("plant#") {
                    &label
                } else {
                    "plant"
                },
            )
            .append(g);
        }
    }
}
pub(crate) fn build(a: &mut Assembly, scene: &IndoorManifest) {
    let district = District::sample(scene.seed);
    let ExteriorPlan {
        windows,
        sites,
        rows,
        ..
    } = ExteriorPlan::new(scene, &district);
    ground::grade(a, scene, &windows, &district);
    let sides = frontages::viewed_sides(&sites, &windows);
    for row in &rows {
        buildings::terrace::build(a, row);
    }
    let mut entrances = Vec::new();
    for (index, (p, side)) in sites.iter().zip(sides).enumerate() {
        if p.kind == 0 {
            if let Some(entry) = buildings::build(
                a,
                p,
                &Building::sample(p, super::background::radius(scene), scene.generator_version),
                side,
            ) {
                entrances.push((index, entry));
            }
            continue;
        }
        let mut vegetation = garden_plants(scene, p);
        if scene.generator_version >= ground::FIRST_VERSION && matches!(p.kind, 2 | 3) {
            ground::patio(a, p, &mut vegetation);
        } else {
            box_at(
                a,
                p,
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
        }
        garden(a, p, &vegetation, scene.generator_version);
        if p.kind == 2 {
            // Courtyard bench, backrest and two legs, kept inside its parcel.
            for z in [-0.22, 0., 0.22] {
                box_at(
                    a,
                    p,
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
                    p,
                    Surface::Metal,
                    "chair",
                    Vec3::new(p.size.x * x, 0.25, p.size.y * 0.32),
                    Vec3::new(0.055, 0.44, 0.32),
                    0.006,
                );
            }
            box_at(
                a,
                p,
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
                    p,
                    Surface::Metal,
                    "chair",
                    Vec3::new(p.size.x * x, 0.60, p.size.y * 0.32 + 0.12),
                    Vec3::new(0.045, 0.60, 0.045),
                    0.004,
                );
            }
        } else if p.kind == 3 {
            pergola::build(a, p);
        }
    }
    streets::build(a, scene, &sites, &rows, &entrances);
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
            let plan = ExteriorPlan::new(&scene, &District::sample(seed));
            fractions.extend(window_context_fraction(&plan.windows, &plan.sites));
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
            let d = District::sample(seed);
            let sites = ExteriorPlan::new(&scene, &d).sites;
            assert_eq!(sites, ExteriorPlan::new(&scene, &d).sites);
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
                            || v.z > h.y + scene.adjoining_space().max_depth() + 0.3
                            || (label.split('#').next() == Some("floor")
                                && scene.adjoining_space().recess(scene.room_size).is_some_and(
                                    |(lo, hi)| v.xz().cmpge(lo - Vec2::splat(0.001)).all()
                                        && v.xz().cmple(hi + Vec2::splat(0.001)).all()
                                )),
                        "seed {seed}: exterior enters room at {v}"
                    );
                }
            }
        }
        assert_eq!(kinds.len(), 4);
    }
}
