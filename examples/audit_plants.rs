//! Orthographic canopy coverage from actual leaf triangles, independent of RGB.
use bevy::prelude::*;
use bevy_zeroverse::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest, ObjectKind},
    materials::Surface,
    objects, plant_species_name, PLANT_SPECIES,
};
use serde_json::json;

fn coverage(triangles: &[[Vec3; 3]], axes: [usize; 2]) -> f32 {
    let project = |v: Vec3| Vec2::new(v[axes[0]], v[axes[1]]);
    let lo = triangles
        .iter()
        .flatten()
        .map(|&p| project(p))
        .fold(Vec2::splat(f32::INFINITY), Vec2::min);
    let hi = triangles
        .iter()
        .flatten()
        .map(|&p| project(p))
        .fold(Vec2::splat(f32::NEG_INFINITY), Vec2::max);
    if (hi - lo).min_element() < 1e-5 {
        return 0.;
    }
    const N: usize = 128;
    let mut occupied = vec![false; N * N];
    for t in triangles {
        let t = t.map(|p| (project(p) - lo) / (hi - lo) * N as f32);
        let min = t
            .iter()
            .copied()
            .fold(Vec2::splat(f32::INFINITY), Vec2::min)
            .floor()
            .max(Vec2::ZERO);
        let max = t
            .iter()
            .copied()
            .fold(Vec2::splat(f32::NEG_INFINITY), Vec2::max)
            .ceil()
            .min(Vec2::splat((N - 1) as f32));
        let determinant = (t[1] - t[0]).perp_dot(t[2] - t[0]);
        if determinant.abs() < 1e-7 {
            continue;
        }
        for y in min.y as usize..=max.y as usize {
            for x in min.x as usize..=max.x as usize {
                let d = Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - t[0];
                let u = d.perp_dot(t[2] - t[0]) / determinant;
                let v = (t[1] - t[0]).perp_dot(d) / determinant;
                if u >= 0. && v >= 0. && u + v <= 1. {
                    occupied[y * N + x] = true;
                }
            }
        }
    }
    occupied.iter().filter(|&&v| v).count() as f32 / (N * N) as f32
}
fn main() -> anyhow::Result<()> {
    let output = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or("out/plant_audit.json".into()),
    );
    let scene = IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.65, 0, 0.)
        .map_err(anyhow::Error::msg)?;
    let mut rows = Vec::new();
    for species in 0..PLANT_SPECIES {
        for seed in 0..16 {
            let mut o = scene.objects[0].clone();
            o.kind = ObjectKind::Plant;
            o.variant = species;
            o.seed = seed;
            o.finish_seed = None;
            o.position = Vec3::ZERO;
            o.yaw = 0.;
            o.support = None;
            o.interaction_target = None;
            o.size = Vec3::new(
                0.9,
                match species {
                    4 => 0.9,
                    7 => 0.55,
                    9 => 1.1,
                    _ => 1.7,
                },
                0.9,
            );
            let assembly = objects::build_object(&o);
            let leaf: Vec<_> =
                assembly
                    .parts
                    .iter()
                    .filter(|((s, _), _)| {
                        matches!(
                            s,
                            Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated
                        ) || (species == 7 && *s == Surface::Succulent)
                    })
                    .flat_map(|(_, g)| {
                        g.indices.as_chunks::<3>().0.iter().map(|t| {
                            [t[0], t[1], t[2]].map(|i| Vec3::from(g.positions[i as usize]))
                        })
                    })
                    .collect();
            let area: f32 = leaf
                .iter()
                .map(|t| (t[1] - t[0]).cross(t[2] - t[0]).length() * 0.5)
                .sum();
            // Cacti intentionally have a photosynthetic stem, not leaf geometry.
            anyhow::ensure!(
                matches!(species, 8 | 9) || area > 0.,
                "leafless canopy {species}/{seed}"
            );
            rows.push(json!({"species":plant_species_name(species),"seed":seed,"size_m":o.size,
                "leaf_triangle_area_m2":area,"leaf_triangles":leaf.len(),
                "leaf_coverage_xy_xz_zy":if leaf.is_empty(){vec![]}else{vec![coverage(&leaf,[0,1]),coverage(&leaf,[0,2]),coverage(&leaf,[2,1])]}}));
        }
    }
    if let Some(p) = output.parent() {
        std::fs::create_dir_all(p)?;
    }
    std::fs::write(
        output,
        serde_json::to_vec_pretty(
            &json!({"build":bevy_zeroverse::provenance::capture_provenance(),
        "protocol":"All registered habits x 16 prespecified seeds; 128-square orthographic binary coverage within each projected leaf AABB; actual triangles, not visible RGB pixels. Sheet surface area includes both sides. Cacti excluded from leaf coverage by construction.","samples":rows}),
        )?,
    )?;
    Ok(())
}
