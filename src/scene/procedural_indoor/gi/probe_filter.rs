//! Spatial reconstruction of noisy diffuse irradiance. Each cardinal lobe keeps
//! its own RGB estimate and sampling variance. Neighbours must see one another.
use super::{BakeScene, UVec3, Vec3, Vec4};

pub(super) type Lobes = [Vec4; 6];

/// Contributions already include 1/N. The trace of the RGB sample-mean
/// covariance is a noise estimate; the spherical primary rays are stratified.
pub(super) fn finish_moments(sum: Vec4, samples: u32) -> Vec4 {
    let n = samples as f32;
    let rgb = sum.truncate();
    if samples <= 1 {
        return rgb.extend(0.);
    }
    rgb.extend(((n * sum.w - rgb.length_squared()) / (n - 1.)).max(0.))
}

impl BakeScene {
    pub(super) fn probe_links(&self, grid: UVec3, points: &[Vec3], valid: &[bool]) -> Vec<u8> {
        let mut links = vec![0; points.len()];
        if !self
            .probe_domain
            .as_ref()
            .is_some_and(|d| d.filter_irradiance)
        {
            return links;
        }
        let strides = [1, grid.x as usize, (grid.x * grid.y) as usize];
        for (i, &point) in points.iter().enumerate().filter(|(i, _)| valid[*i]) {
            let cell = UVec3::new(
                i as u32 % grid.x,
                i as u32 / grid.x % grid.y,
                i as u32 / (grid.x * grid.y),
            );
            for axis in 0..3 {
                let j = i + strides[axis];
                if cell[axis] + 1 >= grid[axis] || !valid[j] {
                    continue;
                }
                if !self.occluded(point, Vec3::AXES[axis], points[j][axis] - point[axis]) {
                    links[i] |= 1 << (axis * 2);
                    links[j] |= 1 << (axis * 2 + 1);
                }
            }
        }
        links
    }
}

/// Symmetric flux exchange preserves the sum of each RGB lobe over valid
/// probes, including at walls with fewer neighbours. A statistically resolved
/// colour or intensity difference suppresses the exchange. See GPU resolve.
pub(super) fn reconstruct(raw: &[Lobes], i: usize, links: &[u8], grid: UVec3) -> [Vec3; 6] {
    let center = raw[i];
    let mut flux = [Vec3::ZERO; 6];
    let mask = links[i];
    let degree = mask.count_ones();
    for (axis, stride) in [1, grid.x as usize, (grid.x * grid.y) as usize]
        .into_iter()
        .enumerate()
    {
        for side in 0..2 {
            if mask & (1 << (axis * 2 + side)) == 0 {
                continue;
            }
            let j = if side == 0 { i + stride } else { i - stride };
            // The sum of all weights is at most 0.85 at either endpoint.
            let spatial = 0.85 / degree.max(links[j].count_ones()) as f32;
            for face in 0..6 {
                let other = raw[j][face];
                let threshold = 9. * (center[face].w + other.w);
                let delta = other.truncate() - center[face].truncate();
                let difference = delta.length_squared();
                if difference < threshold {
                    let confidence = 1. - difference / threshold;
                    flux[face] += delta * (spatial * confidence * confidence);
                }
            }
        }
    }
    std::array::from_fn(|f| center[f].truncate() + flux[f])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        gi::{probe_domain::ProbeDomain, tests::empty},
        layout::{IndoorLayout, IndoorManifest},
        materials::Surface,
        objects::Assembly,
    };
    use bevy::prelude::Transform;

    #[test]
    fn variance_estimates_mean_error_in_linear_rgb() {
        // Two rays return 0 and (2,4,0). The stored contributions include 1/N.
        let sum = Vec4::new(1., 2., 0., 5.);
        assert_eq!(finish_moments(sum, 2), Vec4::new(1., 2., 0., 5.));
        assert_eq!(finish_moments(Vec4::new(2., 4., 0., 10.), 2).w, 0.);
    }

    #[test]
    fn links_stop_at_real_partition_mesh_invalid_cells_and_grid_edges() {
        let room = IndoorManifest::generate(58103, IndoorLayout::Gymnasium, 0.65, 0).unwrap();
        let mut scene = empty();
        let mut assembly = Assembly::default();
        assembly.box_part(
            Surface::Paint,
            "partition",
            Vec3::new(0.5, 0., 0.),
            Vec3::new(0.035, 6., 6.),
            0.,
        );
        for part in assembly.parts.values() {
            scene.add_geometry(part, Transform::IDENTITY, 0);
        }
        drop(scene.build_bvh());
        let res = UVec3::new(3, 2, 2);
        let points: Vec<_> = (0..12)
            .map(|i| Vec3::new((i % 3) as f32, (i / 3 % 2) as f32, (i / 6) as f32))
            .collect();
        let mut valid = [true; 12];
        valid[11] = false;
        // Legacy bake stays raw, including on replay of pre-filter manifests.
        let mut old_room = room.clone();
        old_room.generator_version = 288;
        scene.probe_domain = ProbeDomain::from_manifest(&old_room);
        assert_eq!(scene.probe_links(res, &points, &valid), [0; 12]);
        scene.probe_domain = ProbeDomain::from_manifest(&room);
        let mask = scene.probe_links(res, &points, &valid);
        for (i, p) in points.iter().enumerate() {
            let mut expected = 0;
            for (e, delta) in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z]
                .into_iter()
                .enumerate()
            {
                if let Some(j) = points.iter().position(|q| *q == *p + delta) {
                    let cuts_wall = (p.x < 0.5) != (points[j].x < 0.5);
                    if valid[i] && valid[j] && !cuts_wall {
                        expected |= 1 << e;
                    }
                }
            }
            assert_eq!(mask[i], expected, "probe {i}");
        }
    }

    #[test]
    fn reconstruction_suppresses_noise_per_lobe_and_preserves_light_and_colour_edges() {
        let grid = UVec3::new(3, 1, 1);
        let mut raw = [[Vec4::ONE; 6]; 3];
        for slot in [0, 2] {
            raw[slot][0] = Vec4::new(0.8, 0.8, 0.8, 0.5);
        }
        raw[0][1] = Vec4::new(0., 1., 0., 0.0001);
        raw[1][1] = Vec4::new(1., 0., 0., 0.0001);
        raw[2][1] = raw[1][1];
        raw[1][2] = Vec4::splat(16.);
        let filtered = reconstruct(&raw, 1, &[1, 3, 2], grid);
        assert!(filtered[0].x > 0.8 && filtered[0].x < 0.91);
        assert_eq!(filtered[1], Vec3::X);
        assert_eq!(filtered[2], Vec3::splat(16.));
        assert!(filtered[3].abs_diff_eq(Vec3::ONE, 1e-6));
        assert_eq!(
            reconstruct(&raw, 1, &[0; 3], grid),
            raw[1].map(Vec4::truncate)
        );
    }
    #[test]
    fn exchanging_noisy_lobes_conserves_each_colour_on_an_irregular_graph() {
        // Four connected probes and one isolated cell. Every direction has a
        // different colour and uncertainty; the endpoints have different degrees.
        let mask = [1, 3, 3, 2, 0];
        let raw: Vec<Lobes> = (0..5)
            .map(|i| {
                std::array::from_fn(|f| {
                    let x = (2 * i + f) as f32;
                    Vec4::new(x * 0.3, 2. + x, 1. / (x + 1.), 1. + 0.05 * x)
                })
            })
            .collect();
        let result: Vec<_> = (0..5)
            .map(|i| reconstruct(&raw, i, &mask, UVec3::new(5, 1, 1)))
            .collect();
        for f in 0..6 {
            let before: Vec3 = raw.iter().map(|v| v[f].truncate()).sum();
            let after: Vec3 = result.iter().map(|v| v[f]).sum();
            assert!(before.abs_diff_eq(after, 5e-6));
            assert_eq!(result[4][f], raw[4][f].truncate());
        }
    }
}
