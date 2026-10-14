//! Exact clipped viewing regions and the realized connections between them.
//! This is derived from the construction program, never a second layout sampler.
use super::*;
use crate::scene::procedural_indoor::envelope::polygon;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Region {
    pub zone: usize,
    pub primary: bool,
    /// Nonoverlapping X/Z triangles; metres in the manifest's local coordinates.
    pub triangles: Vec<[Vec2; 3]>,
    pub area_m2: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portal {
    pub partition: usize,
    pub regions: [usize; 2],
    pub center: Vec3,
    pub width_m: f32,
    pub headroom_m: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoomTopology {
    pub regions: Vec<Region>,
    pub portals: Vec<Portal>,
    /// A clipped or exterior portal can lack a region on either side. Preserve
    /// its identity rather than inventing a connection or silently omitting it.
    pub boundary_partitions: Vec<usize>,
}

impl IndoorManifest {
    pub fn room_topology(&self) -> RoomTopology {
        let Some(program) = &self.program else {
            return RoomTopology::default();
        };
        let half = self.room_size.xz() * 0.5;
        let rectangle = vec![
            -half,
            Vec2::new(half.x, -half.y),
            half,
            Vec2::new(-half.x, half.y),
        ];
        let outline = self
            .envelope
            .as_ref()
            .map_or(rectangle.as_slice(), |e| &e.footprint);
        let footprint = polygon::triangles(outline);
        let primary_bounds = self.primary_room_bounds();
        let regions = program
            .zones
            .iter()
            .enumerate()
            .map(|(zone, z)| {
                let triangles: Vec<_> = footprint
                    .iter()
                    .flat_map(|t| {
                        let clipped = polygon::clip_box(t, z.min, z.max);
                        (1..clipped.len().saturating_sub(1))
                            .map(|i| [clipped[0], clipped[i], clipped[i + 1]])
                            .filter(|t| (t[1] - t[0]).perp_dot(t[2] - t[0]).abs() > 1e-8)
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let area_m2 = triangles
                    .iter()
                    .map(|t| (t[1] - t[0]).perp_dot(t[2] - t[0]).abs() * 0.5)
                    .sum();
                Region {
                    zone,
                    primary: (z.min, z.max) == primary_bounds,
                    triangles,
                    area_m2,
                }
            })
            .collect();
        let mut result = RoomTopology {
            regions,
            ..default()
        };
        let region_at = |p: Vec2| {
            program.zones.iter().position(|z| {
                p.cmpge(z.min).all() && p.cmple(z.max).all() && polygon::contains(outline, p, 0.)
            })
        };
        for (i, p) in program.partitions.iter().enumerate() {
            let center = p.position(p.door_center, 0.);
            let offset = if p.axis == 0 { Vec2::X } else { Vec2::Y } * (p.thickness * 0.5 + 0.12);
            let pair = [
                region_at(center.xz() - offset),
                region_at(center.xz() + offset),
            ];
            if let [Some(a), Some(b)] = pair {
                if a == b {
                    result.boundary_partitions.push(i);
                    continue;
                }
                result.portals.push(Portal {
                    partition: i,
                    regions: [a, b],
                    center,
                    width_m: p.door_width,
                    headroom_m: p.door_height - p.arch_rise,
                });
            } else {
                result.boundary_partitions.push(i);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn primary_triangles_match_camera_membership_and_connections() {
        for seed in 0..128 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
                    .unwrap();
            let topology = scene.room_topology();
            assert_eq!(topology.regions.iter().filter(|r| r.primary).count(), 1);
            for region in &topology.regions {
                assert!(region.area_m2 > 0.);
                for t in &region.triangles {
                    let c = (t[0] + t[1] + t[2]) / 3.;
                    assert_eq!(
                        scene.in_primary_room(Vec3::new(c.x, 1., c.y), 0.),
                        region.primary
                    );
                }
            }
            assert_eq!(
                topology.portals.len() + topology.boundary_partitions.len(),
                scene.program.as_ref().unwrap().partitions.len()
            );
            for portal in topology.portals {
                assert!(portal.width_m > 0. && portal.headroom_m > 1.8);
                assert!(portal
                    .regions
                    .into_iter()
                    .all(|i| i < topology.regions.len()));
            }
        }
    }
}
