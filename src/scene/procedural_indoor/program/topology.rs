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
    /// Static door state. A geometric portal is not necessarily traversable.
    #[serde(default)]
    pub door: Option<super::super::architecture::doors::DoorProgram>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenConnection {
    pub regions: [usize; 2],
    pub start: Vec2,
    pub end: Vec2,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoomTopology {
    pub regions: Vec<Region>,
    pub portals: Vec<Portal>,
    /// Open shared boundaries, with no invented wall or door instance.
    #[serde(default)]
    pub open_connections: Vec<OpenConnection>,
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
                    door: p.door.clone(),
                });
            } else {
                result.boundary_partitions.push(i);
            }
        }
        for (i, a) in program.zones.iter().enumerate() {
            for (j, b) in program.zones.iter().enumerate().skip(i + 1) {
                for axis in 0..2 {
                    let coordinate = if (a.max[axis] - b.min[axis]).abs() < 1e-4 {
                        a.max[axis]
                    } else if (b.max[axis] - a.min[axis]).abs() < 1e-4 {
                        b.max[axis]
                    } else {
                        continue;
                    };
                    let other = 1 - axis;
                    let lo = a.min[other].max(b.min[other]);
                    let hi = a.max[other].min(b.max[other]);
                    if hi - lo < 0.8
                        || program
                            .partitions
                            .iter()
                            .any(|p| p.axis == axis && (p.coordinate - coordinate).abs() < 0.01)
                    {
                        continue;
                    }
                    let mut start = Vec2::ZERO;
                    let mut end = start;
                    start[axis] = coordinate;
                    end[axis] = coordinate;
                    start[other] = lo;
                    end[other] = hi;
                    // Clip against the same envelope, preserving openings only where
                    // there is actual common floor on both sides.
                    let spans = polygon::line_intervals(outline, axis, coordinate);
                    for (x, y) in spans {
                        let x = x.max(lo);
                        let y = y.min(hi);
                        if y - x > 0.8 {
                            start[other] = x;
                            end[other] = y;
                            result.open_connections.push(OpenConnection {
                                regions: [i, j],
                                start,
                                end,
                            });
                        }
                    }
                }
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
