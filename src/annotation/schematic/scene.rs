use super::*;
use crate::scene::procedural_indoor::{
    floorplan,
    layout::{IndoorManifest, ObjectKind},
};

pub(super) fn camera_path(scene: &IndoorManifest, index: usize) -> Vec<[f32; 3]> {
    let rotation = Quat::from_rotation_y(scene.world_yaw);
    scene.cameras.get(index).map_or_else(Vec::new, |c| {
        (0..=32)
            .map(|i| (rotation * c.transform_at(i as f32 / 32.).translation).to_array())
            .collect()
    })
}
fn rectangle(lo: Vec3, hi: Vec3) -> Vec<Vec3> {
    vec![
        lo,
        Vec3::new(hi.x, lo.y, lo.z),
        Vec3::new(hi.x, lo.y, hi.z),
        Vec3::new(lo.x, lo.y, hi.z),
    ]
}
impl Schematic {
    /// Scene-local plans become world-space using the manifest's current yaw.
    /// Planned cameras are convenient for CPU-only audits; `Sample::schematic`
    /// replaces them with recorded capture matrices and skeletons.
    pub fn from_manifest(scene: &IndoorManifest, progress: f32) -> Result<Self, String> {
        if !progress.is_finite() || !(0. ..=1.).contains(&progress) {
            return Err("schematic progress must be in 0..=1".into());
        }
        if !scene.world_yaw.is_finite()
            || !scene.room_size.is_finite()
            || scene.room_size.min_element() <= 0.
        {
            return Err("invalid schematic room extent".into());
        }
        if scene.envelope.as_ref().is_some_and(|e| {
            e.footprint.len() < 3 || e.walls.iter().any(|w| w.edge >= e.footprint.len())
        }) {
            return Err("invalid schematic envelope edges".into());
        }
        let rotation = Quat::from_rotation_y(scene.world_yaw);
        let mut footprints = vec![];
        let mut add = |label: String, id, kind, points: Vec<Vec3>| {
            footprints.push(Footprint {
                label,
                instance_id: id,
                kind,
                points: points
                    .into_iter()
                    .map(|p| (rotation * p).to_array())
                    .collect(),
            });
        };
        let half = scene.room_size * 0.5;
        let floor = scene.envelope.as_ref().map_or_else(
            || {
                rectangle(
                    Vec3::new(-half.x, 0., -half.z),
                    Vec3::new(half.x, 0., half.z),
                )
            },
            |e| {
                e.footprint
                    .iter()
                    .map(|p| Vec3::new(p.x, 0., p.y))
                    .collect()
            },
        );
        add(
            "Primary room".into(),
            None,
            FootprintKind::Floor,
            floor.clone(),
        );
        if let Some(e) = &scene.envelope {
            for p in &e.floor_patches {
                add(
                    format!("{:+.2} m", p.height),
                    None,
                    FootprintKind::Level,
                    rectangle(
                        Vec3::new(p.min.x, p.height, p.min.y),
                        Vec3::new(p.max.x, p.height, p.max.y),
                    ),
                );
            }
            if let Some(m) = &e.mezzanine {
                let p = &m.deck;
                add(
                    format!("Mezzanine +{:.2} m", p.height),
                    None,
                    FootprintKind::Mezzanine,
                    rectangle(
                        Vec3::new(p.min.x, p.height, p.min.y),
                        Vec3::new(p.max.x, p.height, p.max.y),
                    ),
                );
                for (lo, hi) in m.stair_boxes() {
                    add("Step".into(), None, FootprintKind::Level, rectangle(lo, hi));
                }
            }
            for p in &e.pillars {
                add(
                    "Column".into(),
                    None,
                    FootprintKind::Wall,
                    (0..p.sides.max(3))
                        .map(|i| {
                            let a = i as f32 / p.sides.max(3) as f32 * std::f32::consts::TAU;
                            Vec3::new(
                                p.center.x + p.radius * a.cos(),
                                0.,
                                p.center.y + p.radius * a.sin(),
                            )
                        })
                        .collect(),
                );
            }
        }
        // Draw perimeter as separate edges so openings can overwrite it cleanly.
        for i in 0..floor.len() {
            add(
                "Envelope".into(),
                None,
                FootprintKind::Wall,
                vec![floor[i], floor[(i + 1) % floor.len()]],
            );
        }
        if let Some(e) = &scene.envelope {
            for w in &e.walls {
                if let Some(f) = &w.facade {
                    let t = e.wall_transform(w.edge);
                    for opening in &f.openings {
                        add(
                            "Window".into(),
                            None,
                            FootprintKind::Window,
                            vec![
                                t.transform_point(Vec3::new(opening.min.x, opening.min.y, 0.)),
                                t.transform_point(Vec3::new(opening.max.x, opening.min.y, 0.)),
                            ],
                        );
                    }
                }
            }
            // The shared wall's openings use the same metric spans as construction.
            for (lo, hi) in [
                (-half.x + 0.10, scene.door_x - 0.60),
                (scene.door_x + 0.60, half.x - 0.10),
            ] {
                if hi - lo > 0.18 {
                    add(
                        "Interior glazing".into(),
                        None,
                        FootprintKind::Window,
                        vec![Vec3::new(lo, 0.05, half.z), Vec3::new(hi, 0.05, half.z)],
                    );
                }
            }
            // Shared wall has a glass facade with a primary-room doorway.
            add(
                "Door".into(),
                None,
                FootprintKind::Door,
                vec![
                    Vec3::new(scene.door_x - 0.51, 0., half.z),
                    Vec3::new(scene.door_x + 0.51, 0., half.z),
                ],
            );
        }
        for (lo, hi) in floorplan::obstacles(scene) {
            add(
                "Partition".into(),
                None,
                FootprintKind::Wall,
                rectangle(lo, hi),
            );
        }
        // Draw supported clutter after furniture, with stable IDs for calibration.
        let mut objects: Vec<_> = scene.objects.iter().filter(|o| !o.neighbor).collect();
        objects.sort_by(|a, b| a.position.y.total_cmp(&b.position.y).then(a.id.cmp(&b.id)));
        for o in objects {
            let points = rectangle(
                -o.size * Vec3::new(0.5, 0., 0.5),
                o.size * Vec3::new(0.5, 0., 0.5),
            )
            .into_iter()
            .map(|p| o.transform().transform_point(p))
            .collect();
            add(
                format!("{:?} {}", o.kind, o.id),
                Some(o.id),
                if o.kind == ObjectKind::Plant {
                    FootprintKind::Plant
                } else if o.solid {
                    FootprintKind::Furniture
                } else {
                    FootprintKind::Prop
                },
                points,
            );
        }
        let cameras = scene
            .cameras
            .iter()
            .enumerate()
            .map(|(i, c)| Camera {
                label: format!("C{i}"),
                world_from_view: (Mat4::from_quat(rotation) * c.transform_at(progress).to_matrix())
                    .to_cols_array_2d(),
                fov_y: c.fov_degrees.to_radians(),
                aspect: scene.camera_aspect_ratio,
                calibration: None,
                path: camera_path(scene, i),
            })
            .collect();
        let poses = scene
            .humans
            .iter()
            .map(|h| Pose {
                label: format!("P{}", h.id),
                joints: vec![(rotation * h.position).to_array()],
                parents: vec![-1],
            })
            .collect();
        Ok(Self {
            schema_version: 1,
            seed: scene.seed,
            trajectory_progress: progress,
            time_seconds: None,
            footprints,
            cameras,
            poses,
        })
    }
}
