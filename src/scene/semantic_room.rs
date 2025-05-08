use std::collections::HashMap;

use bevy::{
    prelude::*,
    render::render_resource::Face,
};
use rand::Rng;

use crate::{
    asset::WaitForAssets,
    camera::{
        ExtrinsicsSampler,
        ExtrinsicsSamplerType,
        LookingAtSampler,
        TrajectorySampler,
        ZeroverseCamera,
    },
    scene::{
        lighting::{
            setup_lighting,
            ZeroverseLightingSettings,
        },
        RegenerateSceneEvent,
        RotationAugment,
        SceneAabbNode,
        SceneLoadedEvent,
        ZeroverseScene,
        ZeroverseSceneRoot,
        ZeroverseSceneSettings,
        ZeroverseSceneType,
    },
    primitive::{
        CountSampler,
        PositionSampler,
        RotationSampler,
        ScaleSampler,
        ZeroversePrimitives,
        ZeroversePrimitiveSettings,
    },
    render::semantic::SemanticLabel,
};


pub struct ZeroverseSemanticRoomPlugin;
impl Plugin for ZeroverseSemanticRoomPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ZeroverseSemanticRoomSettings>();
        app.register_type::<ZeroverseSemanticRoomSettings>();

        app.add_systems(
            PreUpdate,
            regenerate_scene,
        );
    }
}


#[derive(Clone, Debug, Reflect, Resource)]
#[reflect(Resource)]
pub struct ZeroverseSemanticRoomSettings {
    pub camera_floor_padding: f32,
    pub camera_wall_padding: f32,
    pub chair_wall_padding: f32,
    pub human_wall_padding: f32,
    pub table_wall_padding: f32,
    pub looking_at_sampler: LookingAtSampler,
    pub room_size: ScaleSampler,
    pub chair_count: CountSampler,
    pub chair_settings: ZeroversePrimitiveSettings,
    pub door_settings: ZeroversePrimitiveSettings,
    pub human_count: CountSampler,
    pub human_settings: ZeroversePrimitiveSettings,  // TODO: support parametric SMPL morphing
    pub table_settings: ZeroversePrimitiveSettings,  // TODO: table scale relative to room scale
    pub window_probability: f32,
    pub window_size_min: Vec2,
    pub window_size_max: Vec2,
    pub neighborhood_depth: usize,
}

impl Default for ZeroverseSemanticRoomSettings {
    fn default() -> Self {
        Self {
            camera_floor_padding: 3.0,
            camera_wall_padding: 1.0,
            chair_wall_padding: 0.5,
            human_wall_padding: 0.25,
            table_wall_padding: 2.0,
            looking_at_sampler: LookingAtSampler::Sphere {
                geometry: Sphere::new(4.0),
                transform: Transform::from_translation(Vec3::new(0.0, 3.0, 0.0)),
            },
            room_size: ScaleSampler::Bounded(
                Vec3::new(12.0, 8.0, 12.0),
                Vec3::new(35.0, 15.0, 35.0),
            ),
            chair_count: CountSampler::Bounded(3, 8),
            chair_settings: ZeroversePrimitiveSettings {
                cull_mode: Some(Face::Back),
                available_types: vec![ZeroversePrimitives::Mesh("chair".into())],
                components: CountSampler::Exact(1),
                wireframe_probability: 0.0,
                noise_probability: 0.0,
                cast_shadows: false,
                rotation_sampler: RotationSampler::Bounded {
                    min: Vec3::new(0.0, 0.0, 0.0),
                    max: Vec3::new(0.0, std::f32::consts::PI, 0.0),
                },
                scale_sampler: ScaleSampler::Bounded(
                    Vec3::new(2.0, 3.0, 2.0),
                    Vec3::new(4.0, 5.0, 4.0),
                ),
                smooth_normals_probability: 0.0,
                ..default()
            },
            door_settings: ZeroversePrimitiveSettings {
                cull_mode: Some(Face::Back),
                available_types: vec![ZeroversePrimitives::Mesh("door".into())],
                components: CountSampler::Exact(1),
                wireframe_probability: 0.0,
                noise_probability: 0.0,
                cast_shadows: false,
                rotation_sampler: RotationSampler::Identity,
                scale_sampler: ScaleSampler::Bounded(
                    Vec3::new(3.5, 7.0, 0.0001),
                    Vec3::new(4.5, 12.0, 0.001),
                ),
                smooth_normals_probability: 0.0,
                ..default()
            },
            human_count: CountSampler::Bounded(1, 3),
            human_settings: ZeroversePrimitiveSettings {
                cull_mode: Some(Face::Back),
                available_types: vec![ZeroversePrimitives::Mesh("human".into())],
                components: CountSampler::Exact(1),
                wireframe_probability: 0.0,
                noise_probability: 0.0,
                cast_shadows: false,
                rotation_sampler: RotationSampler::Bounded {
                    min: Vec3::new(0.0, 0.0, 0.0),
                    max: Vec3::new(0.0, std::f32::consts::PI, 0.0),
                },
                scale_sampler: ScaleSampler::Bounded(
                    Vec3::new(0.8, 0.8, 0.8),
                    Vec3::new(1.2, 1.2, 1.2),
                ),
                smooth_normals_probability: 0.0,
                ..default()
            },
            table_settings: ZeroversePrimitiveSettings {
                cull_mode: Some(Face::Back),
                available_types: vec![ZeroversePrimitives::Mesh("table".into())],
                components: CountSampler::Exact(1),
                wireframe_probability: 0.0,
                noise_probability: 0.0,
                cast_shadows: false,
                rotation_sampler: RotationSampler::Identity,
                scale_sampler: ScaleSampler::Bounded(
                    Vec3::new(4.0, 2.0, 4.0),
                    Vec3::new(12.0, 4.0, 12.0),
                ),
                smooth_normals_probability: 0.0,
                ..default()
            },
            window_probability: 0.6,
            window_size_min: Vec2::new(0.25, 0.25),
            window_size_max: Vec2::new(0.75, 0.75),
            neighborhood_depth: 2,
        }
    }
}


fn check_aabb_collision(
    center: Vec3,
    scale: Vec3,
    aabb_colliders: &[(Vec3, Vec3)],
) -> bool {
    let half_scale: Vec3 = scale * 0.3;
    let min_a = center - half_scale;
    let max_a = center + half_scale;

    for (other_center, other_scale) in aabb_colliders.iter() {
        let half_other_scale = *other_scale * 0.25;
        let min_b = *other_center - half_other_scale;
        let max_b = *other_center + half_other_scale;

        if max_a.x < min_b.x || min_a.x > max_b.x {
            continue;
        }

        if max_a.y < min_b.y || min_a.y > max_b.y {
            continue;
        }

        if max_a.z < min_b.z || min_a.z > max_b.z {
            continue;
        }

        return true;
    }

    false
}


const CLEARANCE: f32 = 1e-4;

#[inline]
fn depth_scaled_prob(base: f32, depth: i32) -> f32 {
    base * 0.5_f32.powi(depth)
}

#[allow(clippy::too_many_arguments)]
fn spawn_face(
    commands: &mut ChildSpawnerCommands,
    room_settings: &ZeroverseSemanticRoomSettings,
    origin: Vec3,
    basis: Quat,
    half_extents: Vec3,
    cull_mode: Option<Face>,
    invert_normals: bool,
    label: SemanticLabel,
    name: &str,
    window: bool,
    depth: i32,
) -> bool {
    let mut rng = rand::thread_rng();

    let base = ZeroversePrimitiveSettings {
        cull_mode,
        invert_normals,
        available_types: vec![ZeroversePrimitives::Plane],
        components: CountSampler::Exact(1),
        wireframe_probability: 0.0,
        noise_probability: 0.0,
        cast_shadows: false,
        rotation_sampler: RotationSampler::Exact(basis),
        ..default()
    };

    let mut spawn_segment = |pos: Vec3, half_size: Vec3, suffix: &str| {
        commands.spawn((
            ZeroversePrimitiveSettings {
                position_sampler: PositionSampler::Exact { position: origin + basis * pos },
                scale_sampler: ScaleSampler::Exact(half_size),
                ..base.clone()
            },
            Name::new(format!("{name}{suffix}")),
            label.clone(),
        ));
    };

    let depth_scaled_window_prob = depth_scaled_prob(room_settings.window_probability, depth);
    let make_window = window && rng.gen_bool(depth_scaled_window_prob as f64);

    if !make_window {
        spawn_segment(Vec3::ZERO, half_extents, "");
        return false;
    }


    let hx = half_extents.x;
    let hz = half_extents.z;

    let max_w_frac = 1.0 - CLEARANCE / hx;
    let max_h_frac = 1.0 - CLEARANCE / hz;

    let min_w_frac = room_settings.window_size_min.x.clamp(0.0, max_w_frac);
    let max_w_frac = room_settings.window_size_max.x.clamp(min_w_frac, max_w_frac);
    let min_h_frac = room_settings.window_size_min.y.clamp(0.0, max_h_frac);
    let max_h_frac = room_settings.window_size_max.y.clamp(min_h_frac, max_h_frac);

    let w_frac = rng.gen_range(min_w_frac..=max_w_frac);
    let h_frac = rng.gen_range(min_h_frac..=max_h_frac);

    let half_win_w = hx * w_frac;
    let half_win_h = hz * h_frac;

    let win_cx = rng.gen_range((-hx + half_win_w + CLEARANCE)..=(hx - half_win_w - CLEARANCE));
    let win_cz = rng.gen_range((-hz + half_win_h + CLEARANCE)..=(hz - half_win_h - CLEARANCE));


    let top_half_h = (hz - (win_cz + half_win_h)) / 2.0;
    spawn_segment(
        Vec3::new(0.0, 0.0, win_cz + half_win_h + top_half_h),
        Vec3::new(hx, half_extents.y, top_half_h),
        "_top",
    );

    let bottom_half_h = (win_cz - half_win_h + hz) / 2.0;
    spawn_segment(
        Vec3::new(0.0, 0.0, win_cz - half_win_h - bottom_half_h),
        Vec3::new(hx, half_extents.y, bottom_half_h),
        "_bottom",
    );


    let left_half_w = (win_cx - half_win_w + hx) / 2.0;
    spawn_segment(
        Vec3::new(-hx + left_half_w, 0.0, win_cz),
        Vec3::new(left_half_w, half_extents.y, half_win_h),
        "_left",
    );

    let right_half_w = (hx - (win_cx + half_win_w)) / 2.0;
    spawn_segment(
        Vec3::new(hx - right_half_w, 0.0, win_cz),
        Vec3::new(right_half_w, half_extents.y, half_win_h),
        "_right",
    );

    true
}


// TODO: support bailing on room spawn if required room features are too large (e.g. no door)
fn spawn_room(
    commands: &mut ChildSpawnerCommands,
    room_scale: &Vec3,
    room_settings: &ZeroverseSemanticRoomSettings,
    depth: i32,
    leaf: bool,
) -> [bool; 4] {
    let mut windows = [false; 4];
    let mut rng = rand::thread_rng();

    {// outer walls
        let hx = room_scale.x / 2.0;
        let hy = room_scale.y;
        let hz = room_scale.z / 2.0;

        let ceiling_floor = [
            (Vec3::new(0.0, hy, 0.0), Quat::IDENTITY, Vec3::new(hx, 1.0, hz), Some(Face::Front), true, SemanticLabel::Ceiling, "top", false),
            (Vec3::new(0.0, 0.0, 0.0), Quat::IDENTITY, Vec3::new(hx, 1.0, hz), Some(Face::Back), false, SemanticLabel::Floor, "bottom", false),
        ];

        ceiling_floor.iter().for_each(|(
            origin,
            basis,
            half,
            cull,
            invert,
            label,
            name,
            wall,
        )| {
            spawn_face(
                commands,
                room_settings,
                *origin,
                *basis,
                *half,
                *cull,
                *invert,
                label.clone(),
                name,
                *wall && !leaf,
                depth,
            );
        });

        let walls = [
            (Vec3::new(0.0, hy / 2.0, hz), Quat::from_rotation_x(90_f32.to_radians()), Vec3::new(hx, 1.0, hy / 2.0), Some(Face::Front), true, SemanticLabel::Wall, "front", true),
            (Vec3::new(0.0, hy / 2.0, -hz),Quat::from_rotation_x(90_f32.to_radians()), Vec3::new(hx, 1.0, hy / 2.0), Some(Face::Back), false, SemanticLabel::Wall, "back", true),
            (Vec3::new(-hx, hy / 2.0, 0.0),Quat::from_rotation_z(90_f32.to_radians()), Vec3::new(hy / 2.0, 1.0, hz), Some(Face::Front), true, SemanticLabel::Wall, "left", true),
            (Vec3::new(hx, hy / 2.0, 0.0), Quat::from_rotation_z(90_f32.to_radians()), Vec3::new(hy / 2.0, 1.0, hz), Some(Face::Back), false, SemanticLabel::Wall, "right", true),
        ];

        walls.iter().enumerate().for_each(|(i, (
            origin,
            basis,
            half,
            cull,
            invert,
            label,
            name,
            wall,
        ))| {
            // TODO: if spawning adjacent room, use the same wall
            let has_window = spawn_face(
                commands,
                room_settings,
                *origin,
                *basis,
                *half,
                *cull,
                *invert,
                label.clone(),
                name,
                *wall,
                depth,
            );
            windows[i] = has_window;
        });
    }

    let mut aabb_colliders: Vec<(Vec3, Vec3)> = Vec::new();

    { // table
        let mut table_scale = room_settings.table_settings.scale_sampler.sample();

        let room_half_x = room_scale.x / 2.0;
        let room_half_z = room_scale.z / 2.0;

        let max_half_table_scale_x = room_half_x - room_settings.table_wall_padding;
        let max_half_table_scale_z = room_half_z - room_settings.table_wall_padding;

        let max_half_table_scale_x = if max_half_table_scale_x > 0.0 {
            max_half_table_scale_x
        } else {
            0.1
        };

        let max_half_table_scale_z = if max_half_table_scale_z > 0.0 {
            max_half_table_scale_z
        } else {
            0.1
        };

        table_scale.x = table_scale.x.min(max_half_table_scale_x * 2.0);
        table_scale.z = table_scale.z.min(max_half_table_scale_z * 2.0);

        let half_table_scale = table_scale * 0.5;
        let height_offset = Vec3::new(
            0.0,
            table_scale.y / 4.0,
            0.0,
        );

        let center_sampler = PositionSampler::Cube {
            extents: Vec3::new(
                room_half_x - room_settings.table_wall_padding - half_table_scale.x,
                0.00001,
                room_half_z - room_settings.table_wall_padding - half_table_scale.z,
            ),
        };

        if center_sampler.is_valid() {
            let position = center_sampler.sample() + height_offset;

            aabb_colliders.push((position, table_scale));

            commands.spawn((
                ZeroversePrimitiveSettings {
                    position_sampler: PositionSampler::Exact {
                        position,
                    },
                    scale_sampler: ScaleSampler::Exact(table_scale),
                    rotation_sampler: RotationSampler::Identity,
                    ..room_settings.table_settings.clone()
                },
                Transform::from_translation(position),
                Name::new("table"),
                SemanticLabel::Table,
            ));
        }
    }

    { // chairs
        let chair_scale = room_settings.chair_settings.scale_sampler.sample();
        let chair_scale = Vec3::new(chair_scale.x, chair_scale.y, chair_scale.x);
        let center_sampler = PositionSampler::Cube {
            extents: Vec3::new(
                room_scale.x / 2.0 - room_settings.chair_wall_padding - chair_scale.x / 2.0,
                0.00001,
                room_scale.z / 2.0 - room_settings.chair_wall_padding - chair_scale.z / 2.0,
            ),
        };
        let chair_scale_sampler = ScaleSampler::Exact(chair_scale);

        let height_offset = Vec3::new(
            0.0,
            chair_scale.y / 4.0,
            0.0,
        );

        if center_sampler.is_valid() {
            for _ in 0..room_settings.chair_count.sample() {
                let mut position = center_sampler.sample() + height_offset;

                let mut max_attempts = 100;
                while check_aabb_collision(position, chair_scale, &aabb_colliders) && max_attempts > 0 {
                    position = center_sampler.sample() + height_offset;
                    max_attempts -= 1;
                }

                if max_attempts == 0 {
                    continue;
                }

                aabb_colliders.push((position, chair_scale));

                commands.spawn((
                    ZeroversePrimitiveSettings {
                        position_sampler: PositionSampler::Exact {
                            position,
                        },
                        scale_sampler: chair_scale_sampler.clone(),
                        // rotation_sampler: RotationSampler::Identity,
                        ..room_settings.chair_settings.clone()
                    },
                    Transform::from_translation(position),
                    Name::new("chair"),
                    SemanticLabel::Chair,
                ));
            }
        }
    }

    // TODO: tv, whiteboard, door, rug
    { // door
        let face = rng.gen_range(0..4);
        let mut door_scale = room_settings.door_settings.scale_sampler.sample();

        let hw = door_scale.x / 2.0;
        let (x_offset, z_offset) = match face {
            0 => (0.0, hw),
            1 => (0.0, hw),
            2 => (hw, 0.0),
            3 => (hw, 0.0),
            _ => unreachable!(),
        };

        let perimeter = Vec3::new(
            room_scale.x / 2.0 - 0.001 - x_offset,
            0.0,
            room_scale.z / 2.0 - 0.001 - z_offset,
        );

        let perimeter_is_valid = perimeter.x > 0.0 && perimeter.z > 0.0;
        if perimeter_is_valid {
            let (x, z) = match face {
                0 => (-perimeter.x / 2.0, rng.gen_range(-perimeter.z / 2.0..perimeter.z / 2.0)),
                1 => (perimeter.x / 2.0, rng.gen_range(-perimeter.z / 2.0..perimeter.z / 2.0)),
                2 => (rng.gen_range(-perimeter.x / 2.0..perimeter.x / 2.0), -perimeter.z / 2.0),
                3 => (rng.gen_range(-perimeter.x / 2.0..perimeter.x / 2.0), perimeter.z / 2.0),
                _ => unreachable!(),
            };
            door_scale.y = door_scale.y.min(room_scale.y);
            let door_position = Vec3::new(x, door_scale.y / 4.0, z);
            let door_rotation = match face {
                0 => Quat::from_rotation_y(90.0_f32.to_radians()),
                1 => Quat::from_rotation_y(-90.0_f32.to_radians()),
                2 => Quat::from_rotation_y(180.0_f32.to_radians()),
                3 => Quat::IDENTITY,
                _ => unreachable!(),
            };

            commands.spawn((
                ZeroversePrimitiveSettings {
                    position_sampler: PositionSampler::Exact {
                        position: door_position,
                    },
                    scale_sampler: ScaleSampler::Exact(door_scale),
                    rotation_sampler: RotationSampler::Exact(door_rotation),
                    ..room_settings.door_settings.clone()
                },
                Transform::from_translation(door_position),
                Name::new("door"),
                SemanticLabel::Door,
            ));
        }
    }

    { // humans
        for _ in 0..room_settings.human_count.sample() {
            let human_scale = room_settings.human_settings.scale_sampler.sample();
            let human_scale = Vec3::new(human_scale.x, human_scale.y, human_scale.x);
            let center_sampler = PositionSampler::Cube {
                extents: Vec3::new(
                    room_scale.x / 2.0 - room_settings.human_wall_padding - human_scale.x / 2.0,
                    0.00001,
                    room_scale.z / 2.0 - room_settings.human_wall_padding - human_scale.z / 2.0,
                ),
            };

            if !center_sampler.is_valid() {
                continue;
            }

            let human_scale_sampler = ScaleSampler::Exact(human_scale);

            let height_offset = Vec3::new(
                0.0,
                0.0,
                0.0,
            );

            let mut position = center_sampler.sample() + height_offset;

            let mut max_attempts = 100;
            while check_aabb_collision(position, human_scale, &aabb_colliders) && max_attempts > 0 {
                position = center_sampler.sample() + height_offset;
                max_attempts -= 1;
            }

            if max_attempts == 0 {
                continue;
            }

            aabb_colliders.push((position, human_scale));

            commands.spawn((
                ZeroversePrimitiveSettings {
                    position_sampler: PositionSampler::Exact {
                        position,
                    },
                    scale_sampler: human_scale_sampler.clone(),
                    ..room_settings.human_settings.clone()
                },
                Transform::from_translation(position),
                Name::new("human"),
                SemanticLabel::Human,
            ));
        }
    }

    windows
}


const DIRS: [(i32, i32); 4] = [(0, 1), (0, -1), (-1, 0), (1, 0)];
const MAX_SHRINK_ITERS: usize = 8;
const MIN_FRACTION: f32 = 0.20;
const ROOM_GAP: f32 = 0.05;

#[inline]
fn overlaps(a_pos: Vec3, a_scale: Vec3, b_pos: Vec3, b_scale: Vec3) -> bool {
    let dx = (a_pos.x - b_pos.x).abs();
    let dz = (a_pos.z - b_pos.z).abs();
    dx + ROOM_GAP < (a_scale.x + b_scale.x) * 0.5 &&
    dz + ROOM_GAP < (a_scale.z + b_scale.z) * 0.5
}

#[allow(clippy::too_many_arguments)]
fn spawn_room_rec(
    commands: &mut ChildSpawnerCommands,
    coord: (i32, i32),
    parent_coord: (i32, i32),
    dir: (i32, i32),
    depth: i32,
    depth_left: i32,
    base_scale: Vec3,
    rooms: &mut HashMap<(i32, i32), (Vec3, Vec3)>,
    settings: &ZeroverseSemanticRoomSettings,
) {
    let mut my_scale = settings.room_size.sample();
    my_scale.y = base_scale.y;

    let update_pos = |scale: Vec3| {
        let (parent_pos, parent_scale) = rooms[&parent_coord];
        let mut p = parent_pos;

        if dir.0 != 0 {
            p.x += dir.0 as f32 * ((parent_scale.x + scale.x) * 0.5 + ROOM_GAP);
        }
        if dir.1 != 0 {
            p.z += dir.1 as f32 * ((parent_scale.z + scale.z) * 0.5 + ROOM_GAP);
        }
        p
    };
    let mut my_pos = update_pos(my_scale);

    for _ in 0..MAX_SHRINK_ITERS {
        let mut shrink_x: f32 = 0.0;
        let mut shrink_z: f32 = 0.0;

        for &(other_pos, other_scale) in rooms.values() {
            if overlaps(my_pos, my_scale, other_pos, other_scale) {
                let dx = (my_pos.x - other_pos.x).abs();
                let dz = (my_pos.z - other_pos.z).abs();
                shrink_x = shrink_x.max((my_scale.x + other_scale.x) * 0.5 - dx + ROOM_GAP);
                shrink_z = shrink_z.max((my_scale.z + other_scale.z) * 0.5 - dz + ROOM_GAP);
            }
        }

        if shrink_x == 0.0 && shrink_z == 0.0 {
            break;
        }

        if shrink_x > 0.0 {
            my_scale.x = (my_scale.x - shrink_x).max(base_scale.x * MIN_FRACTION);
        }
        if shrink_z > 0.0 {
            my_scale.z = (my_scale.z - shrink_z).max(base_scale.z * MIN_FRACTION);
        }

        my_pos = update_pos(my_scale);
    }

    rooms.insert(coord, (my_pos, my_scale));

    let mut windows = [false; 4];
    commands
        .spawn((
            Name::new(format!("room_{}_{}", coord.0, coord.1)),
            Transform::from_translation(my_pos),
            InheritedVisibility::default(),
            Visibility::default(),
        ))
        .with_children(|c| {
            windows = spawn_room(
                c,
                &my_scale,
                settings,
                depth,
                depth_left == 0,
            );
        });

    if depth_left > 0 {
        for (idx, &(sx, sz)) in DIRS.iter().enumerate() {
            if !windows[idx] { continue }
            let next = (coord.0 + sx, coord.1 + sz);
            if !rooms.contains_key(&next) {
                spawn_room_rec(
                    commands,
                    next,
                    coord,
                    (sx, sz),
                    depth + 1,
                    depth_left - 1,
                    base_scale,
                    rooms,
                    settings,
                );
            }
        }
    }
}

// TODO: support similar windows across adjacent room faces
fn spawn_room_neighborhood(
    commands: &mut ChildSpawnerCommands,
    base_scale: &Vec3,
    settings: &ZeroverseSemanticRoomSettings,
) {
    commands
        .spawn((
            InheritedVisibility::default(),
            Name::new("room"),
            SceneAabbNode,
            Transform::default(),
            Visibility::default(),
        ))
        .with_children(|root| {
            let windows_root = spawn_room(
                root,
                base_scale,
                settings,
                0,
                false,
            );

            let mut rooms: HashMap<(i32, i32), (Vec3, Vec3)> = HashMap::new();
            rooms.insert((0, 0), (Vec3::ZERO, *base_scale));

            for (idx, &(dx, dz)) in DIRS.iter().enumerate() {
                if !windows_root[idx] { continue; }
                spawn_room_rec(
                    root,
                    (dx, dz),
                    (0, 0),
                    (dx, dz),
                    1,
                    settings.neighborhood_depth as i32 - 1,
                    *base_scale,
                    &mut rooms,
                    settings,
                );
            }
        });
}


fn setup_scene(
    mut commands: Commands,
    mut load_event: EventWriter<SceneLoadedEvent>,
    room_settings: Res<ZeroverseSemanticRoomSettings>,
    scene_settings: Res<ZeroverseSceneSettings>,
) {
    let room_scale = room_settings.room_size.sample();

    // TODO: set global Y rotation to prevent wall aligned plucker embeddings

    commands.spawn((
        Name::new("rooms"),
        RotationAugment,
        ZeroverseSceneRoot,
        ZeroverseScene,
    ))
    .with_children(|commands| {
        spawn_room_neighborhood(
            commands,
            &room_scale,
            &room_settings,
        );

        { // cameras
            let size: Vec3 = Vec3::new(
                room_scale.x - (room_settings.camera_wall_padding + scene_settings.max_camera_radius) * 2.0,
                room_scale.y - (room_settings.camera_floor_padding + scene_settings.max_camera_radius),
                room_scale.z - (room_settings.camera_wall_padding + scene_settings.max_camera_radius) * 2.0,
            );
            let origin_camera_sampler = ExtrinsicsSampler {
                position: ExtrinsicsSamplerType::Band {
                    size,
                    rotation: Quat::IDENTITY,
                    translate: Vec3::new(
                        0.0,
                        room_settings.camera_floor_padding + size.y / 2.0,
                        0.0,
                    ),
                },
                looking_at: room_settings.looking_at_sampler.clone(),
                ..default()
            };
            let origin_camera_center = origin_camera_sampler.sample();
            let mut rng = rand::thread_rng();

            for _ in 0..scene_settings.num_cameras {
                if scene_settings.max_camera_radius <= 0.0 {
                    let size: Vec3 = Vec3::new(
                        room_scale.x - room_settings.camera_wall_padding * 2.0,
                        room_scale.y - room_settings.camera_floor_padding,
                        room_scale.z - room_settings.camera_wall_padding * 2.0,
                    );

                    let camera_sampler = ExtrinsicsSampler {
                        position: ExtrinsicsSamplerType::Band {
                            size,
                            rotation: Quat::IDENTITY,
                            translate: Vec3::new(
                                0.0,
                                room_settings.camera_floor_padding + size.y / 2.0,
                                0.0,
                            ),
                        },
                        looking_at: room_settings.looking_at_sampler.clone(),
                        ..default()
                    };

                    commands.spawn(ZeroverseCamera {
                        trajectory: TrajectorySampler::Linear {
                            start: camera_sampler.clone(),
                            end: camera_sampler,
                        },
                        ..default()
                    });
                } else {
                    let circular_sampler = ExtrinsicsSampler {
                        position: ExtrinsicsSamplerType::Circle {
                            radius: scene_settings.max_camera_radius,
                            rotation: Quat::from_rng(&mut rng),
                            translate: origin_camera_center.translation,
                        },
                        looking_at: room_settings.looking_at_sampler.clone(),
                        ..default()
                    };

                    commands.spawn(ZeroverseCamera {
                        trajectory: TrajectorySampler::Static {
                            start: circular_sampler,
                        },
                        ..default()
                    });
                }
            }
        }
    });

    load_event.write(SceneLoadedEvent);
}


#[allow(clippy::too_many_arguments)]
fn regenerate_scene(
    mut commands: Commands,
    room_settings: Res<ZeroverseSemanticRoomSettings>,
    clear_zeroverse_scenes: Query<Entity, With<ZeroverseScene>>,
    mut regenerate_events: EventReader<RegenerateSceneEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
    load_event: EventWriter<SceneLoadedEvent>,
    lighting_settings: Res<ZeroverseLightingSettings>,
    wait_for: Res<WaitForAssets>,
    mut recover_from_wait: Local<bool>,
) {
    if scene_settings.scene_type != ZeroverseSceneType::SemanticRoom {
        return;
    }

    if regenerate_events.is_empty() && !*recover_from_wait {
        return;
    }
    regenerate_events.clear();

    if wait_for.is_waiting() {
        // TODO: send out a regenerate event when asset wait is complete (and the current scene is waiting) (works better across scenes)
        *recover_from_wait = true;
        return;
    }
    *recover_from_wait = false;

    for entity in clear_zeroverse_scenes.iter() {
        commands.entity(entity).despawn();
    }

    setup_lighting(
        commands.reborrow(),
        lighting_settings,
    );

    setup_scene(
        commands,
        load_event,
        room_settings,
        scene_settings,
    );
}
