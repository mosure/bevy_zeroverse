//! Position, look-target and rectilinear-intrinsic samplers. Kept independent
//! of ECS camera installation and nonlinear source-view rendering.
use bevy::{
    prelude::*,
    shape::{Circle, ShapeSample, Sphere},
};
use rand::Rng;

#[derive(Clone, Debug, Reflect)]
#[type_path = "bevy_zeroverse::camera"]
pub enum LookingAtSampler {
    Exact(Vec3),
    Sphere {
        geometry: Sphere,
        transform: Transform,
    },
}

impl Default for LookingAtSampler {
    fn default() -> Self {
        Self::Exact(Vec3::ZERO)
    }
}

impl LookingAtSampler {
    pub fn sample(&self) -> Vec3 {
        match *self {
            LookingAtSampler::Exact(pos) => pos,
            LookingAtSampler::Sphere {
                geometry,
                transform,
            } => transform * geometry.sample_interior(&mut rand_bevy::rng()),
        }
    }

    pub fn look_at(&self, mut transform: Transform) -> Transform {
        transform.look_at(self.sample(), Vec3::Y);
        transform
    }
}

// TODO: add gizmos for sampler types
#[derive(Clone, Debug, Reflect)]
#[type_path = "bevy_zeroverse::camera"]
pub enum ExtrinsicsSamplerType {
    Band {
        size: Vec3,
        rotation: Quat,
        translate: Vec3,
    },
    BandShell {
        inner_size: Vec3,
        outer_size: Vec3,
        rotation: Quat,
        translate: Vec3,
    },
    Circle {
        radius: f32,
        rotation: Quat,
        translate: Vec3,
    },
    Sphere {
        radius: f32,
        translate: Vec3,
    },
    SphereShell {
        inner_radius: f32,
        outer_radius: f32,
        translate: Vec3,
    },
    Cuboid {
        size: Vec3,
        rotation: Quat,
    },
    Ellipsoid {
        radius: Vec3,
        rotation: Quat,
    },
    Transform(Transform),
}

#[derive(Clone, Debug, Reflect, Default)]
#[type_path = "bevy_zeroverse::camera"]
pub struct ExtrinsicsSampler {
    pub cache: Option<Transform>,
    pub looking_at: LookingAtSampler,
    pub position: ExtrinsicsSamplerType, // TODO: use position sampler here
}

impl Default for ExtrinsicsSamplerType {
    fn default() -> Self {
        Self::Transform(Transform::default())
    }
}

impl ExtrinsicsSampler {
    pub fn sample_cache(&mut self) -> Transform {
        if let Some(transform) = self.cache {
            return transform;
        }

        let transform = self.sample();
        self.cache = Some(transform);
        transform
    }

    pub fn sample(&self) -> Transform {
        let transform: Transform = match self.position {
            ExtrinsicsSamplerType::Band {
                size,
                rotation,
                translate,
            } => {
                let mut rng = rand::rng();

                let face = rng.random_range(0..4);

                let (x, z) = match face {
                    0 => (-size.x / 2.0, rng.random_range(-size.z / 2.0..size.z / 2.0)),
                    1 => (size.x / 2.0, rng.random_range(-size.z / 2.0..size.z / 2.0)),
                    2 => (rng.random_range(-size.x / 2.0..size.x / 2.0), -size.z / 2.0),
                    3 => (rng.random_range(-size.x / 2.0..size.x / 2.0), size.z / 2.0),
                    _ => unreachable!(),
                };

                let y = rng.random_range(-size.y / 2.0..size.y / 2.0);
                let pos = Vec3::new(x, y, z) + translate;
                let pos = rotation.mul_vec3(pos);
                Transform::from_translation(pos)
            }
            ExtrinsicsSamplerType::BandShell {
                inner_size,
                outer_size,
                rotation,
                translate,
            } => {
                let mut rng = rand::rng();

                let face = rng.random_range(0..4);

                let (x, z) = match face {
                    0 => (
                        if rng.random_bool(0.5) {
                            outer_size.x / 2.0
                        } else {
                            -outer_size.x / 2.0
                        },
                        rng.random_range(-outer_size.z / 2.0..outer_size.z / 2.0),
                    ),
                    1 => (
                        if rng.random_bool(0.5) {
                            inner_size.x / 2.0
                        } else {
                            -inner_size.x / 2.0
                        },
                        rng.random_range(-inner_size.z / 2.0..inner_size.z / 2.0),
                    ),
                    2 => (
                        rng.random_range(-outer_size.x / 2.0..outer_size.x / 2.0),
                        if rng.random_bool(0.5) {
                            outer_size.z / 2.0
                        } else {
                            -outer_size.z / 2.0
                        },
                    ),
                    3 => (
                        rng.random_range(-inner_size.x / 2.0..inner_size.x / 2.0),
                        if rng.random_bool(0.5) {
                            inner_size.z / 2.0
                        } else {
                            -inner_size.z / 2.0
                        },
                    ),
                    _ => unreachable!(),
                };

                let y = rng.random_range(-outer_size.y / 2.0..outer_size.y / 2.0);
                let pos = Vec3::new(x, y, z) + translate;
                let pos = rotation.mul_vec3(pos);
                Transform::from_translation(pos)
            }
            ExtrinsicsSamplerType::Circle {
                radius,
                rotation,
                translate,
            } => {
                let xz = Circle::new(radius).sample_boundary(&mut rand_bevy::rng());
                let pos = rotation.mul_vec3(Vec3::new(xz.x, 0.0, xz.y)) + translate;
                Transform::from_translation(pos)
            }
            ExtrinsicsSamplerType::Sphere { radius, translate } => {
                let pos = Sphere::new(radius).sample_boundary(&mut rand_bevy::rng());
                Transform::from_translation(pos + translate)
            }
            ExtrinsicsSamplerType::SphereShell {
                inner_radius,
                outer_radius,
                translate,
            } => {
                let mut rng = rand::rng();

                let radius = rng
                    .random_range(inner_radius.powi(3)..outer_radius.powi(3))
                    .powf(1.0 / 3.0);
                let direction = Vec3::new(
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                )
                .normalize();

                let pos = direction * radius;
                Transform::from_translation(pos + translate)
            }
            ExtrinsicsSamplerType::Transform(transform) => transform,
            _ => Transform::default(),
        };

        self.looking_at.look_at(transform)
    }
}

#[derive(Component, Debug, Reflect)]
#[type_path = "bevy_zeroverse::camera"]
pub struct PerspectiveSampler {
    pub min_fov_deg: f32,
    pub max_fov_deg: f32,
}

impl Default for PerspectiveSampler {
    fn default() -> Self {
        Self {
            min_fov_deg: 20.0,
            max_fov_deg: 90.0,
        }
    }
}

impl PerspectiveSampler {
    pub fn sample(&self) -> PerspectiveProjection {
        let fov_deg = if self.min_fov_deg == self.max_fov_deg {
            self.min_fov_deg
        } else {
            let mut rng = rand::rng();
            rng.random_range(self.min_fov_deg..self.max_fov_deg)
        };

        let fov = fov_deg * std::f32::consts::PI / 180.0;
        PerspectiveProjection {
            fov,
            near: 0.1,
            far: 50.0,
            ..default()
        }
    }

    pub fn exact(fov_deg: f32) -> Self {
        Self {
            min_fov_deg: fov_deg,
            max_fov_deg: fov_deg,
        }
    }
}
