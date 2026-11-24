//! Fit a cabinet and its physical yoke to the footprint already cleared by layout.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum MountKind {
    Desktop,
    FlushWall,
    WallYoke,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Mount {
    pub kind: MountKind,
    pub pitch_radians: f32,
    pub center: Vec3,
    pub rear_plane: f32,
    pub strap: f32,
    pub bearing_gap: f32,
}

impl Mount {
    pub(super) fn fit(o: &IndoorObject, raw: Vec3) -> (Self, Vec3) {
        use crate::scene::procedural_indoor::layout::equipment::services::is_wall_service;
        let wall = is_wall_service(o);
        let mut mount = Self {
            kind: if wall {
                MountKind::FlushWall
            } else {
                MountKind::Desktop
            },
            pitch_radians: 0.,
            center: Vec3::Y * (raw.y * 0.5),
            rear_plane: -0.5 * raw.z,
            strap: 0.,
            bearing_gap: 0.,
        };
        if !wall {
            return (mount, raw);
        }
        let mut design = stream(o.seed, u64::from_le_bytes(*b"SpkrYoke"));
        if !design.random_bool(if raw.x > 1.6 * raw.y { 0.58 } else { 0.84 }) {
            return (mount, raw);
        }
        mount.kind = MountKind::WallYoke;
        mount.strap = (raw.min_element() * design.random_range(0.024..0.042)).clamp(0.0015, 0.0055);
        mount.bearing_gap = mount.strap * design.random_range(0.65..1.1);
        let back_gap = (raw.z * design.random_range(0.075..0.15)).max(mount.strap * 1.6);
        let depth = raw.z * design.random_range(0.64..0.82);
        let span = o.size.z * 0.5 - mount.rear_plane - back_gap;
        // Individual brackets on a matched installation aim at different listener positions.
        let mut install = stream(
            o.seed ^ (o.id as u64).rotate_left(29),
            u64::from_le_bytes(*b"SpkrAim "),
        );
        let listener_y = install.random_range(1.04..1.55);
        let distance = install.random_range(2.0..5.6);
        let aim = (o.position.y + raw.y * 0.5 - listener_y).atan2(distance);
        // Increasing pitch first grows each projected extent as base*cos(a) + arm*sin(a).
        // Stop at its first contact in either axis. All brackets for this product
        // retain the same cabinet dimensions even when fitted at different angles.
        let limit_in = |base: f32, arm: f32, span: f32| {
            let diagonal = base.hypot(arm);
            if span >= diagonal {
                FRAC_PI_2
            } else {
                (arm.atan2(base) - (span / diagonal).clamp(0., 1.).acos()).max(0.)
            }
        };
        let h = raw.y;
        let limit = limit_in(depth, h, span).min(limit_in(h, depth, o.size.y));
        let adjustment = install.random_range(0.68..1.0);
        mount.pitch_radians = aim.clamp(0., 28_f32.to_radians()).min(limit * adjustment);
        let (sin, cos) = mount.pitch_radians.sin_cos();
        let w = raw
            .x
            .min(o.size.x - 2. * (mount.bearing_gap + 1.65 * mount.strap));
        let body = Vec3::new(w, h, depth);
        mount.center = Vec3::new(
            0.,
            0.5 * (h * cos + depth * sin),
            mount.rear_plane + back_gap + 0.5 * (h * sin + depth * cos),
        );
        (mount, body)
    }

    pub(super) fn cabinet(self, body: Vec3) -> Transform {
        let rotation = Quat::from_rotation_x(self.pitch_radians);
        Transform::from_translation(self.center - rotation * (Vec3::Y * (body.y * 0.5)))
            .with_rotation(rotation)
    }

    pub(super) fn build_yoke(self, a: &mut Assembly, body: Vec3) {
        if self.kind != MountKind::WallYoke {
            return;
        }
        let t = self.strap;
        let half = body.x * 0.5 + self.bearing_gap;
        let rise = (0.19 * body.y).clamp(4.8 * t, 0.060).min(body.z * 0.78);
        let back = self.rear_plane;
        let tip = self.center.z + rise * 0.45;
        let g = a.part(Surface::Metal, acoustic::YOKE_LABEL);
        g.cuboid(
            Vec3::new(2. * (half + t), rise, t),
            0.,
            Transform::from_xyz(0., self.center.y, back + 0.5 * t),
        );
        for side in [-1., 1.] {
            let z0 = back + t;
            g.cuboid(
                Vec3::new(t, rise, tip - z0),
                0.,
                Transform::from_xyz(side * (half + t * 0.5), self.center.y, (z0 + tip) * 0.5),
            );
            // Each pivot passes through the side cheek into the tapered cabinet.
            let outer = half + t * 1.6;
            let inner = 0.38 * body.x;
            g.cylinder(
                rise * 0.30,
                outer - inner,
                Transform::from_xyz(side * (outer + inner) * 0.5, self.center.y, self.center.z)
                    .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            );
        }
    }
}
