//! Manufactured work desks. The columns, feet and rear modesty screen participate
//! in the same seated-person clearance tests as the tabletop and its other frames.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct DeskFrame {
    pub leg_span: f32,
    pub rear_setback: f32,
    pub foot_fraction: f32,
    pub post_width_m: f32,
    pub sleeve_fraction: f32,
    pub metal: Surface,
    pub modesty: Option<ModestyPanel>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModestyPanel {
    pub drop_m: f32,
    pub end_inset: f32,
    pub thickness_m: f32,
    pub surface: Surface,
}

pub(crate) fn is_raised_desk(o: &IndoorObject) -> bool {
    o.kind == ObjectKind::Desk && o.size.y >= 0.93 && o.size.x >= 0.88 && o.size.z >= 0.48
}

impl DeskFrame {
    pub(super) fn sample(o: &IndoorObject) -> Option<Self> {
        if o.kind != ObjectKind::Desk
            || o.variant == SHARED_ROW_VARIANT
            || o.size.x < 0.88
            || o.size.z < 0.48
            || o.size.y < 0.60
        {
            return None;
        }
        let mut r = stream(o.seed, 0x4445_534b_4c45_4753);
        if !r.random_bool(0.53) && !is_raised_desk(o) {
            return None;
        }
        // T-foot and offset C-foot assemblies use the same connected construction.
        Some(Self {
            leg_span: r.random_range(0.53..0.84),
            rear_setback: r.random_range(-0.025..0.24),
            foot_fraction: r.random_range(0.56..0.85),
            post_width_m: r.random_range(0.055..0.091),
            sleeve_fraction: r.random_range(0.36..0.67),
            metal: if r.random_bool(0.18) {
                Surface::Chrome
            } else {
                Surface::Metal
            },
            modesty: r.random_bool(0.54).then(|| ModestyPanel {
                drop_m: r.random_range(0.16..0.38),
                end_inset: r.random_range(0.02..0.20),
                thickness_m: r.random_range(0.012..0.024),
                surface: [Surface::Wood, Surface::FabricAlt, Surface::Metal][r.random_range(0..3)],
            }),
        })
    }

    pub(super) fn build(
        &self,
        a: &mut impl StructureSink,
        o: &IndoorObject,
        outline: &[Vec2],
        top: f32,
    ) {
        let label = o.kind.class_name();
        let post = self.post_width_m;
        let mut half = o.size.x * self.leg_span * 0.5;
        let mut z = -o.size.z * self.rear_setback;
        // Keep both upper mounting rails beneath the *actual* tapered top.
        let rail = Vec2::new(post * 0.60, o.size.z * 0.17);
        while ![-half, half].iter().all(|&x| {
            let center = Vec2::new(x, z);
            supports_outline(outline, center - rail, center + rail, 0.004)
        }) && half > 0.10
        {
            half *= 0.88;
            z *= 0.88;
        }
        let foot = o.size.z * self.foot_fraction;
        let raised = is_raised_desk(o);
        let joint = if raised {
            0.36 + 0.22 * self.sleeve_fraction
        } else {
            top * self.sleeve_fraction
        };
        for x in [-half, half] {
            // End glides transfer the load to the floor. Sleeve and riser share
            // one axis; the upper wall is inset on all four sides.
            for glide in [-0.40, 0.40] {
                a.box_part(
                    Surface::Rubber,
                    label,
                    Vec3::new(x, 0.005, foot * glide),
                    Vec3::new(post * 0.80, 0.010, 0.045),
                    0.003,
                );
            }
            a.box_part(
                self.metal,
                label,
                Vec3::new(x, 0.025, 0.),
                Vec3::new(post * 1.18, 0.030, foot),
                0.006,
            );
            a.box_part(
                self.metal,
                label,
                Vec3::new(x, (joint + 0.040) * 0.5, z),
                Vec3::new(post, joint - 0.040, post * 0.84),
                0.003,
            );
            let riser = if raised {
                joint + (top - joint) * 0.48
            } else {
                joint
            };
            if raised {
                a.box_part(
                    self.metal,
                    label,
                    Vec3::new(x, (joint + riser) * 0.5, z),
                    Vec3::new(post * 0.87, riser - joint, post * 0.74),
                    0.002,
                );
            }
            a.box_part(
                self.metal,
                label,
                Vec3::new(x, (riser + top - 0.016) * 0.5, z),
                Vec3::new(post * 0.78, top - 0.016 - riser, post * 0.65),
                0.002,
            );
            a.box_part(
                self.metal,
                label,
                Vec3::new(x, top - 0.008, z),
                Vec3::new(rail.x * 2., 0.016, rail.y * 2.),
                0.002,
            );
        }
        let beam = 2. * half - post * 0.78;
        a.box_part(
            self.metal,
            label,
            Vec3::new(0., top - 0.046, z),
            Vec3::new(beam, 0.044, post * 0.59),
            0.003,
        );
        if raised {
            // Pendant controls mount to the operator's side of the actual top.
            let mut controls = stream(o.seed, 0x4c494654504144);
            let x = o.size.x
                * controls.random_range(0.15..0.29)
                * if controls.random_bool(0.5) { 1. } else { -1. };
            let mut front = o.size.z * 0.48;
            let half = Vec2::new(0.052, 0.019);
            while front > 0.
                && !supports_outline(
                    outline,
                    Vec2::new(x, front) - half,
                    Vec2::new(x, front) + half,
                    0.002,
                )
            {
                front -= 0.012;
            }
            a.box_part(
                Surface::Rubber,
                label,
                Vec3::new(x, top - 0.011, front),
                Vec3::new(0.102, 0.022, 0.037),
                0.004,
            );
            for dx in [-0.023, 0.004, 0.031] {
                a.box_part(
                    Surface::Plastic,
                    label,
                    Vec3::new(x + dx, top - 0.012, front + 0.0198),
                    Vec3::new(0.015, 0.008, 0.003),
                    0.001,
                );
            }
        }
        if let Some(p) = &self.modesty {
            let width = beam * (1. - p.end_inset);
            // Small square brackets project from the rear face of the crossbeam.
            let reach = o.size.z * 0.08;
            let rear = z - post * 0.295;
            for x in [-width * 0.36, width * 0.36] {
                a.box_part(
                    self.metal,
                    label,
                    Vec3::new(x, top - 0.045, rear - reach * 0.5),
                    Vec3::new(0.022, 0.024, reach),
                    0.002,
                );
            }
            a.box_part(
                p.surface,
                label,
                Vec3::new(
                    0.,
                    top - 0.025 - p.drop_m * 0.5,
                    rear - reach - p.thickness_m * 0.5,
                ),
                Vec3::new(width, p.drop_m, p.thickness_m),
                0.003,
            );
        }
    }
}
