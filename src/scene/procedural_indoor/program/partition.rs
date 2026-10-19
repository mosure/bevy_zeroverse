//! Partition construction consumes the same room-wide joinery as its neighbors.
use super::Partition;
use crate::scene::procedural_indoor::{
    architecture::{details::FinishParameters, trim},
    materials::Surface,
    objects::Assembly,
};
use bevy::prelude::*;

impl Partition {
    pub fn build(&self, height: f32, a: &mut Assembly) {
        self.build_with_finishes(height, a, None);
    }

    pub(crate) fn build_with_finishes(
        &self,
        height: f32,
        a: &mut Assembly,
        finishes: Option<&FinishParameters>,
    ) {
        let glazing = finishes.and_then(|f| f.interior_glazing.as_ref());
        let fw = glazing.map_or(0.035, |p| p.frame_width_m);
        let frame = glazing.map_or(Surface::Metal, |p| p.frame);
        let frame_label = glazing.map_or_else(
            || "window".into(),
            |p| format!("window#finish{}", p.joinery.frame_finish_slot),
        );
        let depth = glazing.map_or(self.thickness, |p| {
            self.thickness * p.joinery.mullion_depth_fraction
        });

        // A narrow opaque gasket closes the construction reveal at each wall
        // junction. Adjacent caps meet with opposite normals and no shared area.
        for along in [self.start + 0.003, self.end - 0.003] {
            a.box_part(
                Surface::Rubber,
                "wall",
                self.position(along, height * 0.5),
                self.size(0.006, height, self.thickness),
                0.0,
            );
        }
        for (index, (mut lo, mut hi)) in self.spans().into_iter().enumerate() {
            // Window posts must terminate beside the door jamb, not occupy the
            // same volume. A 4.5 mm construction reveal separates their side faces.
            if index == 0 {
                hi -= 0.022;
                lo += 0.006;
            } else {
                lo += 0.022;
                hi -= 0.006;
            }
            let length = hi - lo;
            let center = (lo + hi) * 0.5;
            let glazed = (height - self.sill - 0.16) * self.glazing_fraction;
            let top = self.sill + glazed;
            // Metal rails replace this strip of plaster. Identical thickness
            // plus overlapping extents made both surfaces compete in depth.
            let rail_half = if glazed > 0.01 { fw * 0.5 } else { 0.0 };
            for (bottom, upper) in [(0.0, self.sill - rail_half), (top + rail_half, height)] {
                if upper - bottom > 0.002 {
                    a.box_part(
                        Surface::Paint,
                        "wall",
                        self.position(center, (bottom + upper) * 0.5),
                        self.size(length, upper - bottom, self.thickness),
                        0.003,
                    );
                }
            }
            if glazed > 0.01 {
                a.box_part(
                    Surface::GlassInterior,
                    "window",
                    self.position(center, self.sill + glazed * 0.5),
                    self.size(
                        (length - 2. * fw - 0.004).max(0.001),
                        (glazed - fw - 0.004).max(0.001),
                        0.010,
                    ),
                    0.001,
                );
                let bays = (length / self.mullion_pitch).ceil().max(1.0) as u32;
                for i in 0..=bays {
                    a.box_part(
                        frame,
                        &frame_label,
                        self.position(
                            if i == 0 {
                                lo + fw * 0.5
                            } else if i == bays {
                                hi - fw * 0.5
                            } else {
                                lo + length * i as f32 / bays as f32
                            },
                            self.sill + glazed * 0.5,
                        ),
                        self.size(fw, (glazed - fw).max(0.005), depth),
                        0.002,
                    );
                }
                for y in [self.sill, top] {
                    a.box_part(
                        frame,
                        &frame_label,
                        self.position(center, y),
                        self.size(length, fw, self.thickness),
                        0.002,
                    );
                }
                if let Some(fraction) = self.transom_fraction {
                    for bay in 0..bays {
                        let lo = lo + length * bay as f32 / bays as f32 + fw + 0.003;
                        let hi = lo + length / bays as f32 - 2. * (fw + 0.003);
                        if hi > lo {
                            a.box_part(
                                frame,
                                &frame_label,
                                self.position((lo + hi) * 0.5, self.sill + glazed * fraction),
                                self.size(hi - lo, fw * 0.77, depth),
                                0.002,
                            );
                        }
                    }
                }
            }
            if let Some(trim) = finishes.and_then(|f| f.trim.as_ref()) {
                if let Some(base) = &trim.base {
                    let mut base = base.clone();
                    // Keep the existing conservative obstacle envelope. A thin
                    // applied skirting is plausible on a demountable partition.
                    base.depth_m = base.depth_m.min(0.022);
                    let normal = if self.axis == 0 { Vec3::X } else { Vec3::Z };
                    for sign in [-1., 1.] {
                        let yaw = if self.axis == 0 {
                            std::f32::consts::FRAC_PI_2
                        } else {
                            0.
                        } + if sign < 0. { std::f32::consts::PI } else { 0. };
                        let tf = Transform::from_translation(
                            self.position(center, 0.) + normal * sign * self.thickness * 0.5,
                        )
                        .with_rotation(Quat::from_rotation_y(yaw));
                        // A low glazed sill leaves trim only on its solid backing.
                        let mut base = base.clone();
                        if glazed > 0.01 {
                            base.height_m = base
                                .height_m
                                .min(self.sill - fw * 0.5 - trim.joint_m - 0.002);
                        }
                        if base.height_m > 0.008 {
                            trim::build(
                                a,
                                tf,
                                Vec2::new(-length * 0.5, length * 0.5),
                                Vec2::splat(trim.joint_m),
                                &base,
                                &[],
                                false,
                            );
                        }
                    }
                }
            } else {
                a.box_part(
                    Surface::WoodEdge,
                    "wall",
                    self.position(center, 0.05),
                    self.size(length, 0.10, self.thickness + 0.035),
                    0.002,
                );
            }
        }
        if self.arch_rise > 0. {
            super::super::envelope::construction::arched_portal(self, height, a);
            return;
        }
        a.box_part(
            Surface::Paint,
            "wall",
            self.position(self.door_center, (height + self.door_height) * 0.5),
            self.size(self.door_width, height - self.door_height, self.thickness),
            0.003,
        );
        if self.door.is_none() {
            for along in [
                self.door_center - self.door_width * 0.5,
                self.door_center + self.door_width * 0.5,
            ] {
                a.box_part(
                    Surface::WoodEdge,
                    "door_frame",
                    self.position(along, (self.door_height - 0.0175) * 0.5),
                    self.size(0.035, self.door_height - 0.0175, self.thickness + 0.02),
                    0.002,
                );
            }
            a.box_part(
                Surface::WoodEdge,
                "door_frame",
                self.position(self.door_center, self.door_height),
                self.size(self.door_width + 0.035, 0.035, self.thickness + 0.02),
                0.002,
            );
        }
    }
}
