use super::*;
use crate::scene::procedural_indoor::motion::{self, MotionKind};
const CLASS: &str = "whiteboard";

fn corners(half: Vec2, z: f32) -> [Vec3; 4] {
    [
        (-half.x, -half.y),
        (half.x, -half.y),
        (half.x, half.y),
        (-half.x, half.y),
    ]
    .map(|(x, y)| Vec3::new(x, y, z))
}

impl Installation {
    pub(in super::super) fn build(
        &self,
        a: &mut Assembly,
        o: &IndoorObject,
        frame: Surface,
        size: Vec3,
        pose: Transform,
        version: u32,
    ) {
        for (index, leaf) in self.leaves.iter().enumerate() {
            let mut sheet = Assembly::default();
            self.sheet(&mut sheet, frame, leaf, size);
            let swing = self.swing(leaf.hinge, size.z);
            let flat = Transform::from_xyz(leaf.flat_x_m, 0., 0.);
            let placed = pose.mul_transform(swing).mul_transform(flat);
            let seed = o.seed ^ u64::from_le_bytes(*b"BoardArm") ^ index as u64;
            if let Some(h) = leaf.hinge.filter(|h| {
                version >= motion::OPERATION_CYCLE_VERSION
                    && h.stop_rad.abs() > 0.01
                    && a.motion_selected(MotionKind::TeachingBoard, seed)
            }) {
                let pin = pose.transform_point(self.pivot(h, size.z));
                let target = h.stop_rad.signum()
                    * motion::construction::endpoint(seed, h.angle_rad.abs(), h.stop_rad.abs());
                let mut joint = motion::construction::rigid(
                    MotionKind::TeachingBoard,
                    seed,
                    pin,
                    pose.rotation * Vec3::Y,
                    target - h.angle_rad,
                    Vec3::ZERO,
                );
                joint.frame = Transform::from_translation(pin).with_rotation(placed.rotation);
                let contact = format!("contact_board_pin{index}");
                let mut wing = Assembly::default();
                append_at(&mut wing, sheet, placed);
                self.mount(a, Some((&mut wing, &contact)), leaf, size, pose);
                joint.attachment = Some(contact);
                a.articulated(wing, joint);
            } else {
                append_at(a, sheet, placed);
                self.mount(a, None, leaf, size, pose);
            }
        }
        if let Some(t) = &self.ledge {
            // The level ledge bears on the stationary carrier, even on tilted boards.
            let center = (t.lo_m + t.hi_m) * 0.5;
            let span = t.hi_m - t.lo_m;
            a.box_part(
                frame,
                CLASS,
                Vec3::new(center.x, t.top_m - 0.002, center.y),
                Vec3::new(span.x, 0.004, span.y),
                0.0007,
            );
            a.box_part(
                frame,
                CLASS,
                Vec3::new(center.x, t.top_m + t.lip_m * 0.5, t.hi_m.y - 0.0015),
                Vec3::new(span.x, t.lip_m, 0.003),
                0.0007,
            );
            for item in &t.contents {
                presentation::contents::build_ledge_item(a, t.top_m, item);
            }
        }
    }

    pub(super) fn sheet(&self, a: &mut Assembly, frame: Surface, leaf: &Leaf, size: Vec3) {
        let half = Vec2::new(leaf.width_m, size.y) * 0.5;
        let inner = half - Vec2::splat(self.border_m);
        let z = size.z * 0.5;
        let b = self.bevel_m;
        let face = z - self.recess_m;
        // Trace the visible sandwich from one writing skin to the other.
        // Its mitered rim and chamfers share edges, with no hidden cuboids.
        let rings = [
            (inner, face),
            (inner, z - b),
            (inner + Vec2::splat(b), z),
            (half - Vec2::splat(b), z),
            (half, z - b),
            (half, -z + b),
            (half - Vec2::splat(b), -z),
            (inner + Vec2::splat(b), -z),
            (inner, -z + b),
            (inner, -face),
        ]
        .map(|(d, z)| corners(d, z));
        let g = a.part(frame, CLASS);
        for band in rings.windows(2) {
            for i in 0..4 {
                let j = (i + 1) % 4;
                let quad = [band[0][i], band[1][i], band[1][j], band[0][j]];
                g.metric_quad(quad, (quad[1] - quad[0]).cross(quad[2] - quad[0]));
            }
        }
        let write = |a: &mut Assembly, p: [Vec3; 4], normal, slot, u: [f32; 2]| {
            a.part(
                Surface::Whiteboard,
                &format!("{CLASS}#finish{slot}#uv_local"),
            )
            .vertex_quad(
                p,
                [normal; 4],
                [[u[0], 1.], [u[1], 1.], [u[1], 0.], [u[0], 0.]],
            );
        };
        write(a, rings[0], Vec3::Z, leaf.front_slot, leaf.front_u);
        let [bl, br, tr, tl] = rings[rings.len() - 1];
        let back = [br, bl, tl, tr];
        if let Some(slot) = leaf.reverse_slot {
            write(a, back, Vec3::NEG_Z, slot, [0., 1.]);
        } else {
            a.part(self.rear_stock, CLASS)
                .metric_quad(back, Vec3::NEG_Z);
        }
    }

    fn mount(
        &self,
        a: &mut Assembly,
        mut moving: Option<(&mut Assembly, &str)>,
        leaf: &Leaf,
        size: Vec3,
        pose: Transform,
    ) {
        let Some(h) = leaf.hinge else { return };
        let axis = self.pivot(h, size.z);
        let side = (leaf.flat_x_m - h.axis_x_m).signum();
        let reach = (leaf.flat_x_m - h.axis_x_m).abs() - leaf.width_m * 0.5 + self.border_m * 0.65;
        let radius = self.pin_radius_m;
        let height = self.hinge_height_m;
        let pin = radius * 0.54;
        let groove = 0.0007;
        let third = height / 6.;
        let profile = [
            (0., -height * 0.5),
            (radius, -height * 0.5),
            (radius, -third - groove),
            (pin, -third - groove),
            (pin, -third + groove),
            (radius, -third + groove),
            (radius, third - groove),
            (pin, third - groove),
            (pin, third + groove),
            (radius, third + groove),
            (radius, height * 0.5),
            (0., height * 0.5),
        ];
        let turned = pose.mul_transform(self.swing(Some(h), size.z));
        for y in [-1., 1.].map(|s| s * self.hinge_inset_fraction * size.y) {
            let center = axis.with_y(y);
            let spindle = pose.mul_transform(Transform::from_translation(center));
            if let Some((wing, contact)) = &mut moving {
                // The bearing gets one continuous fixed axle and a separate
                // middle sleeve. The bore sits just inside the axle's inradius
                // so turning its polygonal cross-section never exposes a crack.
                let mut fixed = profile;
                fixed[5].0 = pin;
                fixed[6].0 = pin;
                a.part(Surface::Metal, &format!("{CLASS}#{contact}"))
                    .lathe(&fixed, 12, spindle);
                let sleeve = [
                    (pin * 0.90, profile[5].1),
                    profile[5],
                    profile[6],
                    (pin * 0.90, profile[6].1),
                ];
                wing.part(Surface::Metal, CLASS).lathe(&sleeve, 12, spindle);
            } else {
                a.part(Surface::Metal, CLASS).lathe(&profile, 12, spindle);
            }
            // Outer knuckles fasten to the stationary leaf. The middle strap
            // follows the moving leaf around the same pin.
            for part in [-1, 0, 1] {
                let on_wing = part == 0;
                let dx = if on_wing { side } else { -side };
                let p = Vec3::new(
                    axis.x + dx * reach * 0.5,
                    y + part as f32 * height / 3.,
                    size.z * 0.5 + 0.0007,
                );
                let dst = match &mut moving {
                    Some((wing, _)) if on_wing => &mut **wing,
                    _ => &mut *a,
                };
                dst.part(Surface::Metal, CLASS).cuboid(
                    Vec3::new(reach, height / 3. - 2. * groove, 0.0014),
                    0.,
                    (if on_wing { turned } else { pose })
                        .mul_transform(Transform::from_translation(p)),
                );
            }
        }
    }
}
