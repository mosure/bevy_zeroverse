use super::*;
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
        frame: Surface,
        size: Vec3,
        pose: Transform,
    ) {
        for leaf in &self.leaves {
            let mut sheet = Assembly::default();
            self.sheet(&mut sheet, frame, leaf, size);
            let swing = self.swing(leaf.hinge, size.z);
            let flat = Transform::from_xyz(leaf.flat_x_m, 0., 0.);
            append_at(a, sheet, pose.mul_transform(swing).mul_transform(flat));
            if let Some(h) = leaf.hinge {
                self.mount(a, h, leaf, size, pose);
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

    fn mount(&self, a: &mut Assembly, h: Hinge, leaf: &Leaf, size: Vec3, pose: Transform) {
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
            a.part(Surface::Metal, CLASS).lathe(
                &profile,
                12,
                pose.mul_transform(Transform::from_translation(center)),
            );
            // Outer knuckles fasten to the stationary leaf. The middle strap
            // follows the moving leaf around the same pin.
            for part in [-1, 0, 1] {
                let moving = part == 0;
                let dx = if moving { side } else { -side };
                let p = Vec3::new(
                    axis.x + dx * reach * 0.5,
                    y + part as f32 * height / 3.,
                    size.z * 0.5 + 0.0007,
                );
                a.part(Surface::Metal, CLASS).cuboid(
                    Vec3::new(reach, height / 3. - 2. * groove, 0.0014),
                    0.,
                    (if moving { turned } else { pose })
                        .mul_transform(Transform::from_translation(p)),
                );
            }
        }
    }
}
