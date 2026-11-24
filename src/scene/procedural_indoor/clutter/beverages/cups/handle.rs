//! Ceramic grip with two embedded sockets in the actual wall of the cup.
use super::*;

impl mesh::Section {
    pub(super) fn handle(
        &self,
        a: &mut Assembly,
        f: &CupForm,
        p: &BeverageProgram,
        gauge: f32,
        label: &str,
    ) {
        let g = a.part(Surface::Ceramic, label);
        let first = g.positions.len() as u32;
        let tf = f.transform();
        let angle = PI / f.sides as f32;
        let (s0, c0) = angle.sin_cos();
        let (s1, c1) = (3. * angle).sin_cos();
        // Fit a point to the wall by horizontal shear. The first two facets on
        // either side of +X cover the width of the grip even on a narrow belly.
        let mut vertex = |mut q: Vec3, n: Vec3, uv: [f32; 2]| {
            let (r, dr_dy) = self.radius_and_slope(q.y);
            let z = q.z.abs();
            let (scale, tangent) = if r - z * s0 / c0 <= (r * c0 - z * s1) / c1 {
                (1., s0 / c0)
            } else {
                (c0 / c1, s1 / c1)
            };
            q.x += scale * r - z * tangent - f.wall_m * 0.44;
            let dx_dz = if z < 1e-8 {
                0.
            } else {
                -q.z.signum() * tangent
            };
            // Inverse transpose of the shear, before the rigid cup transform.
            let fitted = Vec3::new(n.x, n.y - n.x * scale * dr_dy, n.z - n.x * dx_dz);
            g.positions.push(tf.transform_point(q).to_array());
            g.normals
                .push((tf.rotation * fitted.normalize()).to_array());
            g.uvs.push(uv);
        };
        let low = (f.height_m * p.handle_bottom)
            .max(f.saucer.as_ref().map_or(0., |s| s.rim_y_m - f.base_y_m) + gauge * 1.3);
        let high = f.height_m * p.handle_top;
        let mid = (high + low) * 0.5;
        let half = (high - low) * 0.5;
        let shape = (p.handle_shape - 1.) * 0.5;
        let reach = f.handle_reach_m;
        const SIDES: u32 = 10;
        const BENDS: u32 = 20;
        static RING: OnceLock<[Vec2; SIDES as usize + 1]> = OnceLock::new();
        let ring = RING.get_or_init(|| {
            std::array::from_fn(|i| {
                Vec2::from_angle(TAU * (i as u32 % SIDES) as f32 / SIDES as f32)
            })
        });
        // Nested half-ovals preserve the grip's opening on small or shallow cups.
        // Offsetting a tight centreline by its normal can invert the inner skin.
        let mut distance = 0.;
        let mut last = Vec2::new(0., high);
        for i in 0..=BENDS {
            let (sn, cs) = (PI * i as f32 / BENDS as f32).sin_cos();
            let sn = sn.max(0.);
            let x = sn * (1. + shape * cs * cs);
            let dx = cs * (1. + shape * (3. * cs * cs - 2.));
            let center = Vec2::new(reach * x, mid + half * cs);
            distance += center.distance(last);
            last = center;
            for (j, &v) in ring.iter().enumerate() {
                let width = reach + gauge * v.x;
                let height = half + gauge * v.x;
                let pos = Vec3::new(width * x, mid + height * cs, gauge * v.y);
                let normal = Vec3::new(
                    height * sn * v.x,
                    width * dx * v.x,
                    v.y * (width * dx * cs + height * sn * x),
                );
                vertex(
                    pos,
                    normal,
                    [TAU * gauge * j as f32 / SIDES as f32, distance],
                );
            }
        }
        for (end, y, sign) in [(0, high, 1.), (1, low, -1.)] {
            let cap = first + (BENDS + 1) * (SIDES + 1) + end * (SIDES + 2);
            vertex(Vec3::new(0., y, 0.), Vec3::NEG_X, [0., 0.]);
            for &v in ring {
                let dy = sign * gauge * v.x;
                let z = gauge * v.y;
                vertex(Vec3::new(0., y + dy, z), Vec3::NEG_X, [dy, z]);
            }
            for k in 0..SIDES {
                let a = cap + 1 + k;
                g.indices.extend(if end == 0 {
                    [cap, a + 1, a]
                } else {
                    [cap, a, a + 1]
                });
            }
        }
        for i in 0..BENDS {
            for j in 0..SIDES {
                let a = first + i * (SIDES + 1) + j;
                let b = a + SIDES + 1;
                g.indices.extend([a, a + 1, b + 1, a, b + 1, b]);
            }
        }
    }
}
