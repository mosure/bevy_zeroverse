//! Each key and display has one opaque front. Neighboring digit quads share
//! edges, so neither the housing nor an underlying digit can fight the print.
use super::*;

fn corners(xy: Vec2, size: Vec2, z: f32) -> [Vec3; 4] {
    [
        Vec2::new(-1., -1.),
        Vec2::new(1., -1.),
        Vec2::ONE,
        Vec2::new(-1., 1.),
    ]
    .map(|corner| (xy + corner * size * 0.5).extend(z))
}

fn glyph(g: &mut Geometry, pose: Transform, xy: Vec2, size: Vec2, z: f32, symbol: Glyph) {
    g.vertex_quad(
        corners(xy, size, z).map(|v| pose.transform_point(v)),
        [pose.rotation * Vec3::Z; 4],
        symbol.uvs(),
    );
}

/// Narrow chamfer around an insert. Its top is supplied by the glyph face.
fn lip(g: &mut Geometry, pose: Transform, xy: Vec2, size: Vec2, height: f32) {
    let base = corners(xy, size + Vec2::splat(height * 0.8), 0.).map(|v| pose.transform_point(v));
    let face = corners(xy, size, height).map(|v| pose.transform_point(v));
    for i in 0..4 {
        let j = (i + 1) % 4;
        let points = [base[i], base[j], face[j], face[i]];
        let normal = (base[j] - base[i]).cross(face[i] - base[i]).normalize();
        g.metric_quad(points, normal);
    }
}

impl Controls {
    pub(in super::super) fn build(&self, a: &mut Assembly, o: &IndoorObject, casing: Surface) {
        let l = o.kind.class_name();
        let f = self.fascia;
        let pose = Transform::from_translation(f.face).with_rotation(f.orientation);
        a.part(casing, l).cuboid(
            f.size.extend(f.thickness),
            0.003,
            pose * Transform::from_translation(Vec3::NEG_Z * f.thickness * 0.5),
        );
        let print = self.key_film.label(l);
        for dial in &self.dials {
            let xy = dial.center;
            let r = dial.radius_m;
            let size = Vec2::splat(r * 2.76);
            let relief = 0.0012;
            lip(a.part(casing, l), pose, xy, size, relief);
            glyph(
                a.part(Surface::Ink, &print),
                pose,
                xy,
                size,
                relief,
                Glyph::Scale,
            );
            let foot = xy.extend(relief);
            let h = dial.depth_m;
            a.part(self.dial_surface, l).lathe(
                &[
                    (r * 0.94, 0.),
                    (r, h * 0.16),
                    (r * 0.94, h * 0.88),
                    (r * 0.84, h),
                    (0., h),
                ],
                24,
                pose * Transform::from_translation(foot)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            );
            let q = Quat::from_rotation_z(dial.angle_rad);
            let mark = if self.dial_surface == Surface::Metal {
                Surface::Ink
            } else {
                Surface::Metal
            };
            a.part(mark, l).cuboid(
                Vec3::new(r * 0.13, r * 0.48, 0.0016),
                0.00025,
                pose * Transform::from_translation(
                    foot + q * (Vec3::Y * r * 0.52) + Vec3::Z * (h + 0.00065),
                )
                .with_rotation(q),
            );
        }
        for key in &self.keys {
            let xy = key.center;
            let h = (key.size.min_element() * 0.09).min(0.0016);
            lip(a.part(casing, l), pose, xy, key.size, h);
            glyph(
                a.part(Surface::Ink, &print),
                pose,
                xy,
                key.size,
                h,
                key.glyph,
            );
        }
        if let Some(display) = self.readout {
            let xy = display.center;
            let h = 0.002;
            lip(a.part(casing, l), pose, xy, display.size, h);
            let face = a.part(Surface::Ink, &display.film.label(l));
            if display.blank {
                glyph(face, pose, xy, display.size, h, Glyph::Blank);
                return;
            }
            // Fixed cell edges are also used for blank digits and standby.
            let edges = [-0.5, -0.28, -0.06, 0.06, 0.28, 0.5];
            for (symbol, pair) in display.glyphs().into_iter().zip(edges.windows(2)) {
                let w = display.size.x * (pair[1] - pair[0]);
                let mid = xy + Vec2::X * display.size.x * ((pair[0] + pair[1]) * 0.5);
                glyph(face, pose, mid, Vec2::new(w, display.size.y), h, symbol);
            }
        }
    }
}
