//! Bounded local assemblies. All geometry shares the camera/placement reserve.
mod panels;
mod recesses;
mod services;
mod truss;
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

struct Frame<'a> {
    m: &'a CeilingModule,
    tf: Transform,
    length: f32,
    width: f32,
    height: f32,
    label: String,
}
impl<'a> Frame<'a> {
    fn new(m: &'a CeilingModule, finish: u8) -> Self {
        let rotate = m.size.x < m.size.z;
        Self {
            m,
            tf: Transform::from_translation(m.center.with_y(m.body_bounds().0.y)).with_rotation(
                Quat::from_rotation_y(if rotate {
                    -std::f32::consts::FRAC_PI_2
                } else {
                    0.
                }),
            ),
            length: m.size.x.max(m.size.z),
            width: m.size.x.min(m.size.z),
            height: m.size.y,
            label: format!("ceiling#finish{finish}"),
        }
    }
    fn part<'b>(&self, a: &'b mut Assembly, material: Surface) -> &'b mut Geometry {
        a.part(
            material,
            if super::super::super::materials::variants::supports(material) {
                &self.label
            } else {
                "ceiling"
            },
        )
    }
    fn block(&self, a: &mut Assembly, material: Surface, p: Vec3, size: Vec3, bevel: f32) {
        self.part(a, material)
            .cuboid(size, bevel, Transform::from_translation(p));
    }
    fn tube(&self, a: &mut Assembly, material: Surface, points: &[Vec3], radius: f32, sides: u32) {
        self.part(a, material).tube(points, radius, sides);
    }
    fn rod(&self, a: &mut Assembly, material: Surface, p: Vec3, q: Vec3, radius: f32) {
        if p.distance_squared(q) > 1e-10 {
            self.tube(a, material, &[p, q], radius, 8);
        }
    }
    fn suspend(&self, a: &mut Assembly, scene: &IndoorManifest, anchors: &[Vec3]) {
        if let Some(supports) = &self.m.suspension {
            let inverse = self.tf.compute_affine().inverse();
            for support in supports {
                let p = inverse.transform_point3(support.start);
                let end = inverse.transform_point3(support.end);
                self.rod(a, Surface::Metal, p, end, 0.0025);
                self.block(
                    a,
                    Surface::Metal,
                    end - Vec3::Y * 0.001,
                    Vec3::new(0.02, 0.006, 0.02),
                    0.001,
                );
            }
            return;
        }
        for &p in anchors {
            let world = self.tf.transform_point(p);
            let roof = scene.ceiling_height(world.xz()) - self.tf.translation.y;
            // Rods connect to actual load-bearing rails or pipe saddles. End
            // plates are recessed 1 mm from the slab (no coincident faces).
            self.rod(a, Surface::Metal, p, p.with_y(roof - 0.003), 0.0025);
            self.block(
                a,
                Surface::Metal,
                p.with_y(roof - 0.004),
                Vec3::new(0.02, 0.006, 0.02),
                0.001,
            );
        }
    }
}

pub(super) fn build(a: &mut Assembly, scene: &IndoorManifest, m: &CeilingModule) {
    build_internal(a, scene, m, false);
}

pub(super) fn anchors(scene: &IndoorManifest, m: &CeilingModule) -> Vec<Vec3> {
    build_internal(&mut Assembly::default(), scene, m, true)
}

fn build_internal(
    a: &mut Assembly,
    scene: &IndoorManifest,
    m: &CeilingModule,
    anchors_only: bool,
) -> Vec<Vec3> {
    let Some(detail) = &m.detail else {
        super::legacy::build(a, scene, m);
        return vec![];
    };
    let f = Frame::new(m, detail.finish);
    let mut local = Assembly::default();
    let anchors = match detail.shape {
        CeilingShape::Coffer {
            aspect,
            beam_fraction,
            recess_fraction,
            timber,
        } => recesses::coffers(
            &mut local,
            &f,
            aspect,
            beam_fraction,
            recess_fraction,
            timber,
        ),
        CeilingShape::Diffuser {
            round,
            blades,
            angle,
        } => recesses::diffuser(&mut local, &f, round, blades, angle),
        CeilingShape::Panel {
            camber,
            taper,
            phase,
        } => panels::panels(&mut local, &f, camber, taper, phase),
        CeilingShape::Grid { aspect, recess_m } => panels::grid(&mut local, &f, aspect, recess_m),
        CeilingShape::Duct {
            section,
            end_scale,
            bend,
            rise,
            branches,
            seam_pitch_m,
        } => services::duct(
            &mut local,
            &f,
            section,
            end_scale,
            bend,
            rise,
            branches,
            seam_pitch_m,
        ),
        CeilingShape::Sprinkler {
            radius_m,
            branch_reach,
            head_pitch_m,
            upright,
        } => services::sprinkler(
            &mut local,
            &f,
            radius_m,
            branch_reach,
            head_pitch_m,
            upright,
        ),
        CeilingShape::Conduit {
            count,
            radius_m,
            bend,
            rise,
            junction_pitch_m,
        } => services::conduit(
            &mut local,
            &f,
            count,
            radius_m,
            bend,
            rise,
            junction_pitch_m,
        ),
        CeilingShape::Tray {
            cables,
            rung_pitch_m,
            rise,
        } => services::tray(&mut local, &f, cables, rung_pitch_m, rise),
        CeilingShape::Truss {
            web,
            chord_radius_m,
            cell_pitch_m,
            taper,
        } => truss::build(&mut local, &f, web, chord_radius_m, cell_pitch_m, taper),
    };
    if anchors_only {
        return anchors.iter().map(|&p| f.tf.transform_point(p)).collect();
    }
    if matches!(detail.shape, CeilingShape::Diffuser { .. }) {
        // A suspended terminal needs an air connection as well as hangers.
        // The neck occupies the existing support reservation and ends just
        // below the lowest roof sample across its section on a sloping slab.
        let radius = f.width.min(f.length) * 0.14;
        let roof = [Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z]
            .map(|d| {
                scene.ceiling_height(f.tf.transform_point(d * radius).xz()) - f.tf.translation.y
            })
            .into_iter()
            .fold(f32::INFINITY, f32::min);
        f.rod(
            &mut local,
            Surface::Metal,
            Vec3::Y * (f.height * 0.76 + 0.016),
            Vec3::Y * (roof - 0.003),
            radius,
        );
    }
    f.suspend(&mut local, scene, &anchors);
    for (key, mut g) in local.parts {
        for p in &mut g.positions {
            *p = f.tf.transform_point(Vec3::from_array(*p)).to_array();
        }
        for n in &mut g.normals {
            *n = (f.tf.rotation * Vec3::from_array(*n)).to_array();
        }
        a.parts.entry(key).or_default().append(g);
    }
    vec![]
}

/// Closed loft along local X. Ring planes remain parallel; oblique sections
/// produce real elbows/reducers without intersecting capped pipe primitives.
fn loft(g: &mut Geometry, rings: &[(Vec3, Vec2)], round: bool) {
    let sides = if round { 20 } else { 4 };
    let radial = |i: usize| {
        if round {
            let t = std::f32::consts::TAU * i as f32 / sides as f32;
            Vec2::new(t.cos(), t.sin())
        } else {
            [
                Vec2::new(1., 1.),
                Vec2::new(-1., 1.),
                Vec2::new(-1., -1.),
                Vec2::new(1., -1.),
            ][i % sides]
        }
    };
    let point = |ring: &(Vec3, Vec2), i| {
        let p = radial(i) * ring.1;
        ring.0 + Vec3::new(0., p.x, p.y)
    };
    let mut distance = 0.;
    for pair in rings.windows(2) {
        let step = pair[0].0.distance(pair[1].0);
        for j in 0..sides {
            let ps = [
                point(&pair[0], j),
                point(&pair[0], j + 1),
                point(&pair[1], j + 1),
                point(&pair[1], j),
            ];
            let n = (ps[1] - ps[0]).cross(ps[2] - ps[0]).normalize();
            let index = g.positions.len() as u32;
            let circumference = std::f32::consts::TAU * pair[0].1.max_element();
            for (k, p) in ps.into_iter().enumerate() {
                let r = radial(j + usize::from(k == 1 || k == 2));
                let delta = pair[1].0 - pair[0].0;
                let dr = pair[1].1 - pair[0].1;
                let ny = r.x / pair[usize::from(k >= 2)].1.x;
                let nz = r.y / pair[usize::from(k >= 2)].1.y;
                let smooth = Vec3::new(
                    -((delta.y + r.x * dr.x) * ny + (delta.z + r.y * dr.y) * nz) / delta.x,
                    ny,
                    nz,
                )
                .normalize();
                g.positions.push(p.to_array());
                g.normals.push(if round { smooth } else { n }.to_array());
                g.uvs.push([
                    distance + if k >= 2 { step } else { 0. },
                    circumference * (j + usize::from(k == 1 || k == 2)) as f32 / sides as f32,
                ]);
            }
            g.indices
                .extend([index, index + 1, index + 2, index, index + 2, index + 3]);
        }
        distance += step;
    }
    for (ring, normal, reverse) in [
        (rings.first().unwrap(), Vec3::NEG_X, true),
        (rings.last().unwrap(), Vec3::X, false),
    ] {
        for j in 0..sides {
            let points = if reverse {
                [ring.0, point(ring, j + 1), point(ring, j)]
            } else {
                [ring.0, point(ring, j), point(ring, j + 1)]
            };
            let first = g.positions.len() as u32;
            for p in points {
                g.positions.push(p.to_array());
                g.normals.push(normal.to_array());
                g.uvs.push([p.y, p.z]);
            }
            g.indices.extend([first, first + 1, first + 2]);
        }
    }
}
