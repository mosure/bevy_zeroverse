//! Extruded, cut-to-width shade fabric. Cellular stock has a through cavity
//! and a single bonded partition between successive cells.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;
use pleated::{PleatLayout, TRACK_Z};

/// Rear/front at the top, front crest, front/rear at the bottom, rear crest.
/// Points are ordered outwards at the right-hand cut end.
type Ring = [[Vec3; 2]; 6];

struct Cut {
    left: Vec3,
    span: Vec3,
    up: Vec3,
    forward: Vec3,
    width: f32,
    origin: Vec2,
}

impl Cut {
    fn ring(&self, y: [f32; 3], seams: [f32; 2], crests: [f32; 2]) -> Ring {
        let [rear, front] = seams;
        let [top, mid, bottom] = y;
        [
            (top, rear),
            (top, front),
            (mid, crests[1]),
            (bottom, front),
            (bottom, rear),
            (mid, crests[0]),
        ]
        .map(|(y, z)| {
            let point = self.left + self.up * y + self.forward * z;
            [point, point + self.span]
        })
    }

    fn walls(
        &self,
        g: &mut Geometry,
        ring: &Ring,
        v: [f32; 3],
        land: f32,
        caps: [bool; 2],
        inside: bool,
    ) {
        let warp = [v[0], v[0], v[1], v[2], v[2], v[1]];
        let x = self.origin.x;
        for i in 0..6 {
            if (i == 0 && !caps[0]) || (i == 3 && !caps[1]) {
                continue;
            }
            let j = (i + 1) % 6;
            let a = self.origin.y + warp[i];
            let b = if i == 0 || i == 3 {
                a + land
            } else {
                self.origin.y + warp[j]
            };
            face(
                g,
                [ring[i][0], ring[j][0], ring[j][1], ring[i][1]],
                [[x, a], [x, b], [x + self.width, b], [x + self.width, a]],
                inside,
            );
        }
    }
}

fn face(g: &mut Geometry, mut points: [Vec3; 4], mut uv: [[f32; 2]; 4], reverse: bool) {
    if reverse {
        points.reverse();
        uv.reverse();
    }
    let normal = (points[1] - points[0])
        .cross(points[3] - points[0])
        .normalize();
    g.vertex_quad(points, [normal; 4], uv);
}

/// Two narrow cut edges, charted in material coordinates through the cloth.
fn ends(g: &mut Geometry, rings: [&[Vec3; 2]; 4], length: f32, gauge: f32) {
    let chart = [[0., 0.], [length, 0.], [length, gauge], [0., gauge]];
    for end in 0..2 {
        face(g, rings.map(|r| r[end]), chart, end == 0);
    }
}

impl Builder<'_> {
    pub(super) fn accordion(
        &mut self,
        stock: PleatedStock,
        l: PleatLayout,
        opening: &WindowOpening,
        moving: bool,
    ) {
        let mut dye = stream(
            installation_seed(self.p, opening),
            u64::from_le_bytes(*b"PleatCut"),
        );
        let tf = self.tf;
        let x = self.center();
        let cut = Cut {
            left: tf.transform_point(Vec3::new(x - l.width * 0.5, 0., TRACK_Z)),
            span: tf.rotation * (tf.scale * Vec3::X * l.width),
            up: tf.rotation * (tf.scale * Vec3::Y),
            forward: tf.rotation * (tf.scale * Vec3::Z),
            width: l.width,
            origin: Vec2::new(dye.random_range(0.0..2.), dye.random_range(0.0..3.)),
        };
        let label = if moving {
            format!("{}#pleat", self.cloth_label)
        } else {
            self.cloth_label.clone()
        };
        let cloth = self.a.part(self.p.fabric, &label);
        panel(cloth, &cut, l, stock);
        let rail_depth = (l.packed_depth + 0.003).max(0.018);
        self.block(
            self.p.slat_material,
            Vec3::new(
                x,
                l.rail_bottom(stock) + stock.rail_height_m * 0.5,
                l.rail_z,
            ),
            Vec3::new(l.width + 0.006, stock.rail_height_m, rail_depth),
            0.001,
        );
        if stock.cordless {
            let grip = (l.width * dye.random_range(0.09..0.15)).clamp(0.045, 0.105);
            self.block(
                Surface::Plastic,
                Vec3::new(
                    x + l.width * dye.random_range(-0.12..0.12),
                    l.rail_bottom(stock) + 0.44 * stock.rail_height_m,
                    l.rail_z + 0.5 * rail_depth + 0.002,
                ),
                Vec3::new(grip, 0.30 * stock.rail_height_m, 0.010),
                0.001,
            );
        } else {
            // Run in the jamb gap. Both legs enter the head, clear of the
            // moving bottom rail throughout the draw.
            let cord_x = if self.p.identity.is_multiple_of(2) {
                opening.max.x - 0.016
            } else {
                opening.min.x + 0.016
            };
            let y = l.top + 0.004;
            let low = y - (self.height() * 0.68).min(1.15);
            self.tube(
                self.p.fabric,
                &[
                    Vec3::new(cord_x, y, l.rail_z - 0.007),
                    Vec3::new(cord_x, low, l.rail_z - 0.007),
                    Vec3::new(cord_x, low - 0.005, l.rail_z - 0.004),
                    Vec3::new(cord_x, low - 0.005, l.rail_z + 0.004),
                    Vec3::new(cord_x, low, l.rail_z + 0.007),
                    Vec3::new(cord_x, y, l.rail_z + 0.007),
                ],
                0.0012,
            );
        }
    }
}

fn panel(g: &mut Geometry, cut: &Cut, l: PleatLayout, stock: PleatedStock) {
    let t = stock.cloth_thickness_m;
    let cellular = stock.construction == PleatConstruction::Cellular;
    let half = if cellular {
        stock.bond_width_m * 0.5
    } else {
        t * 0.5
    };
    let rear_crest = if cellular {
        -half - l.depth
    } else {
        -half + l.depth
    };
    let quads = l.count * if cellular { 22 } else { 8 } + 2;
    g.positions.reserve(4 * quads);
    g.normals.reserve(4 * quads);
    g.uvs.reserve(4 * quads);
    g.indices.reserve(6 * quads);
    for cell in 0..l.count {
        let row = 2 * cell;
        let levels = [row, row + 1, row + 2];
        let y = levels.map(|i| l.top - i as f32 * l.rise);
        let v = levels.map(|i| i as f32 * l.face_m);
        let outer = cut.ring(y, [-half, half], [rear_crest, half + l.depth]);
        cut.walls(
            g,
            &outer,
            v,
            2. * half,
            [cell == 0, cell + 1 == l.count],
            false,
        );
        if cellular {
            let inner = cut.ring(
                [y[0] - t * 0.5, y[1], y[2] + t * 0.5],
                [-half + t, half - t],
                [rear_crest + t, half + l.depth - t],
            );
            cut.walls(g, &inner, v, 2. * (half - t), [true, true], true);
            for i in 0..6 {
                let j = (i + 1) % 6;
                let edge = if i == 0 || i == 3 {
                    2. * half
                } else {
                    l.face_m
                };
                ends(g, [&outer[i], &outer[j], &inner[j], &inner[i]], edge, t);
            }
        } else {
            for [a, b, c, d] in [[0, 1, 2, 5], [5, 2, 3, 4]] {
                ends(g, [&outer[a], &outer[b], &outer[c], &outer[d]], t, l.face_m);
            }
        }
    }
}
