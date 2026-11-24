//! The exterior skin, metal lip, inner walls and socket back share the same
//! aperture loop. In a grazing view the near edge can occlude the contacts.
use super::*;

impl Port {
    // Start at the right edge of the top-right quadrant in the (Z,Y) plane.
    // Three points per quadrant give a rounded USB-C mouth, chamfered Type-A
    // stock, a keyed HDMI mouth or a polygonal audio bore.
    fn outline(self) -> [Vec2; 12] {
        let half = self.connector.dimensions().0 * 0.5;
        let center = self.center_m.zy();
        if self.connector == Connector::Audio {
            return std::array::from_fn(|i| {
                let u = Vec2::from_angle((i as f32 + 0.5) * TAU / 12.);
                center + half * u
            });
        }
        let r = match self.connector {
            Connector::UsbC => half.y * 0.94,
            Connector::Hdmi => 0.00035,
            _ => 0.00045,
        };
        let mut ring = palmrest::rounded_outline(half, r, Vec2::ZERO);
        for pt in &mut ring {
            if self.connector == Connector::Hdmi && pt.y < 0. {
                pt.x *= 1. + 0.20 * pt.y / half.y;
            }
            *pt += center;
        }
        ring
    }

    fn at(self, pt: Vec2, inset: f32) -> Vec3 {
        Vec3::new(
            self.center_m.x - self.center_m.x.signum() * inset,
            pt.y,
            pt.x,
        )
    }

    fn pocket(self, a: &mut Assembly, label: &str, plastic_label: &str) {
        let mouth = self.outline();
        let center = self.center_m.zy();
        let (span, depth) = self.connector.dimensions();
        let throat =
            mouth.map(|p| center + (p - center) * (Vec2::ONE - Vec2::splat(0.00042) / span));
        let inward = |i: usize, j: usize| {
            let v = center - (throat[i] + throat[j]) * 0.5;
            Vec3::new(0., v.y, v.x)
        };
        let lip = a.part(Surface::Chrome, label);
        for i in 0..mouth.len() {
            let j = (i + 1) % mouth.len();
            lip.metric_quad(
                [
                    self.at(mouth[i], 0.),
                    self.at(mouth[j], 0.),
                    self.at(throat[j], 0.00036),
                    self.at(throat[i], 0.00036),
                ],
                inward(i, j),
            );
        }
        let well = a.part(Surface::Ink, label);
        for i in 0..throat.len() {
            let j = (i + 1) % throat.len();
            well.metric_quad(
                [
                    self.at(throat[i], 0.00036),
                    self.at(throat[j], 0.00036),
                    self.at(throat[j], depth),
                    self.at(throat[i], depth),
                ],
                inward(i, j),
            );
            triangle(
                well,
                self.center_m.x - self.center_m.x.signum() * depth,
                self.center_m.x.signum(),
                [center, throat[i], throat[j]],
            );
        }
        if self.connector != Connector::Audio {
            let setback = if self.connector == Connector::UsbC {
                0.00095
            } else {
                0.0017
            };
            let length = depth - setback + 0.0002;
            let offset = if self.connector == Connector::UsbA {
                span.y * 0.20
            } else {
                0.
            };
            let height = if self.connector == Connector::UsbA {
                0.0012
            } else {
                0.00068
            };
            // Only the leading corners guide a mating plug. The rear of the
            // molded insert is buried in the backing. Stamp it from the same
            // dielectric stock as the pad to keep a single material batch.
            let x = length * 0.5;
            let z = span.x * 0.345;
            let cut = 0.00045;
            let outline = [
                Vec2::new(x, -z + cut),
                Vec2::new(x, z - cut),
                Vec2::new(x - cut, z),
                Vec2::new(-x, z),
                Vec2::new(-x, -z),
                Vec2::new(x - cut, -z),
            ];
            let mut tf = Transform::from_translation(
                self.at(center + Vec2::Y * offset, setback + length * 0.5),
            );
            if self.center_m.x < 0. {
                tf.rotation = Quat::from_xyzw(0., 1., 0., 0.);
            }
            a.part(Surface::Plastic, plastic_label)
                .profile_slab(&outline, height, 0.00012, tf);
        }
    }
}

// Plane coordinates use metric Z along the device edge and Y above its support.
fn triangle(g: &mut Geometry, x: f32, side: f32, mut p: [Vec2; 3]) {
    if (p[1] - p[0]).perp_dot(p[2] - p[0]) * side > 0. {
        p.swap(1, 2);
    }
    let start = g.positions.len() as u32;
    for q in p {
        g.positions.push([x, q.y, q.x]);
        g.uvs.push(q.to_array());
        g.normals.push([side, 0., 0.]);
    }
    g.indices.extend([start, start + 1, start + 2]);
}

fn corners(lo: Vec2, hi: Vec2) -> [Vec2; 4] {
    [hi, Vec2::new(lo.x, hi.y), lo, Vec2::new(hi.x, lo.y)]
}

impl Bank {
    pub(in super::super) fn deck_edges(
        &self,
        g: &mut Geometry,
        outer: &[Vec2; 12],
        height: f32,
        profile: (f32, f32),
        tf: &Transform,
    ) {
        // Between rounded corners 5→6 and 11→0 are the two continuous flat sides.
        let has_left = self.ports.iter().any(|p| p.center_m.x < 0.);
        let has_right = self.ports.iter().any(|p| p.center_m.x > 0.);
        g.slab_edge_sections(
            outer,
            height,
            profile,
            tf,
            |_, d, rings| rings.map(|(y, _)| Vec2::new(d, y)),
            |i| i == 5 && has_left || i == 11 && has_right,
        );
        for (edge, side, has_ports) in [(5, -1., has_left), (11, 1., has_right)] {
            if !has_ports {
                continue;
            }
            let a = outer[edge];
            let b = outer[(edge + 1) % 12];
            let lo = Vec2::new(
                a.y.min(b.y) + tf.translation.z,
                tf.translation.y + (-height * 0.5 + profile.0),
            );
            let hi = Vec2::new(
                a.y.max(b.y) + tf.translation.z,
                tf.translation.y + (height * 0.5 - profile.0),
            );
            self.pierced_side(g, a.x, side, lo, hi);
        }
    }

    fn pierced_side(&self, g: &mut Geometry, x: f32, side: f32, lo: Vec2, hi: Vec2) {
        let mut ports: Vec<_> = self
            .ports
            .iter()
            .filter(|p| p.center_m.x.signum() == side)
            .collect();
        ports.sort_unstable_by(|a, b| a.center_m.z.total_cmp(&b.center_m.z));
        // Keep outer edges whole for a watertight join to the bevels. Fan the
        // thin top/bottom strips into the individual socket cells on the inside.
        let inset = Vec2::new(0.001, 0.00012);
        let outer = corners(lo, hi);
        let inner = corners(lo + inset, hi - inset);
        let mut tri = |a, b, c| triangle(g, x, side, [a, b, c]);
        for i in [0, 2] {
            let j = (i + 3) % 4;
            tri(outer[i], outer[j], inner[j]);
            tri(outer[i], inner[j], inner[i]);
        }
        let mut left = lo.x + inset.x;
        for (i, p) in ports.iter().enumerate() {
            let right = if let Some(next) = ports.get(i + 1) {
                // Bisect the free metal between mouths of unequal widths.
                ((p.center_m.z + p.connector.dimensions().0.x * 0.5)
                    + (next.center_m.z - next.connector.dimensions().0.x * 0.5))
                    * 0.5
            } else {
                hi.x - inset.x
            };
            let cell = corners(Vec2::new(left, inner[2].y), Vec2::new(right, inner[0].y));
            tri(outer[1], cell[1], cell[0]);
            tri(outer[2], cell[2], cell[3]);
            let hole = p.outline();
            for j in 0..4 {
                let k = j * 3;
                tri(cell[j], hole[k], hole[k + 1]);
                tri(cell[j], hole[k + 1], hole[k + 2]);
                let next = (j + 1) % 4;
                tri(cell[j], hole[k + 2], hole[next * 3]);
                tri(cell[j], hole[next * 3], cell[next]);
            }
            left = right;
        }
        tri(outer[1], inner[0], outer[0]);
        tri(outer[2], inner[3], outer[3]);
    }

    pub(in super::super) fn pockets(&self, a: &mut Assembly, label: &str, plastic_label: &str) {
        for port in &self.ports {
            port.pocket(a, label, plastic_label);
        }
    }
}
