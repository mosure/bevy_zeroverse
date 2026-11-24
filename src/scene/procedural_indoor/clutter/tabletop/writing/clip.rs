//! One formed spring clip with a buried attachment and a rounded return.
use super::*;

impl WritingProgram {
    pub(crate) fn clip_rings(&self) -> [[Vec3; 4]; 5] {
        let r = self.radius;
        let (root, tip) = if self.mechanism == 2 {
            (self.cap_length() * 0.20, self.cap_length() * 0.87)
        } else {
            (
                self.length * (self.tail_fraction + 0.025),
                self.length * 0.38,
            )
        };
        let path = [
            (0.91, root),
            (1.35, root + r * 0.52),
            (1.46, root + r * 1.28),
            (1.34, tip - r * 0.9),
            (1.06, tip),
        ];
        path.map(|(at, y)| {
            let x = -at * r;
            [
                Vec3::new(x - r * 0.07, y, -r * 0.22),
                Vec3::new(x - r * 0.07, y, r * 0.22),
                Vec3::new(x + r * 0.07, y, r * 0.22),
                Vec3::new(x + r * 0.07, y, -r * 0.22),
            ]
        })
    }

    pub(super) fn draw_clip(&self, g: &mut Geometry, tf: Transform) {
        let sections = self
            .clip_rings()
            .map(|ring| ring.map(|v| tf.transform_point(v)));
        for pair in sections.windows(2) {
            let [a, b] = [pair[0], pair[1]];
            let middle = (a[0] + a[2] + b[0] + b[2]) * 0.25;
            for i in 0..4 {
                let j = (i + 1) % 4;
                let face = [a[i], a[j], b[j], b[i]];
                g.metric_quad(face, (a[i] + a[j] + b[i] + b[j]) * 0.25 - middle);
            }
        }
        // The root terminates inside the barrel.
        g.metric_quad(sections[4], tf.rotation * Vec3::Y);
    }
}
