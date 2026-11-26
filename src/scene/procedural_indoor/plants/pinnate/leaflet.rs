use super::*;

pub(super) struct Blade {
    pub root: Vec3,
    pub delta: Vec3,
    pub normal: Vec3,
    pub arch: f32,
    pub half_width: f32,
    pub fold: f32,
    pub twist: f32,
    pub distal_side: f32,
}

impl Blade {
    pub(super) fn write(&self, g: &mut Geometry, kind: Kind) {
        // Each row specifies the midrib position and widths towards/away from
        // the frond tip. Fern pinnae have a basal ear on the distal margin.
        const PALM: [[f32; 3]; 4] = [
            [0.12, 0.55, 0.55],
            [0.39, 1., 1.],
            [0.68, 0.83, 0.83],
            [0.86, 0.45, 0.45],
        ];
        const FERN: [[f32; 3]; 6] = [
            [0.12, 1.12, 0.66],
            [0.23, 0.79, 0.91],
            [0.38, 1.01, 1.],
            [0.51, 0.76, 0.77],
            [0.65, 0.80, 0.70],
            [0.82, 0.44, 0.40],
        ];
        const PALM_TRIS: [u32; 48] = leaf::strip_indices();
        const FERN_TRIS: [u32; 72] = leaf::strip_indices();
        match kind {
            Kind::Palm => self.strip::<14>(g, &PALM, &PALM_TRIS),
            Kind::Fern => self.strip::<20>(g, &FERN, &FERN_TRIS),
        }
    }

    fn strip<const V: usize>(&self, g: &mut Geometry, rows: &[[f32; 3]], indices: &[u32]) {
        let axis = self.delta.normalize();
        let side = axis.cross(self.normal).normalize();
        let up = side.cross(axis);
        let mut p = [Vec3::ZERO; V];
        let mut uv = [[0.; 2]; V];
        p[0] = self.root;
        uv[0] = [0.5, 0.];
        for (i, &[t, distal, proximal]) in rows.iter().enumerate() {
            let mid = self.root + self.delta * t + Vec3::Y * (self.arch * 4. * t * (1. - t));
            let (left, right) = if self.distal_side > 0. {
                (proximal, distal)
            } else {
                (distal, proximal)
            };
            for (j, width) in [-left, 0., right].into_iter().enumerate() {
                let k = 1 + i * 3 + j;
                p[k] = mid
                    + self.half_width
                        * (side * width + up * (self.twist * width * t - self.fold * width.abs()));
                uv[k] = [0.5 + width * 0.44, t];
            }
        }
        p[V - 1] = self.root + self.delta;
        uv[V - 1] = [0.5, 1.];
        leaf::append_patch(g, p, uv, indices);
    }
}
