use super::*;

impl Foundation {
    pub(super) fn hardware(
        &self,
        a: &mut Assembly,
        label: &'static str,
        h: f32,
        s: &Section,
        cuts: &mut TimberCuts,
    ) {
        let [front, rear] = s.ends_z();
        let top = h + 0.010;
        let r = self.half_width;
        let pad = self.glide_height;
        let deck = pad + r * 0.83;
        match self.base {
            Base::Plinth => {
                let x0 = s.station_x(0);
                let x1 = s.station_x(s.intervals);
                let gauge = r.min((x1 - x0) * 0.12).min((rear - front) * 0.12);
                let g = a.part(self.stock, label);
                for z in [front, rear] {
                    cuts.cuboid(
                        g,
                        self.stock,
                        Vec3::new(x1 - x0 + gauge, top, gauge),
                        0.002,
                        Transform::from_xyz((x0 + x1) * 0.5, top * 0.5, z),
                        GrainAxis::X,
                    );
                }
                for x in [x0, x1] {
                    cuts.cuboid(
                        g,
                        self.stock,
                        Vec3::new(gauge, top, rear - front - gauge),
                        0.002,
                        Transform::from_xyz(x, top * 0.5, (front + rear) * 0.5),
                        GrainAxis::Z,
                    );
                }
                return;
            }
            Base::Pedestal => {
                let radius = ((s.right - s.left) * 0.44).min((s.back - s.front) * 0.39);
                let stem = r.min(radius * 0.30);
                a.part(self.stock, label).lathe(
                    &[
                        (0., 0.),
                        (radius, 0.),
                        (radius, pad),
                        (stem * 1.35, h * 0.45),
                        (stem, h * 0.55),
                        (stem, top),
                        (0., top),
                    ],
                    32,
                    Transform::from_translation(s.center()),
                );
                return;
            }
            Base::LengthwiseSled => {
                for z in [front, rear] {
                    structure::u_support(
                        a.part(self.stock, label),
                        Vec3::new(s.station_x(0), 0., z),
                        Vec3::new(s.station_x(s.intervals), 0., z),
                        [deck, top],
                        false,
                        r,
                    );
                }
            }
            _ => {}
        }
        for i in 0..=s.intervals {
            let x = s.station_x(i);
            let end = if i == 0 {
                -1.
            } else if i == s.intervals {
                1.
            } else {
                0.
            };
            if end == 0. && s.center_levelers {
                let pos = Vec3::new(x, 0., (front + rear) * 0.5);
                a.part(Surface::Metal, label)
                    .rod(pos.with_y(pad * 0.62), pos.with_y(top), 0.013);
                self.pad(a, label, pos, 0.026);
                continue;
            }
            if self.base == Base::CrossSled {
                structure::u_support(
                    a.part(self.stock, label),
                    Vec3::new(x, 0., front),
                    Vec3::new(x, 0., rear),
                    [deck, top],
                    false,
                    r,
                );
            }
            for (z, side) in [(front, -1.), (rear, 1.)] {
                let foot = Vec3::new(x, 0., z);
                let head = foot.with_y(top) - Vec3::new(end, 0., side) * self.rake;
                match self.base {
                    Base::CrossSled | Base::LengthwiseSled => {
                        if self.base == Base::LengthwiseSled && end == 0. {
                            a.part(self.stock, label).rod(
                                foot.with_y(deck),
                                foot.with_y(top),
                                r * 0.88,
                            );
                        }
                        // The pad sits on the straight portion, inside the return bend.
                        let shift = if self.base == Base::CrossSled {
                            Vec3::Z * side
                        } else {
                            Vec3::X * end
                        };
                        self.pad(a, label, foot - shift * 0.043, r * 1.65);
                    }
                    Base::Hairpin => {
                        let turn = r * 2.65;
                        let center = foot.with_y(pad + r * 0.76 + turn);
                        let mut path = Vec::with_capacity(11);
                        path.push(head + Vec3::Z * turn * 1.62);
                        for k in 0..=8 {
                            let t = PI * k as f32 / 8.;
                            path.push(center + Vec3::new(0., -turn * t.sin(), turn * t.cos()));
                        }
                        path.push(head - Vec3::Z * turn * 1.62);
                        a.part(self.stock, label).tube(&path, r, 8);
                        self.pad(a, label, foot, r * 2.5);
                    }
                    Base::Tapered | Base::Block => {
                        let tapered = self.base == Base::Tapered;
                        let bottom_r = if tapered { r * self.foot_ratio } else { r };
                        let head = if tapered { head } else { foot.with_y(top) };
                        let origin = cuts.origin();
                        cut_leg(
                            a.part(self.stock, label),
                            foot.with_y(pad * 0.86),
                            head,
                            Vec2::new(1., self.aspect),
                            [bottom_r, r],
                            self.stock.is_cut_timber().then_some(origin),
                        );
                        self.pad(a, label, foot, bottom_r.min(bottom_r * self.aspect) * 0.88);
                    }
                    Base::Pedestal | Base::Plinth => unreachable!(),
                }
            }
        }
    }

    fn pad(&self, a: &mut Assembly, label: &'static str, foot: Vec3, r: f32) {
        let h = self.glide_height;
        a.part(Surface::Rubber, label).lathe(
            &[(0., 0.), (r * 0.90, 0.), (r, h * 0.24), (r, h), (0., h)],
            12,
            Transform::from_translation(foot),
        );
    }
}

/// Horizontal cuts keep a raked leg flush with its glide and frame. The four
/// clipped corners are flat facets, with longitudinal grain on cut timber.
fn cut_leg(
    g: &mut Geometry,
    foot: Vec3,
    head: Vec3,
    aspect: Vec2,
    r: [f32; 2],
    origin: Option<Vec2>,
) {
    let shape = [
        (-0.78, -1.),
        (0.78, -1.),
        (1., -0.78),
        (1., 0.78),
        (0.78, 1.),
        (-0.78, 1.),
        (-1., 0.78),
        (-1., -0.78),
    ];
    let ring = |center: Vec3, radius: f32| {
        shape.map(|(x, z)| center + Vec3::new(x * aspect.x * radius, 0., z * aspect.y * radius))
    };
    let lo = ring(foot, r[0]);
    let hi = ring(head, r[1]);
    let length = (head - foot).normalize();
    for i in 0..8 {
        let j = (i + 1) % 8;
        let outward = ((lo[i] + lo[j]) * 0.5 - foot).normalize();
        g.stock_quad([lo[i], hi[i], hi[j], lo[j]], outward, length, origin);
    }
    // Three quads tessellate each octagonal cap without a zero-area fan.
    for (ring, n) in [(lo, -Vec3::Y), (hi, Vec3::Y)] {
        for k in [1, 3, 5] {
            g.metric_quad([ring[0], ring[k], ring[k + 1], ring[k + 2]], n);
        }
    }
}
