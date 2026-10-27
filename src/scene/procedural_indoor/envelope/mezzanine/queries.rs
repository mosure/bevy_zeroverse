use super::*;

impl Mezzanine {
    pub fn ascending_positive(&self) -> bool {
        self.construction
            .as_ref()
            .is_some_and(|c| c.ascending_positive)
    }
    pub fn riser_height(&self) -> f32 {
        self.deck.height / self.steps as f32
    }
    pub fn stair_width(&self) -> f32 {
        self.stair_max[1 - self.stair_axis] - self.stair_min[1 - self.stair_axis]
    }
    pub fn tread_depth(&self) -> f32 {
        let extra = self
            .construction
            .as_ref()
            .and_then(|c| c.landing.as_ref())
            .map_or(0., |l| l.extra_run);
        (self.stair_max[self.stair_axis] - self.stair_min[self.stair_axis] - extra)
            / self.steps as f32
    }
    fn point(&self, distance: f32, side: f32, height: f32) -> Vec3 {
        let mut p = self.stair_min;
        p[self.stair_axis] = if self.ascending_positive() {
            self.stair_min[self.stair_axis] + distance
        } else {
            self.stair_max[self.stair_axis] - distance
        };
        p[1 - self.stair_axis] = side;
        Vec3::new(p.x, height, p.y)
    }
    /// Conservative stair support/reservation prisms, in ascending order.
    /// Open-riser rendering uses their top surfaces, not their filled volumes.
    pub fn treads(&self) -> impl Iterator<Item = (Vec3, Vec3)> + '_ {
        let tread = self.tread_depth();
        let landing = self.construction.as_ref().and_then(|c| c.landing.as_ref());
        (0..self.steps).map(move |i| {
            let extra_before = landing
                .filter(|l| i >= l.after_riser)
                .map_or(0., |l| l.extra_run);
            let extra_after = landing
                .filter(|l| i + 1 >= l.after_riser)
                .map_or(0., |l| l.extra_run);
            let begin = i as f32 * tread + extra_before;
            let end = (i + 1) as f32 * tread + extra_after;
            let mut lo = self.stair_min;
            let mut hi = self.stair_max;
            if self.ascending_positive() {
                lo[self.stair_axis] = self.stair_min[self.stair_axis] + begin;
                hi[self.stair_axis] = self.stair_min[self.stair_axis] + end;
            } else {
                lo[self.stair_axis] = self.stair_max[self.stair_axis] - end;
                hi[self.stair_axis] = self.stair_max[self.stair_axis] - begin;
            }
            (
                Vec3::new(lo.x, 0., lo.y),
                Vec3::new(
                    hi.x,
                    self.deck.height * (i + 1) as f32 / self.steps as f32,
                    hi.y,
                ),
            )
        })
    }
    pub fn stair_boxes(&self) -> Vec<(Vec3, Vec3)> {
        self.treads().collect()
    }
    pub fn stair_highest_in(&self, lo: Vec2, hi: Vec2) -> Option<f32> {
        self.treads()
            .filter(|(a, b)| lo.cmplt(b.xz()).all() && hi.cmpgt(a.xz()).all())
            .map(|(_, b)| b.y)
            .max_by(f32::total_cmp)
    }
    pub(super) fn stair_support(&self, p: Vec3) -> f32 {
        self.treads()
            .filter(|(a, b)| {
                p.xz().cmpge(a.xz() - Vec2::splat(1e-5)).all()
                    && p.xz().cmple(b.xz() + Vec2::splat(1e-5)).all()
            })
            .map(|(_, b)| b.y)
            .fold(0., f32::max)
    }
    /// Slab plus primary frame, for underside fixture clearance.
    pub fn underside_height(&self) -> f32 {
        self.deck.height - self.thickness - self.construction.as_ref().map_or(0., |c| c.beam_depth)
    }
    pub(crate) fn columns(&self) -> Vec<(Vec3, Vec3)> {
        let Some(c) = &self.construction else {
            let mut out = Vec::new();
            for x in [self.deck.min.x + 0.12, self.deck.max.x - 0.12] {
                for z in [self.deck.min.y + 0.12, self.deck.max.y - 0.12] {
                    out.push((
                        Vec3::new(x - 0.065, 0., z - 0.065),
                        Vec3::new(x + 0.065, self.deck.height - self.thickness, z + 0.065),
                    ));
                }
            }
            return out;
        };
        let axis =
            usize::from((self.deck.max - self.deck.min).y > (self.deck.max - self.deck.min).x);
        let lo = self.deck.min + Vec2::splat(c.support_inset);
        let hi = self.deck.max - Vec2::splat(c.support_inset);
        let count = ((hi[axis] - lo[axis]) / c.support_pitch)
            .ceil()
            .clamp(1., 3.) as usize;
        let half = Vec2::new(c.support_width, c.support_depth) * 0.5;
        let mut out = Vec::new();
        for i in 0..=count {
            for side in [lo[1 - axis], hi[1 - axis]] {
                let mut p = lo.lerp(hi, i as f32 / count as f32);
                p[1 - axis] = side;
                out.push((
                    Vec3::new(p.x - half.x, 0., p.y - half.y),
                    Vec3::new(p.x + half.x, self.underside_height(), p.y + half.y),
                ));
            }
        }
        out
    }
    pub(in super::super) fn beams(&self) -> Vec<(Vec3, Vec3)> {
        let Some(c) = &self.construction else {
            return vec![];
        };
        let columns = self.columns();
        let mut out = Vec::new();
        let top = self.deck.height - self.thickness;
        for pair in columns.as_chunks::<2>().0 {
            let lo = pair[0].0.min(pair[1].0).with_y(top - c.beam_depth);
            let hi = pair[0].1.max(pair[1].1).with_y(top);
            out.push((lo, hi));
        }
        // Ties butt into cross-beams instead of overlapping their undersides.
        // This also keeps texture coordinates from competing on coplanar faces.
        let axis = if (self.deck.max - self.deck.min).y > (self.deck.max - self.deck.min).x {
            2
        } else {
            0
        };
        for bay in 0..columns.len() / 2 - 1 {
            for side in 0..2 {
                let left = columns[bay * 2 + side];
                let right = columns[(bay + 1) * 2 + side];
                let mut lo = left.0.with_y(top - c.beam_depth);
                let mut hi = right.1.with_y(top);
                lo[axis] = left.1[axis];
                hi[axis] = right.0[axis];
                out.push((lo, hi));
            }
        }
        out
    }
    /// Intermediate landings on framed flights get grounded cross-supports;
    /// they must not hang between the two inclined stringers without a frame.
    pub(super) fn landing_frame(&self) -> Option<LandingFrame> {
        let c = self.construction.as_ref()?;
        let l = c.landing.as_ref()?;
        if c.stair_structure == StairStructure::Solid {
            return None;
        }
        let half = Vec2::new(c.support_width, c.support_depth) * 0.5;
        let cross = 1 - self.stair_axis;
        let distance = l.after_riser as f32 * self.tread_depth() + l.extra_run * 0.5;
        let top = self.deck.height * l.after_riser as f32 / self.steps as f32 - c.tread_thickness;
        let columns = [
            self.stair_min[cross] + half[cross] + 0.02,
            self.stair_max[cross] - half[cross] - 0.02,
        ]
        .map(|side| {
            let p = self.point(distance, side, 0.);
            (
                p - Vec3::new(half.x, 0., half.y),
                p + Vec3::new(half.x, top - c.beam_depth, half.y),
            )
        });
        let mut beam = (
            columns[0].0.min(columns[1].0).with_y(top - c.beam_depth),
            columns[0].1.max(columns[1].1).with_y(top),
        );
        // Butt into the two side stringers; overlapping their horizontal
        // undersides would reintroduce competing coplanar material samples.
        let component = if cross == 0 { 0 } else { 2 };
        beam.0[component] = self.stair_min[cross] + 0.1 + 0.0325;
        beam.1[component] = self.stair_max[cross] - 0.1 - 0.0325;
        Some(LandingFrame { columns, beam })
    }
    pub(super) fn rail_spans(&self) -> Vec<RailSpan> {
        if self.construction.is_none() {
            let rails = self.legacy_rails();
            let count = rails.len();
            return rails
                .into_iter()
                .enumerate()
                .map(|(i, (start, end))| RailSpan {
                    start,
                    end,
                    stair: i + 2 >= count,
                })
                .collect();
        }
        let mut segments = Vec::new();
        for axis in 0..2 {
            for high in [false, true] {
                let other = 1 - axis;
                let coordinate = if high {
                    self.deck.max[axis]
                } else {
                    self.deck.min[axis]
                };
                let spans = if axis == self.stair_axis && high != self.ascending_positive() {
                    vec![
                        (self.deck.min[other], self.stair_min[other]),
                        (self.stair_max[other], self.deck.max[other]),
                    ]
                } else {
                    vec![(self.deck.min[other], self.deck.max[other])]
                };
                for (lo, hi) in spans {
                    if hi - lo < 0.035 {
                        continue;
                    }
                    let mut a = Vec2::ZERO;
                    a[axis] = coordinate;
                    a[other] = lo;
                    let mut b = a;
                    b[other] = hi;
                    segments.push(RailSpan {
                        start: Vec3::new(a.x, self.deck.height, a.y),
                        end: Vec3::new(b.x, self.deck.height, b.y),
                        stair: false,
                    });
                }
            }
        }
        let run = self.stair_max[self.stair_axis] - self.stair_min[self.stair_axis];
        for side in [
            self.stair_min[1 - self.stair_axis],
            self.stair_max[1 - self.stair_axis],
        ] {
            let mut points = vec![self.point(0., side, 0.)];
            if let Some(l) = self.construction.as_ref().and_then(|c| c.landing.as_ref()) {
                let d = l.after_riser as f32 * self.tread_depth();
                let h = self.deck.height * l.after_riser as f32 / self.steps as f32;
                points.push(self.point(d, side, h));
                points.push(self.point(d + l.extra_run, side, h));
            }
            points.push(self.point(run, side, self.deck.height));
            for pq in points.windows(2) {
                segments.push(RailSpan {
                    start: pq[0],
                    end: pq[1],
                    stair: true,
                });
            }
        }
        segments
    }
    pub fn rails(&self) -> Vec<(Vec3, Vec3)> {
        if self.construction.is_none() {
            return self.legacy_rails();
        }
        self.rail_spans()
            .into_iter()
            .map(|p| (p.start, p.end))
            .collect()
    }
    fn legacy_rails(&self) -> Vec<(Vec3, Vec3)> {
        let mut segments = Vec::new();
        for axis in 0..2 {
            for high in [false, true] {
                let other = 1 - axis;
                let coordinate = if high {
                    self.deck.max[axis]
                } else {
                    self.deck.min[axis]
                };
                let spans = if axis == self.stair_axis && high {
                    vec![
                        (self.deck.min[other], self.stair_min[other]),
                        (self.stair_max[other], self.deck.max[other]),
                    ]
                } else {
                    vec![(self.deck.min[other], self.deck.max[other])]
                };
                for (lo, hi) in spans {
                    if hi - lo < 0.03 {
                        continue;
                    }
                    let mut a = Vec2::ZERO;
                    let mut b = a;
                    a[axis] = coordinate;
                    b[axis] = coordinate;
                    a[other] = lo;
                    b[other] = hi;
                    segments.push((
                        Vec3::new(a.x, self.deck.height, a.y),
                        Vec3::new(b.x, self.deck.height, b.y),
                    ));
                }
            }
        }
        // Two sloping handrails meet the deck's guarded landing.
        let other = 1 - self.stair_axis;
        for side in [self.stair_min[other], self.stair_max[other]] {
            let mut a = self.stair_min;
            let mut b = self.stair_max;
            a[other] = side;
            b[other] = side;
            segments.push((
                Vec3::new(a.x, self.deck.height, a.y),
                Vec3::new(b.x, 0., b.y),
            ));
        }
        segments
    }
}
