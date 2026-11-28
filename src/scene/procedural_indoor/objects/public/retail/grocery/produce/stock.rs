use super::*;

impl ProduceBin {
    pub(super) fn load(&mut self, lot: &[Fruit; 3], fill: f32, max_y: f32, r: &mut impl Rng) {
        // Depletion is local to a bin. Neighbouring fruit often touch, with room for
        // isolated pieces, full cases and the occasional empty crate awaiting refill.
        let density = match r.random_range(0..100) {
            0..=3 => return,
            4..=18 => fill * r.random_range(0.12..0.44),
            _ => (fill * r.random_range(0.86..1.56)).min(1.35),
        };
        let half = self.size_xz_m * 0.5;
        let half = match self.style {
            // A rectangle inside the eight-power superellipse at the narrowest tray floor.
            CrateStyle::FormedTray => (half * 0.94 - Vec2::splat(self.style.wall_m())) * 0.914,
            CrateStyle::TimberSlats => half - Vec2::splat(self.style.wall_m() + 0.015),
            CrateStyle::VentedTote => half - Vec2::splat(self.style.wall_m() + 0.002),
        };
        let area = 4. * half.element_product();
        let target = (area * density / 0.010).round().clamp(1., 36.) as usize;
        self.stock.reserve(target);
        let radii = lot.each_ref().map(|p| p.clearance_radius() + 0.001);
        for _ in 0..target {
            let shape = r.random_range(0..3);
            let radius = radii[shape];
            let limit = half - Vec2::splat(radius);
            if limit.min_element() <= 0. || self.floor_m() + lot[shape].total_height() > max_y {
                continue;
            }
            for _ in 0..8 {
                let q = if !self.stock.is_empty() && r.random_bool(0.55) {
                    let near = &self.stock[r.random_range(0..self.stock.len())];
                    near.position_xz_m
                        + Vec2::from_angle(r.random_range(-PI..PI))
                            * (radii[near.shape] + radius + r.random_range(0.001..0.018))
                } else {
                    Vec2::new(
                        r.random_range(-limit.x..limit.x),
                        r.random_range(-limit.y..limit.y),
                    )
                };
                if q.abs().cmple(limit).all()
                    && self.stock.iter().all(|p| {
                        q.distance_squared(p.position_xz_m) >= (radius + radii[p.shape]).powi(2)
                    })
                {
                    self.stock.push(ProducePiece {
                        position_xz_m: q,
                        shape,
                        yaw: r.random_range(-PI..PI),
                    });
                    break;
                }
            }
        }
    }
}

impl ProduceStand {
    pub(super) fn build_stock(&self, a: &mut Assembly) {
        let mut facings: Vec<[_; 3]> = (0..self.lots.len())
            .map(|_| std::array::from_fn(|_| Vec::new()))
            .collect();
        for bin in &self.bins {
            for piece in &bin.stock {
                let xz = bin.base_m.xz() + piece.position_xz_m;
                let (sin, cos) = piece.yaw.sin_cos();
                facings[bin.lot][piece.shape].push((
                    Vec3::new(xz.x, bin.floor_m(), xz.y),
                    sin,
                    cos,
                ));
            }
        }
        for (samples, shapes) in self.lots.iter().zip(facings) {
            for (fruit, instances) in samples.iter().zip(shapes) {
                if instances.is_empty() {
                    continue;
                }
                let mut template = Assembly::default();
                fruit.build(&mut template, 0.);
                append_stock(a, template, &instances);
            }
        }
    }
}
