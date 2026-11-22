//! Fit complete service bays to the existing power circuit. Adjacent room
//! fixtures and equipment without a compatible aperture keep their own mounts.
use super::*;
use crate::scene::procedural_indoor::{
    architecture::{fixture_layout, fixtures},
    floorplan,
};

fn clear(p: Vec2, half: Vec2, occupied: &[(Vec2, Vec2)], skip: Option<usize>) -> bool {
    occupied
        .iter()
        .enumerate()
        .all(|(j, (q, r))| Some(j) == skip || (p - *q).abs().cmpge(half + *r + 0.10).any())
}

impl CeilingGrid {
    pub(super) fn full_bay(
        &self,
        scene: &IndoorManifest,
        cell: UVec2,
        obstacles: &[(Vec3, Vec3)],
    ) -> bool {
        if cell.min_element() == 0 || (cell + UVec2::ONE).cmpge(self.bays).any() {
            return false;
        }
        let (lo, hi) = self.cell_bounds(scene.room_size.xz(), cell);
        let env = scene.envelope.as_ref().unwrap();
        polygon::box_inside(&env.footprint, lo, hi, 0.035)
            && !obstacles.iter().any(|&(a, b)| {
                b.y >= scene.room_size.y - 0.16
                    && lo.cmplt(b.xz() + 0.035).all()
                    && hi.cmpgt(a.xz() - 0.035).all()
            })
    }
    pub(super) fn fit_services(&mut self, scene: &IndoorManifest, r: &mut impl Rng) {
        let e = scene.envelope.as_ref().unwrap();
        let mut obstacles = floorplan::obstacles(scene);
        obstacles.extend(e.structural_boxes(scene.room_size));
        // Use the unsnapped circuit even when reconstructing an existing manifest.
        let mounts = fixture_layout::positions(scene);
        let mut occupied: Vec<_> = mounts
            .iter()
            .map(|&p| {
                let half = fixture_layout::half(scene, p).xz();
                (p.xz(), half)
            })
            .collect();
        let mut candidates: Vec<_> = self
            .cells()
            .filter(|&c| self.full_bay(scene, c, &obstacles))
            .collect();
        if matches!(
            fixtures::parameters(scene).shape,
            fixtures::FixtureShape::Panel | fixtures::FixtureShape::Louver
        ) {
            let zones = &scene.program.as_ref().unwrap().zones;
            for (fixture, &p) in mounts.iter().enumerate() {
                if !polygon::contains(&e.footprint, p.xz(), 0.) {
                    continue;
                }
                let zone = zones
                    .iter()
                    .find(|z| p.xz().cmpge(z.min).all() && p.xz().cmple(z.max).all());
                let max_move = self.pitch_m.length() * 0.91;
                let best = candidates
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &c)| {
                        let q = self.center(c);
                        let d = q.distance_squared(p.xz());
                        (d <= max_move * max_move
                            && zone.is_none_or(|z| q.cmpge(z.min).all() && q.cmple(z.max).all())
                            && clear(q, self.pitch_m * 0.5, &occupied, Some(fixture)))
                        .then_some((i, d))
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1));
                if let Some((i, _)) = best {
                    let cell = candidates.swap_remove(i);
                    occupied[fixture] = (self.center(cell), self.pitch_m * 0.5);
                    self.troffers.push(RecessedLight { fixture, cell });
                }
            }
        }
        use rand::seq::SliceRandom;
        candidates.shuffle(r);
        let count = (polygon::area(&e.footprint) / r.random_range(21.0..45.))
            .round()
            .clamp(1., 8.) as usize;
        let return_offset = usize::from(r.random_bool(0.5));
        for cell in candidates {
            let p = self.center(cell);
            if !clear(p, self.pitch_m * 0.5, &occupied, None)
                || self
                    .air_terminals
                    .iter()
                    .any(|v| self.center(v.cell).distance_squared(p) < 1.7_f32.powi(2))
            {
                continue;
            }
            let pattern = if (self.air_terminals.len() + return_offset) % 3 == 1 {
                AirPattern::SlottedReturn
            } else {
                AirPattern::FourWaySupply
            };
            self.air_terminals.push(AirTerminal {
                cell,
                pattern,
                blades: r.random_range(3..=5),
                angle_rad: r.random_range(0.36..0.79),
                across_x: r.random_bool(0.5),
            });
            if self.air_terminals.len() == count {
                break;
            }
        }
    }

    pub(super) fn validate_services(&self, scene: &IndoorManifest) -> Result<(), String> {
        let mut obstacles = floorplan::obstacles(scene);
        obstacles.extend(
            scene
                .envelope
                .as_ref()
                .unwrap()
                .structural_boxes(scene.room_size),
        );
        let mut taken = std::collections::BTreeSet::new();
        for c in self
            .troffers
            .iter()
            .map(|m| m.cell)
            .chain(self.air_terminals.iter().map(|v| v.cell))
        {
            if c.cmpge(self.bays).any()
                || !taken.insert((c.x, c.y))
                || !self.full_bay(scene, c, &obstacles)
            {
                return Err("invalid acoustic ceiling service bay".into());
            }
        }
        if !self.troffers.is_empty()
            && !matches!(
                fixtures::parameters(scene).shape,
                fixtures::FixtureShape::Panel | fixtures::FixtureShape::Louver
            )
        {
            return Err("recessed mount belongs to an incompatible circuit".into());
        }
        let raw = fixture_layout::positions(scene);
        let mut indices = std::collections::BTreeSet::new();
        for m in &self.troffers {
            if !indices.insert(m.fixture)
                || raw.get(m.fixture).is_none_or(|p| {
                    !polygon::contains(&scene.envelope.as_ref().unwrap().footprint, p.xz(), 0.)
                        || p.xz().distance(self.center(m.cell)) > self.pitch_m.length() * 0.92
                })
            {
                return Err("invalid recessed luminaire mount".into());
            }
        }
        if self
            .air_terminals
            .iter()
            .any(|v| !(3..=5).contains(&v.blades) || !(0.35..=0.80).contains(&v.angle_rad))
        {
            return Err("invalid ceiling air terminal".into());
        }
        let occupied: Vec<_> = raw
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                self.troffers.iter().find(|m| m.fixture == i).map_or_else(
                    || (p.xz(), fixture_layout::half(scene, p).xz()),
                    |m| (self.center(m.cell), self.pitch_m * 0.5),
                )
            })
            .collect();
        for m in &self.troffers {
            if !clear(
                self.center(m.cell),
                self.pitch_m * 0.5,
                &occupied,
                Some(m.fixture),
            ) {
                return Err("recessed luminaire overlaps another fixture".into());
            }
        }
        for (i, v) in self.air_terminals.iter().enumerate() {
            let p = self.center(v.cell);
            if !clear(p, self.pitch_m * 0.5, &occupied, None)
                || self.air_terminals[..i]
                    .iter()
                    .any(|a| self.center(a.cell).distance_squared(p) < 1.7_f32.powi(2))
            {
                return Err("ceiling grille overlaps another service".into());
            }
        }
        Ok(())
    }
}
