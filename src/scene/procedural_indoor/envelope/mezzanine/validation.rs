use super::*;

impl Mezzanine {
    pub fn validate(&self, envelope: &EnvelopeProgram, size: Vec3) -> Result<(), String> {
        if self.stair_axis > 1
            || !(1..=64).contains(&self.steps)
            || !self.deck.height.is_finite()
            || !self.deck.min.is_finite()
            || !self.deck.max.is_finite()
            || !self.stair_min.is_finite()
            || !self.stair_max.is_finite()
            || self.stair_max.cmple(self.stair_min).any()
            || self.deck.max.cmple(self.deck.min).any()
            || !(0.12..=0.35).contains(&self.thickness)
            || !(1.0..=1.25).contains(&self.rail_height)
        {
            return Err("invalid mezzanine dimensions".into());
        }
        if let Some(c) = &self.construction {
            c.validate(self)?;
            let axis = self.stair_axis;
            let cross = 1 - axis;
            let deck_edge = if c.ascending_positive {
                self.deck.min[axis]
            } else {
                self.deck.max[axis]
            };
            let stair_edge = if c.ascending_positive {
                self.stair_max[axis]
            } else {
                self.stair_min[axis]
            };
            if (deck_edge - stair_edge).abs() > 1e-4
                || self.stair_min[cross] < self.deck.min[cross] + 0.08
                || self.stair_max[cross] > self.deck.max[cross] - 0.08
            {
                return Err("stair does not join the guarded deck opening".into());
            }
        }
        if !polygon::box_inside(&envelope.footprint, self.deck.min, self.deck.max, 0.1)
            || !polygon::box_inside(&envelope.footprint, self.stair_min, self.stair_max, 0.1)
            || !(0.12..=0.18).contains(&self.riser_height())
            || !(0.27..=0.36).contains(&self.tread_depth())
            || envelope.minimum_ceiling_height(size, self.deck.min, self.deck.max)
                - self.deck.height
                < 2.2
            || self.treads().any(|(lo, hi)| {
                envelope.minimum_ceiling_height(size, lo.xz(), hi.xz()) - hi.y < 2.2
            })
        {
            return Err("invalid mezzanine, stair or headroom".into());
        }
        Ok(())
    }
}
