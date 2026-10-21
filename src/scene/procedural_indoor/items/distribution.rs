//! A point mass at empty plus a continuous, bounded per-table clutter prior.
//! Sampling is keyed to the support, so all furnishing passes agree on its use.
use super::*;
use serde::Deserialize;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClutterSettings {
    /// Fraction of tables with loose belongings; remaining tables stay clear.
    pub probability: f32,
    /// Normalized amount on selected tables, before physical admission.
    pub minimum: f32,
    pub maximum: f32,
    /// -1 prefers lightly used tables; +1 prefers busy tables.
    pub bias: f32,
    /// 0 fixes the amount at the median; 1 spans the configured interval.
    pub spread: f32,
}
impl Default for ClutterSettings {
    fn default() -> Self {
        Self {
            probability: 0.75,
            minimum: 0.05,
            maximum: 1.,
            bias: -0.15,
            spread: 1.,
        }
    }
}
impl ClutterSettings {
    pub fn parse(json: &str) -> Result<Self, String> {
        let value: Self = serde_json::from_str(json).map_err(|e| format!("indoor_clutter: {e}"))?;
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.).contains(&self.probability)
            || !(0.0..=1.).contains(&self.minimum)
            || !(self.minimum..=1.).contains(&self.maximum)
            || !(-1.0..=1.).contains(&self.bias)
            || !(0.0..=1.).contains(&self.spread)
        {
            return Err("indoor_clutter requires probability/spread in [0,1], 0 <= minimum <= maximum <= 1, and bias in [-1,1]".into());
        }
        Ok(())
    }
    /// Conditional quantile, excluding the separate probability of an empty table.
    pub fn quantile(&self, q: f32) -> f32 {
        assert!((0.0..=1.).contains(&q));
        let center = 0.5 + 0.49 * self.bias as f64;
        let t = if self.spread == 0. {
            center
        } else if q == 0. || q == 1. {
            q as f64
        } else {
            let logit = (center / (1. - center)).ln()
                + self.spread as f64 * ((q as f64) / (1. - q as f64)).ln();
            1. / (1. + (-logit).exp())
        };
        self.minimum + (self.maximum - self.minimum) * t as f32
    }
    /// Proposal probability, including the zero-amount policy's empty tables.
    pub fn selection_probability(&self) -> f32 {
        if self.maximum > 0. {
            self.probability
        } else {
            0.
        }
    }
    pub fn active(&self, seed: u64) -> bool {
        let probability = self.selection_probability();
        probability > 0. && (probability == 1. || stream(seed, 5512).random::<f32>() < probability)
    }
    pub fn sample(&self, seed: u64) -> f32 {
        if !self.active(seed) {
            return 0.;
        }
        self.quantile(stream(seed, 5513).random())
    }
}

pub(crate) fn is_table(kind: ObjectKind) -> bool {
    matches!(
        kind,
        ObjectKind::Desk | ObjectKind::Table | ObjectKind::CoffeeTable | ObjectKind::SideTable
    )
}

impl IndoorManifest {
    pub(crate) fn permits_table_item(&self, support: &IndoorObject, kind: ObjectKind) -> bool {
        // Installed workstation equipment is functional, not loose clutter.
        !is_table(support.kind)
            || (support.kind == ObjectKind::Desk && kind == ObjectKind::Laptop)
            || matches!(
                kind,
                ObjectKind::Monitor
                    | ObjectKind::Keyboard
                    | ObjectKind::Mouse
                    | ObjectKind::Cable
                    | ObjectKind::Projector
            )
            || self.clutter.active(support.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_replayable_mixture_has_clear_light_and_busy_tables() {
        let settings = ClutterSettings::default();
        let samples: Vec<_> = (0..4096).map(|s| settings.sample(s)).collect();
        assert_eq!(
            samples,
            (0..4096).map(|s| settings.sample(s)).collect::<Vec<_>>()
        );
        let empty = samples.iter().filter(|&&x| x == 0.).count();
        assert!((900..1150).contains(&empty), "{empty}");
        assert!(samples.iter().filter(|&&x| x > 0. && x < 0.25).count() > 500);
        assert!(samples.iter().filter(|&&x| x > 0.75).count() > 400);
        let zero_amount = ClutterSettings {
            minimum: 0.,
            maximum: 0.,
            probability: 1.,
            ..settings
        };
        assert_eq!(zero_amount.selection_probability(), 0.);
        assert!((0..128).all(|seed| !zero_amount.active(seed) && zero_amount.sample(seed) == 0.));
        for probability in [0., 1.] {
            let p = ClutterSettings {
                probability,
                spread: 0.,
                ..settings
            };
            for seed in 0..128 {
                assert_eq!(
                    p.sample(seed),
                    if probability == 0. {
                        0.
                    } else {
                        p.quantile(0.5)
                    }
                );
            }
        }
        for bias in [-1., 0., 1.] {
            let p = ClutterSettings { bias, ..settings };
            let mut last = 0.;
            for i in 0..=1000 {
                let x = p.quantile(i as f32 / 1000.);
                assert!(x >= last && x >= p.minimum && x <= p.maximum);
                last = x;
            }
        }
        for json in [
            r#"{"probability":1.1}"#,
            r#"{"minimum":0.8,"maximum":0.2}"#,
            r#"{"spread":-1}"#,
            r#"{"typo":0}"#,
        ] {
            assert!(ClutterSettings::parse(json).is_err());
        }
    }
}
