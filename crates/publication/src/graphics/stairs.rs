//! Manifest consumer for plan/section diagrams; honours archived stair programs.
use super::*;

pub(super) fn treads(m: &Value) -> Vec<(f64, f64, f64)> {
    let axis = m["stair_axis"].as_u64().unwrap() as usize;
    let steps = m["steps"].as_u64().unwrap();
    let (min, max) = (number(&m["stair_min"][axis]), number(&m["stair_max"][axis]));
    let height = number(&m["deck"]["height"]);
    let c = &m["construction"];
    let positive = c["ascending_positive"].as_bool().unwrap_or(false);
    let landing = c["landing"]["after_riser"].as_u64();
    let extra = if landing.is_some() {
        number(&c["landing"]["extra_run"])
    } else {
        0.
    };
    let tread = (max - min - extra) / steps as f64;
    (0..steps)
        .map(|i| {
            let start = i as f64 * tread
                + if landing.is_some_and(|l| i >= l) {
                    extra
                } else {
                    0.
                };
            let end = (i + 1) as f64 * tread
                + if landing.is_some_and(|l| i + 1 >= l) {
                    extra
                } else {
                    0.
                };
            let project = |v| if positive { min + v } else { max - v };
            (
                project(start),
                project(end),
                height * (i + 1) as f64 / steps as f64,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn publication_treads_respect_ascent_landings_axes_and_legacy_records() {
        for axis in 0..2 {
            for positive in [false, true] {
                for landed in [false, true] {
                    let mut m = json!({"stair_axis":axis,"steps":10,"stair_min":[-3.,-3.],"stair_max":[1.,1.],"deck":{"height":1.6},
                "construction":{"ascending_positive":positive,"landing":if landed {json!({"after_riser":4,"extra_run":1.})} else {Value::Null}}});
                    let slices = treads(&m);
                    assert_eq!(slices.len(), 10);
                    assert_eq!(slices[0].0, if positive { -3. } else { 1. });
                    assert_eq!(slices.last().unwrap().1, if positive { 1. } else { -3. });
                    assert_eq!(slices.last().unwrap().2, 1.6);
                    for (i, (a, b, y)) in slices.iter().enumerate() {
                        let width = if landed && i == 3 {
                            1.3
                        } else if landed {
                            0.3
                        } else {
                            0.4
                        };
                        assert!(((b - a).abs() - width).abs() < 1e-10);
                        assert_eq!(b > a, positive);
                        if i > 0 {
                            assert!((a - slices[i - 1].1).abs() < 1e-10);
                            assert!((y - slices[i - 1].2 - 0.16).abs() < 1e-10);
                        }
                    }
                    m.as_object_mut().unwrap().remove("construction");
                    assert_eq!(treads(&m)[0], (1., 0.6, 0.16));
                }
            }
        }
    }
}
