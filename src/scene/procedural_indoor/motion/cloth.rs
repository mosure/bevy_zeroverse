//! Material-point interpolation shared by render meshes and swept clearance.
use super::RollerFeed;
use bevy::prelude::*;

#[derive(Clone, Copy)]
pub(crate) enum ClothMotion {
    Roller(RollerFeed),
    Pleat {
        forward: Vec3,
        /// Vertical projection of a unit facet in the two endpoint poses.
        rise: [f32; 2],
    },
}

impl ClothMotion {
    pub fn applies(self, label: &str) -> bool {
        let tag = match self {
            Self::Roller(_) => "roller",
            Self::Pleat { .. } => "pleat",
        };
        label.split('#').any(|s| s == tag)
    }

    pub fn at(self, t: f32) -> ClothSample {
        match self {
            Self::Roller(feed) => ClothSample::Roller(feed),
            Self::Pleat {
                forward,
                rise: [a, b],
            } => {
                let s = a.lerp(b, t);
                let depth = |rise: f32| ((1. - rise) * (1. + rise)).max(0.).sqrt();
                // The rail travels linearly. The front/back crest moves on
                // a circle. Rationalize the depth difference near equal poses
                // and evaluate it once for the entire mesh.
                let q = t * ((s + a) / (b + a)) * ((depth(b) + depth(a)) / (depth(s) + depth(a)));
                ClothSample::Pleat {
                    forward,
                    correction: q - t,
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ClothSample {
    Roller(RollerFeed),
    Pleat { forward: Vec3, correction: f32 },
}

impl ClothSample {
    pub fn interpolate(self, a: Vec3, b: Vec3, t: f32) -> Vec3 {
        match self {
            Self::Roller(feed) => feed.interpolate(a, b, t),
            Self::Pleat {
                forward,
                correction,
            } => a.lerp(b, t) + forward * ((b - a).dot(forward) * correction),
        }
    }
}
