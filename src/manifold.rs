use bevy::prelude::*;
use strum_macros::EnumIter;


#[derive(Clone, Debug, EnumIter, Reflect)]
pub enum ManifoldOperations {
    Union,
    Intersection,
    Difference,
    SymmetricDifference,
}

// TODO: wait for https://github.com/bevyengine/bevy/issues/13790
