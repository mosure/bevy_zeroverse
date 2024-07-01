use bevy::prelude::*;
use strum_macros::EnumIter;


#[derive(Clone, Debug, EnumIter, Reflect)]
pub enum ManifoldOperations {
    Union,
    Intersection,
    Difference,
    SymmetricDifference,
}
