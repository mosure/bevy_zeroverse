use bevy::prelude::*;


#[derive(Clone, Debug, Reflect)]
pub enum ManifoldOperations {
    Union,
    Intersection,
    Difference,
    SymmetricDifference,
}
