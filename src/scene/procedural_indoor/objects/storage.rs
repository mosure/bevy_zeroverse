//! Compartment programs shared by cabinet geometry, contents and motion.
use super::*;
pub mod bookcase;
mod cabinet;
mod fronts;
mod program;
pub use program::{parameters, CabinetBay, ShelfProgram};

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    if o.kind == ObjectKind::Bookcase {
        bookcase::build(a, o);
    } else {
        cabinet::build(a, o);
    }
}

#[cfg(test)]
mod tests;
