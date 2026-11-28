//! Compartment programs shared by cabinet geometry, contents and motion.
use super::*;
pub mod bookcase;
mod cabinet;
mod fronts;
pub mod library_stack;
pub(crate) use fronts::hinged;
mod program;
pub use program::{parameters, CabinetBay, ShelfProgram};

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    match o.kind {
        ObjectKind::Bookcase => bookcase::build(a, o, version),
        ObjectKind::LibraryStack => library_stack::build(a, o, version),
        ObjectKind::Cabinet => cabinet::build(a, o, version),
        _ => unreachable!("storage construction routed a non-storage object"),
    }
}

#[cfg(test)]
mod tests;
