//! Finished paper sizes used for physical placement, in metres.
use bevy::prelude::Vec2;

pub(crate) type Stock = (&'static str, Vec2);
pub(crate) const A4: Stock = ("A4", Vec2::new(0.210, 0.297));
pub(crate) const LETTER: Stock = ("Letter", Vec2::new(0.2159, 0.2794));
pub(crate) const A5: Stock = ("A5", Vec2::new(0.148, 0.210));
pub(crate) const PHOTO: Stock = ("4x6", Vec2::new(0.1016, 0.1524));
pub(crate) const A6: Stock = ("A6", Vec2::new(0.105, 0.148));
pub(crate) const INDEX: Stock = ("3x5", Vec2::new(0.0762, 0.127));
pub(crate) const NOTE: Stock = ("square memo", Vec2::splat(0.0762));
pub(crate) const SMALL_NOTE: Stock = ("small memo", Vec2::new(0.038, 0.051));
