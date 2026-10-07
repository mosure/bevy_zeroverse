//! A single support program can emit rendering geometry or cheap collision solids.
use super::*;

pub(super) trait PartSink {
    fn cuboid(&mut self, size: Vec3, bevel: f32, tf: Transform);
    fn cylinder(&mut self, radius: f32, height: f32, tf: Transform);
    fn lathe(&mut self, profile: &[(f32, f32)], segments: u32, tf: Transform);
    fn rod(&mut self, a: Vec3, b: Vec3, radius: f32);
    fn tube(&mut self, path: &[Vec3], radius: f32, sides: u32);
}
pub(super) trait StructureSink {
    type Part: PartSink;
    fn part(&mut self, surface: Surface, label: &str) -> &mut Self::Part;
    fn box_part(&mut self, surface: Surface, label: &str, pos: Vec3, size: Vec3, bevel: f32) {
        self.part(surface, label)
            .cuboid(size, bevel, Transform::from_translation(pos));
    }
}
impl StructureSink for Assembly {
    type Part = Geometry;
    fn part(&mut self, surface: Surface, label: &str) -> &mut Geometry {
        Assembly::part(self, surface, label)
    }
}
impl PartSink for Geometry {
    fn cuboid(&mut self, size: Vec3, bevel: f32, tf: Transform) {
        Geometry::cuboid(self, size, bevel, tf);
    }
    fn cylinder(&mut self, r: f32, h: f32, tf: Transform) {
        Geometry::cylinder(self, r, h, tf);
    }
    fn lathe(&mut self, p: &[(f32, f32)], n: u32, tf: Transform) {
        Geometry::lathe(self, p, n, tf);
    }
    fn rod(&mut self, a: Vec3, b: Vec3, r: f32) {
        Geometry::rod(self, a, b, r);
    }
    fn tube(&mut self, p: &[Vec3], r: f32, n: u32) {
        Geometry::tube(self, p, r, n);
    }
}
