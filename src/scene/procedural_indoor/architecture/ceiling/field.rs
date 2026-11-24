//! A reflected ceiling plan at the finished room datum. The same bays drive
//! construction, recessed photometry, material bindings and capture metadata.
use super::*;
use crate::scene::procedural_indoor::{envelope::polygon, layout::IndoorLayout};
mod mesh;
mod services;
mod terminals;
#[cfg(test)]
mod tests;
pub(crate) use mesh::build;

const BEARING_M: f32 = 0.0036;
const POCKET_M: f32 = 0.095;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CeilingGrid {
    /// Nominal board dimensions, including its share of adjacent runner faces.
    pub pitch_m: Vec2,
    pub bays: UVec2,
    /// World-local x/z of the first interior T runner on each axis.
    pub first_joint_m: Vec2,
    pub tile: CeilingTile,
    pub troffer_depth_m: f32,
    pub troffers: Vec<RecessedLight>,
    pub air_terminals: Vec<AirTerminal>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecessedLight {
    /// Index on the existing ceiling circuit, shared by photometry and the lamp instance ID.
    pub fixture: usize,
    pub cell: UVec2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AirPattern {
    FourWaySupply,
    SlottedReturn,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AirTerminal {
    pub cell: UVec2,
    pub pattern: AirPattern,
    pub blades: u8,
    pub angle_rad: f32,
    pub across_x: bool,
}

impl CeilingGrid {
    fn eligible(scene: &IndoorManifest) -> bool {
        scene.program.is_some()
            && scene.envelope.as_ref().is_some_and(|e| {
                e.ceiling_drop == Vec2::ZERO && e.soffits.is_empty() && e.mezzanine.is_none()
            })
    }

    pub(super) fn sample(scene: &IndoorManifest) -> Option<Self> {
        if scene.generator_version < 158 || scene.program.is_none() || !Self::eligible(scene) {
            return None;
        }
        let mut r = stream(scene.seed, u64::from_le_bytes(*b"RoomGrid"));
        let frequency = match scene.layout {
            IndoorLayout::OpenOffice
            | IndoorLayout::Training
            | IndoorLayout::Classroom
            | IndoorLayout::Cafeteria
            | IndoorLayout::Conference
            | IndoorLayout::Library => 0.82,
            IndoorLayout::Gallery
            | IndoorLayout::Studio
            | IndoorLayout::Workshop
            | IndoorLayout::Kitchen => 0.22,
            _ => 0.54,
        };
        if !r.random_bool(frequency) {
            return None;
        }
        let module = [0.600, 0.6096, 0.625][r.random_range(0..3)];
        let mut pitch = Vec2::splat(module);
        if r.random_bool(0.52) {
            pitch[usize::from(r.random_bool(0.5))] *= 2.;
        }
        let size = scene.room_size.xz();
        let mut bays = (size / pitch).ceil().as_uvec2();
        // Large offices and halls can still use standard stock. A double-length
        // panel on the longer axis halves the bay count at the same detail level.
        if pitch.x == pitch.y && (bays.max_element() > 64 || bays.element_product() > 768) {
            pitch[usize::from(size.y > size.x)] *= 2.;
            bays = (size / pitch).ceil().as_uvec2();
        }
        if bays.min_element() < 3 || bays.max_element() > 64 || bays.element_product() > 768 {
            return None;
        }
        // Keep the two perimeter cuts between half a board and a full board.
        // A modest offset permits joint alignment without narrow slivers.
        let border = (size - (bays.as_vec2() - 2.) * pitch) * 0.5;
        let allowance = (border - pitch * 0.5).min(pitch - border).max(Vec2::ZERO);
        let first_joint_m = -size * 0.5
            + border
            + allowance * Vec2::new(r.random_range(-0.72..0.72), r.random_range(-0.72..0.72));
        let mut grid = Self {
            pitch_m: pitch,
            bays,
            first_joint_m,
            tile: CeilingTile::sample(scene.seed, r.random_range(0..12)),
            troffer_depth_m: r.random_range(0.038..0.076),
            troffers: Vec::new(),
            air_terminals: Vec::new(),
        };
        grid.fit_services(scene, &mut r);
        Some(grid)
    }

    fn boundary(&self, size: Vec2, axis: usize, n: u32) -> f32 {
        if n == 0 {
            -size[axis] * 0.5
        } else if n == self.bays[axis] {
            size[axis] * 0.5
        } else {
            self.first_joint_m[axis] + (n - 1) as f32 * self.pitch_m[axis]
        }
    }
    pub fn cell_bounds(&self, size: Vec2, cell: UVec2) -> (Vec2, Vec2) {
        let point = |n: UVec2| Vec2::new(self.boundary(size, 0, n.x), self.boundary(size, 1, n.y));
        (point(cell), point(cell + UVec2::ONE))
    }
    fn center(&self, cell: UVec2) -> Vec2 {
        // Fixtures occupy only complete interior bays.
        self.first_joint_m + (cell.as_vec2() - 0.5) * self.pitch_m
    }
    fn cells(&self) -> impl Iterator<Item = UVec2> + '_ {
        (0..self.bays.y).flat_map(move |y| (0..self.bays.x).map(move |x| UVec2::new(x, y)))
    }
    fn cell_at(&self, p: Vec2) -> UVec2 {
        (((p - self.first_joint_m) / self.pitch_m).floor() + 1.)
            .max(Vec2::ZERO)
            .as_uvec2()
            .min(self.bays - UVec2::ONE)
    }

    /// A surface-mounted fixture needs a continuous bearing face on one tile.
    /// Leave runners, bevels, air terminals and illuminated service bays clear.
    pub(crate) fn fixture_support(
        &self,
        scene: &IndoorManifest,
        p: Vec2,
        radius: f32,
    ) -> Option<f32> {
        let cell = self.cell_at(p);
        let (lo, hi) = self.cell_bounds(scene.room_size.xz(), cell);
        let margin =
            radius + self.tile.flange_width_m * 0.5 + self.tile.clearance_m + self.tile.drop_m;
        let inset = (p - lo).min(hi - p).min_element();
        (inset > margin
            && !self.troffers.iter().any(|t| t.cell == cell)
            && !self.air_terminals.iter().any(|t| t.cell == cell))
        .then(|| self.mount_height(scene, p))
    }

    /// Contact height on the finished face. On a cut border the board is flush.
    pub(crate) fn mount_height(&self, scene: &IndoorManifest, p: Vec2) -> f32 {
        let h = scene.room_size.y;
        let cell = self.cell_at(p);
        let (lo, hi) = self.cell_bounds(scene.room_size.xz(), cell);
        let dist = (p - lo).min(hi - p).min_element();
        let lip = self.tile.flange_width_m * 0.5;
        if dist <= lip || !self.profiled_bay(scene, lo, hi) {
            return h;
        }
        let neck = lip + self.tile.clearance_m;
        if self.tile.edge == TileEdge::BeveledTegular {
            let t = ((dist - neck) / (self.tile.drop_m * 0.65)).clamp(0., 1.);
            h + BEARING_M - self.tile.drop_m * t
        } else if dist > neck {
            h + BEARING_M - self.tile.drop_m
        } else {
            h + BEARING_M
        }
    }
    fn profiled_bay(&self, scene: &IndoorManifest, lo: Vec2, hi: Vec2) -> bool {
        polygon::box_inside(&scene.envelope.as_ref().unwrap().footprint, lo, hi, 0.0002)
    }
    pub(crate) fn air_reservations(&self, h: f32) -> impl Iterator<Item = (Vec3, Vec3)> + '_ {
        self.air_terminals.iter().map(move |v| {
            let c = self.center(v.cell);
            let (lo, hi) = (c - self.pitch_m * 0.5, c + self.pitch_m * 0.5);
            // Include the contact pad of a device fixed to the finished ceiling.
            (
                Vec3::new(lo.x, h - 0.018, lo.y),
                Vec3::new(hi.x, h + POCKET_M, hi.y),
            )
        })
    }
    pub(crate) fn for_scene(scene: &IndoorManifest) -> Option<&Self> {
        scene.ceiling_installation.as_ref()?.room_grid.as_ref()
    }
    pub(crate) fn lamp_size(&self) -> Vec3 {
        let span = self.pitch_m - self.tile.flange_width_m + 0.002;
        Vec3::new(span.x, self.troffer_depth_m, span.y)
    }
    pub(crate) fn fit_positions(&self, h: f32, positions: &mut [Vec3]) {
        for mount in &self.troffers {
            if let Some(p) = positions.get_mut(mount.fixture) {
                let c = self.center(mount.cell);
                *p = Vec3::new(c.x, h + self.troffer_depth_m * 0.5 - 0.004, c.y);
            }
        }
    }
    pub(crate) fn lamp_at(scene: &IndoorManifest, position: Vec3) -> Option<&Self> {
        Self::for_scene(scene).filter(|grid| grid.contains_lamp(position))
    }
    fn contains_lamp(&self, position: Vec3) -> bool {
        self.troffers
            .iter()
            .any(|m| self.center(m.cell) == position.xz())
    }

    pub(super) fn validate(&self, scene: &IndoorManifest) -> Result<(), String> {
        let size = scene.room_size.xz();
        if !Self::eligible(scene)
            || !self.tile.valid()
            || !self.pitch_m.is_finite()
            || !self.first_joint_m.is_finite()
            || !self.pitch_m.cmpge(Vec2::splat(0.59)).all()
            || !self.pitch_m.cmple(Vec2::splat(1.26)).all()
            || self.bays.min_element() < 3
            || self.bays.max_element() > 64
            || self.bays.element_product() > 768
            || !(0.035..=0.08).contains(&self.troffer_depth_m)
            || self.air_terminals.len() > 8
            || self.troffers.len() > self.bays.element_product() as usize
        {
            return Err("invalid acoustic ceiling grid".into());
        }
        let first = self.first_joint_m + size * 0.5;
        let last = size - first - (self.bays.as_vec2() - 2.) * self.pitch_m;
        if [first, last].iter().any(|b| {
            b.cmplt(self.pitch_m * 0.5 - 0.001).any() || b.cmpgt(self.pitch_m + 0.001).any()
        }) {
            return Err("unbalanced acoustic ceiling border".into());
        }
        self.validate_services(scene)
    }
}
