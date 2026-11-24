//! Metric wall inserts. Routing, mesh construction and dataset metrics use the
//! same mating faces. Archived plates retain their original duplex geometry.
use super::*;

mod mesh;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct OutletBank {
    pub face: OutletFace,
    pub gangs: usize,
    pub rows: usize,
    /// Plate bay covered by an unpowered filler module, ordered left to right.
    pub blank_slot: Option<usize>,
    pub usb: bool,
    pub switched: bool,
    pub case_size: Vec3,
    pub module_size: Vec2,
    pub socket_spacing_m: f32,
    /// Aperture dimensions use metres in a local XY plane facing +Z.
    pub contact_scale: f32,
    pub face_z_m: f32,
    pub well_depth_m: f32,
    pub finish: Surface,
    pub insert: Surface,
    pub screws: bool,
    edge_m: f32,
}

impl OutletBank {
    pub(super) fn fitted(o: &IndoorObject, p: &FixtureProgram) -> Option<Self> {
        if o.kind != ObjectKind::WallOutlet {
            return None;
        }
        let face = OutletFace::from_variant(o.variant)?;
        let mut stock = stream(o.seed, u64::from_le_bytes(*b"WallJack"));
        let mut finish = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"JackFace"),
        );
        let case_size = o.size * Vec3::new(p.case_fraction.x, p.case_fraction.y, 1.);
        let spare = p.gangs < 3 && case_size.x / (p.gangs + 1) as f32 >= 0.060;
        let slots = p.gangs + usize::from(spare);
        let blank_slot =
            spare.then(|| stream(o.seed, u64::from_le_bytes(*b"BlankBay")).random_range(0..slots));
        let pitch = case_size.x / slots as f32;
        let usb = p.style >= 3;
        let rows =
            if face == OutletFace::Polarized && case_size.y >= 0.074 && stock.random_bool(0.79) {
                2
            } else {
                1
            };
        let switched = face == OutletFace::ThreePin && stock.random_bool(0.72);
        let w = match face {
            OutletFace::Polarized => stock.random_range(0.034_f32..0.044),
            OutletFace::RoundWell => stock.random_range(0.048_f32..0.057),
            OutletFace::ThreePin => stock.random_range(0.047_f32..0.063),
        }
        .min(pitch * 0.84);
        let h = match face {
            OutletFace::Polarized if rows == 2 => stock.random_range(0.066_f32..0.081),
            OutletFace::Polarized => stock.random_range(0.047..0.063),
            OutletFace::RoundWell | OutletFace::ThreePin => {
                w + if usb || switched { 0.012 } else { 0. }
            }
        }
        .min(case_size.y * 0.80);
        let face_z_m = case_size.z * stock.random_range(0.27..0.36);
        let well_depth_m = if face == OutletFace::RoundWell {
            stock
                .random_range(0.006_f32..0.011)
                .min((face_z_m + case_size.z * 0.35) * 0.68)
        } else {
            0.
        };
        let row_pitch = if rows == 2 {
            h * stock.random_range(0.45..0.52)
        } else {
            0.
        };
        let useful_h = if rows == 2 {
            h - row_pitch
        } else if usb || switched {
            h * 0.70
        } else {
            h
        };
        let contact_scale = match face {
            OutletFace::Polarized => (w / 0.034).min(useful_h / 0.034).min(1.04),
            OutletFace::RoundWell => (w / 0.049).min(useful_h / 0.045).min(1.10),
            OutletFace::ThreePin => (w / 0.046).min(useful_h / 0.036).min(1.09),
        };
        let insert = if finish.random_bool(0.16) {
            Surface::Ceramic
        } else {
            Surface::Plastic
        };
        Some(Self {
            face,
            gangs: p.gangs,
            rows,
            blank_slot,
            usb,
            switched,
            case_size,
            module_size: Vec2::new(w, h),
            socket_spacing_m: row_pitch,
            contact_scale,
            face_z_m,
            well_depth_m,
            finish: p.finish,
            insert,
            screws: p.screws,
            edge_m: p.edge_radius.min(case_size.min_element() * 0.14),
        })
    }

    pub fn socket_count(&self) -> usize {
        self.gangs * self.rows
    }

    pub fn plate_slots(&self) -> usize {
        self.gangs + usize::from(self.blank_slot.is_some())
    }

    fn slot_center(&self, slot: usize) -> Vec2 {
        let count = self.plate_slots();
        let pitch = self.case_size.x / count as f32;
        Vec2::new(
            (slot as f32 - (count - 1) as f32 * 0.5) * pitch,
            self.case_size.y * 0.5,
        )
    }

    fn gang_center(&self, gang: usize) -> Vec2 {
        self.slot_center(gang + usize::from(self.blank_slot.is_some_and(|slot| gang >= slot)))
    }

    pub fn socket_local(&self, index: usize) -> Option<Vec3> {
        if index >= self.socket_count() {
            return None;
        }
        let mut center = self.gang_center(index / self.rows);
        let y = if self.rows == 2 {
            ((index % self.rows) as f32 - 0.5) * self.socket_spacing_m
        } else if self.usb || self.switched {
            -self.module_size.y * 0.13
        } else {
            0.
        };
        center.y += y;
        Some(center.extend(self.face_z_m - self.well_depth_m))
    }

    pub(super) fn build(&self, a: &mut Assembly, label: &str) {
        mesh::build(self, a, label);
    }
}
