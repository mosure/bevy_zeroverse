//! Loose objects on fitted occasional-table shelves. Every item remains an
//! ordinary room instance with its own transform, support, mask and bounds.
use super::*;
use crate::scene::procedural_indoor::{
    objects::{literature, tables},
    placement::SurfacePattern,
};

const INTRODUCED: u32 = 271;
const CONTACT_M: f32 = 0.002;
const PLACEMENT_MARGIN_M: f32 = 0.012;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportPlane {
    Top,
    LowerShelf,
}

impl SupportPlane {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Top => "top",
            Self::LowerShelf => "lower_shelf",
        }
    }
}

impl IndoorManifest {
    /// Resolve a loose object's contact level from the manifest's support ID,
    /// metric transform and generator version. Routed cables have no flat base.
    pub fn support_plane(&self, object: &IndoorObject) -> Option<SupportPlane> {
        let parent = self.objects.get(object.support?)?;
        if object.kind == ObjectKind::Cable || !parent.kind.is_surface() {
            return None;
        }
        let local_y = object.position.y - parent.position.y;
        if (local_y - parent.size.y).abs() <= CONTACT_M {
            Some(SupportPlane::Top)
        } else {
            self.lower_shelf_contact(object, parent)
                .map(|_| SupportPlane::LowerShelf)
        }
    }

    pub(super) fn has_lower_contact_candidate(
        &self,
        o: &IndoorObject,
        parent: &IndoorObject,
    ) -> bool {
        self.generator_version >= INTRODUCED
            && matches!(parent.kind, ObjectKind::CoffeeTable | ObjectKind::SideTable)
            && o.position.y - parent.position.y < parent.size.y - CONTACT_M
    }

    pub(super) fn lower_shelf_contact(
        &self,
        o: &IndoorObject,
        parent: &IndoorObject,
    ) -> Option<tables::LoungeShelf> {
        if !self.has_lower_contact_candidate(o, parent) {
            return None;
        }
        let deck = tables::parameters(parent).lounge_shelf(parent, self.generator_version)?;
        ((o.position.y - parent.position.y - deck.top_y_m).abs() <= CONTACT_M).then_some(deck)
    }

    /// Tucked seats and the legs of settled occupants may enter the lower bay.
    /// The shared prop gate also checks other shelf contents and routed wires.
    pub(super) fn shelf_access_clear(
        &self,
        o: &IndoorObject,
        table: &IndoorObject,
        margin: f32,
    ) -> bool {
        let (min, max) = o.bounds();
        let overlaps = |(lo, hi): (Vec3, Vec3)| {
            (min - Vec3::splat(margin)).cmplt(hi).all()
                && (max + Vec3::splat(margin)).cmpgt(lo).all()
        };
        !self.objects.iter().any(|other| {
            other.id != table.id
                && other.id != o.id
                && other.neighbor == o.neighbor
                && other.support != Some(table.id)
                && other.kind != ObjectKind::Cable
                && overlaps(other.bounds())
        }) && !self
            .humans
            .iter()
            .any(|h| h.neighbor == o.neighbor && overlaps(h.bounds()))
    }

    pub(super) fn furnish_lounge_shelves(&mut self) {
        if self.generator_version < INTRODUCED || self.clutter.selection_probability() == 0. {
            return;
        }
        for index in 0..self.objects.len() {
            let table = &self.objects[index];
            if table.neighbor
                || !matches!(table.kind, ObjectKind::CoffeeTable | ObjectKind::SideTable)
            {
                continue;
            }
            let amount = self.clutter.sample(table.seed);
            let mut use_rng = stream(table.seed, 0x5348_454c_4655_5345);
            if amount == 0.
                || !use_rng
                    .random_bool(((0.28 + 0.66 * amount) * (0.68 + 0.32 * self.density)) as f64)
            {
                continue;
            }
            let Some(deck) = tables::parameters(table).lounge_shelf(table, self.generator_version)
            else {
                continue;
            };
            let space = (deck.width_m * deck.length_m / 0.24).min(1.);
            let count = 1
                + usize::from(use_rng.random_bool((amount * space * 0.85) as f64))
                + usize::from(use_rng.random_bool((amount * amount * space * 0.35) as f64));
            let pattern = SurfacePattern::sample(&mut use_rng);
            let table = table.clone();
            for slot in 0..count {
                let mut r = stream(table.seed, 0x5348_454c_4649_544d + slot as u64);
                self.place_shelf_item(&table, deck, pattern, &mut r);
            }
        }
    }

    fn place_shelf_item(
        &mut self,
        table: &IndoorObject,
        deck: tables::LoungeShelf,
        pattern: SurfacePattern,
        r: &mut ChaCha8Rng,
    ) {
        let box_chance = match self.layout {
            IndoorLayout::Workshop | IndoorLayout::Studio => 0.50,
            IndoorLayout::Library | IndoorLayout::Reception => 0.13,
            _ => 0.30,
        };
        let kind = if r.random_bool(box_chance) {
            ObjectKind::StorageBox
        } else if r.random_bool(0.24) {
            ObjectKind::Notepad
        } else {
            ObjectKind::Books
        };
        let single_book = kind == ObjectKind::Books && r.random_bool(0.51);
        let bounds = deck.storage_bounds();
        let available = bounds.1 - bounds.0 - 2. * PLACEMENT_MARGIN_M;
        let (width, height, aspect) = match kind {
            ObjectKind::StorageBox => (0.11..0.28, 0.055..0.17, 0.72..1.39),
            ObjectKind::Books if single_book => (0.10..0.24, 0.011..0.041, 1.10..1.55),
            ObjectKind::Books => (0.12..0.29, 0.028..0.16, 1.05..1.45),
            ObjectKind::Notepad => (0.085..0.225, 0.006..0.033, 1.12..1.55),
            _ => return,
        };
        // Select among physically sized products that fit this compartment.
        // A shallow shelf can hold a paperback without shrinking a file box.
        let aspect: f32 = r.random_range(aspect);
        let max_width = available.min_element() / aspect.min(1.);
        let max_width = max_width
            .min(available.max_element() / aspect.max(1.))
            .min(width.end);
        let max_height = deck.clear_height(PLACEMENT_MARGIN_M).min(height.end);
        if width.start >= max_width || height.start >= max_height {
            return;
        }
        let w = r.random_range(width.start..max_width);
        let size = Vec3::new(w, r.random_range(height.start..max_height), w * aspect);
        if !deck.carries(kind, size, PLACEMENT_MARGIN_M) {
            return;
        }
        let mut item = self.candidate(kind, Vec3::ZERO, size, 0., r);
        if single_book {
            item.variant |= literature::SINGLE_VOLUME;
        }
        item.solid = false;
        item.support = Some(table.id);
        let tf = table.transform();
        for _ in 0..6 {
            let yaw = pattern.yaw(r);
            let Some(place) = pattern.propose(bounds, size.xz(), yaw, PLACEMENT_MARGIN_M, r) else {
                continue;
            };
            item.yaw = table.yaw + yaw;
            item.position =
                tf.transform_point(Vec3::new(place.center.x, deck.top_y_m, place.center.y));
            if self.prop_clear_on(&item, table, PLACEMENT_MARGIN_M, PropSurface::Shelf(deck)) {
                self.objects.push(item);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests;
