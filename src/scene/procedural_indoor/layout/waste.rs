//! Optional, clearance-tested commercial waste/recycling pairs. Sampling is
//! local to the existing bin, so unsuccessful stations do not resample furniture.
use super::super::objects::containers::bins;
use super::*;

impl IndoorManifest {
    pub(super) fn furnish_waste_stations(&mut self) {
        let bins: Vec<_> = self
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::TrashCan && !o.neighbor)
            .cloned()
            .collect();
        for original in bins.into_iter().take(3) {
            let mut r = stream(original.seed, 0x5741535445504149);
            let probability = if self.layout.is_public()
                || matches!(
                    self.layout,
                    IndoorLayout::OpenOffice
                        | IndoorLayout::Conference
                        | IndoorLayout::Breakroom
                        | IndoorLayout::Coworking
                        | IndoorLayout::Kitchen
                        | IndoorLayout::Kitchenette
                ) {
                0.62
            } else {
                0.28
            };
            if !r.random_bool(probability) {
                continue;
            }
            let first_side = if r.random_bool(0.5) { 1. } else { -1. };
            // The sampled product and spacing stay fixed across admission
            // attempts. Only its side/location changes; no shrink-to-fit bias.
            let gap = r.random_range(0.065..0.18);
            let stagger = r.random_range(-0.10..0.10) * original.size.z;
            for side in [first_side, -first_side] {
                let offset = original.transform().rotation
                    * Vec3::new(side * (original.size.x + gap), 0., stagger);
                let mut partner = original.clone();
                partner.id = self.objects.len();
                partner.position += offset;
                partner.variant = bins::WASTE_STATION | bins::RECYCLING;
                partner.interaction_target = Some(original.id);
                if self.placement_clear(&partner, 0.03)
                    && self.floor_support_clear(partner.bounds().0, partner.bounds().1)
                {
                    // Shared product geometry/finish, distinct role and aperture.
                    self.objects[original.id].variant = bins::WASTE_STATION;
                    self.objects.push(partner);
                    break;
                }
            }
        }
    }
}
