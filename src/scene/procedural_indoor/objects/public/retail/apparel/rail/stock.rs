use super::*;

impl ApparelRail {
    pub(super) fn dress_arm(
        &mut self,
        o: &IndoorObject,
        arm: &StockArm,
        spots: &[f32],
        cross_space: f32,
        length: f32,
    ) {
        let bay = self.arms.len();
        let mut r = stream(o.seed, u64::from_le_bytes(*b"ShopLine") + bay as u64);
        let side = self.form == RailForm::SideHang;
        let direction = if side {
            if r.random_bool(0.5) {
                0.
            } else {
                PI
            }
        } else {
            -FRAC_PI_2
        };
        let mut line = self.collections.len();
        for (i, &along) in spots.iter().enumerate() {
            if i == 0 || (i > 1 && r.random_bool(0.33)) {
                line = self.collections.len();
                self.collections
                    .push(collection(line, cross_space, length, side, &mut r));
            }
            let garment = &self.collections[line].cut;
            let size = r
                .random_range(0.92_f32..1.065)
                .min((arm.point(along).y - 0.19) / (garment.length_m + 0.014));
            let nearest = |j: usize| (spots[j] - along).abs() * 0.5 - 0.003;
            let mut clearance =
                (along - 0.026).min(arm.run() - along + if side { -0.026 } else { 0.011 });
            if i > 0 {
                clearance = clearance.min(nearest(i - 1));
            }
            if i + 1 < spots.len() {
                clearance = clearance.min(nearest(i + 1));
            }
            // Solve the projected horizontal footprint around the loaded hook.
            // Include the flared dress hem, lapels and fasteners in the bound.
            let wide = garment.width_m.max(garment.shoulder_width_m * 1.26) * 0.5 * size;
            let thick = (garment.half_depth_m + 0.011) * size;
            let angle = ((clearance - thick) / wide)
                .clamp(0., 1.)
                .asin()
                .min(if side { 0.48 } else { 0.25 });
            let yaw = direction + r.random_range(-1.0..1.0) * angle;
            self.hangers.push(HungItem {
                arm: bay,
                collection: line,
                along_m: along,
                yaw_rad: yaw,
                size_grade: size,
            });
        }
    }
}

fn collection(
    line: usize,
    cross_space: f32,
    length: f32,
    side: bool,
    r: &mut ChaCha8Rng,
) -> ApparelCollection {
    let kind = match r.random_range(0..10) {
        0..=4 => GarmentKind::Shirt,
        5..=7 => GarmentKind::Dress,
        _ if length > 0.8 => GarmentKind::Overcoat,
        _ => GarmentKind::Jacket,
    };
    // Reserve the hook offset and allow the largest size in each range to
    // swivel without entering the next arm's bay or the edge of the fixture.
    let limit = (cross_space - 0.080) / 1.065;
    let shoulder = r.random_range(0.34_f32..0.45).min(limit / 1.27);
    let (length_m, tail_fraction) = match kind {
        GarmentKind::Shirt => (length.clamp(0.55, 0.76), r.random_range(0.36..0.54)),
        GarmentKind::Dress => (
            (length + 0.21).clamp(0.86, 1.26),
            r.random_range(0.38..0.57),
        ),
        _ => (length, r.random_range(0.62..0.93)),
    };
    let cut = Garment {
        kind,
        support_index: line,
        anchor_m: Vec3::ZERO,
        support_axis: Vec3::Z,
        support_radius_m: 0.006,
        shoulder_width_m: shoulder,
        width_m: (shoulder + r.random_range(0.062..0.13)).min(limit),
        length_m,
        half_depth_m: if side {
            r.random_range(0.020..0.029)
        } else {
            r.random_range(0.015..0.021)
        },
        tail_fraction,
        fold_phase: r.random_range(-PI..PI),
        fold_count: r.random_range(2..=4),
        closure: r.random_range(0..=1),
        surface: if kind == GarmentKind::Jacket && r.random_bool(0.30) {
            Surface::Leather
        } else if r.random_bool(0.5) {
            Surface::FabricAlt
        } else {
            Surface::Fabric
        },
        finish_slot: garment_stock::sample_slot(r),
    };
    let sleeve_finish_slot = (cut.surface != Surface::Leather
        && r.random_bool(match kind {
            GarmentKind::Shirt => 0.34,
            GarmentKind::Jacket => 0.23,
            _ => 0.10,
        }))
    .then(|| garment_stock::contrasting_slot(cut.finish_slot, r));
    ApparelCollection {
        cut,
        sleeve_finish_slot,
    }
}
