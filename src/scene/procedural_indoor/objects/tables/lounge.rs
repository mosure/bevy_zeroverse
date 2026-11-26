//! Two-tier occasional tables. The posts fit the installed top and the deck
//! bears on two end rails. All dimensions are in the table's local metres.
use super::*;
#[cfg(test)]
mod tests;

const INTRODUCED: u32 = 270;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ShelfDeck {
    TimberPanel,
    SpacedSlats,
    FoldedTray,
}

impl ShelfDeck {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::TimberPanel => "timber_panel",
            Self::SpacedSlats => "spaced_slats",
            Self::FoldedTray => "folded_tray",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct LoungeShelf {
    pub deck: ShelfDeck,
    pub frame: Surface,
    pub center_xz_m: Vec2,
    pub post_centers_half_span_m: Vec2,
    pub post_half_width_m: f32,
    pub round_posts: bool,
    pub runs_along_x: bool,
    pub length_m: f32,
    pub width_m: f32,
    pub top_y_m: f32,
    /// Bottom of the tabletop backing, above the storage compartment.
    pub underside_y_m: f32,
    pub thickness_m: f32,
    pub rail_height_m: f32,
    pub rim_height_m: f32,
    pub boards: usize,
    pub gap_m: f32,
}

impl TableProgram {
    /// Installed shelf and fitted dimensions, using the manifest's generator version.
    pub fn lounge_shelf(&self, o: &IndoorObject, version: u32) -> Option<LoungeShelf> {
        LoungeShelf::fit(o, self, version, || {
            outline::sample(o.size.xz() * 0.5, self)
        })
    }
}

impl LoungeShelf {
    pub(super) fn fit<P: AsRef<[Vec2]>>(
        o: &IndoorObject,
        p: &TableProgram,
        version: u32,
        outline: impl FnOnce() -> P,
    ) -> Option<Self> {
        if version < INTRODUCED
            || !matches!(o.kind, ObjectKind::CoffeeTable | ObjectKind::SideTable)
            || o.variant == SHARED_ROW_VARIANT
            || o.size.y < 0.26
            || o.size.xz().min_element() < 0.30
            || o.size.xz().max_element() > 1.9
        {
            return None;
        }
        let mut r = stream(o.seed, 0x4c4f_5745_5254_4945);
        if !r.random_bool(if o.kind == ObjectKind::CoffeeTable {
            0.42
        } else {
            0.34
        }) {
            return None;
        }
        let deck = match r.random_range(0..10) {
            0..=3 => ShelfDeck::TimberPanel,
            4..=7 => ShelfDeck::SpacedSlats,
            _ => ShelfDeck::FoldedTray,
        };
        let frame = if deck != ShelfDeck::FoldedTray && r.random_bool(0.56) {
            Surface::Wood
        } else if r.random_bool(0.23) {
            Surface::Chrome
        } else {
            Surface::Metal
        };
        let post_half_width_m: f32 = if frame == Surface::Wood {
            r.random_range(0.015..0.028)
        } else {
            r.random_range(0.009..0.020)
        };
        let round_posts = frame != Surface::Wood && r.random_bool(0.56);
        let size = o.size.xz();
        let center_xz_m =
            size * Vec2::new(r.random_range(-0.032..0.032), r.random_range(-0.032..0.032));
        let requested = size * Vec2::new(r.random_range(0.36..0.45), r.random_range(0.35..0.45));
        // Solve the four post corners against each convex edge once. Include
        // the post section, bevel and backing inset, even on triangular tops.
        let polygon = outline();
        let polygon = polygon.as_ref();
        let mut scale = 1_f32;
        for (i, &a) in polygon.iter().enumerate() {
            let n = (polygon[(i + 1) % polygon.len()] - a).perp();
            let spread = n.abs().dot(requested);
            if spread > 0. {
                let room = n.dot(center_xz_m - a)
                    - n.abs().element_sum() * post_half_width_m
                    - n.length() * 0.020;
                scale = scale.min(room / spread);
            }
        }
        let post_centers_half_span_m = requested * scale;
        if post_centers_half_span_m.min_element() < post_half_width_m + 0.060 {
            return None;
        }
        // Slats normally span the long dimension. Nearly square products can
        // be assembled either way.
        let runs_along_x = if size.max_element() / size.min_element() < 1.35 {
            r.random_bool(0.5)
        } else {
            size.x > size.y
        };
        let (along, across) = if runs_along_x {
            (post_centers_half_span_m.x, post_centers_half_span_m.y)
        } else {
            (post_centers_half_span_m.y, post_centers_half_span_m.x)
        };
        let length_m = 2. * (along + post_half_width_m * 0.52);
        let width_m = 2. * (across - post_half_width_m - r.random_range(0.003..0.012));
        let underside = o.size.y - p.top_thickness - 0.012;
        let (boards, gap_m) = if deck == ShelfDeck::SpacedSlats {
            let n = (width_m / r.random_range(0.042..0.090))
                .round()
                .clamp(2., 11.) as usize;
            (n, (width_m / n as f32) * r.random_range(0.13..0.36))
        } else {
            (1, 0.)
        };
        Some(Self {
            deck,
            frame,
            center_xz_m,
            post_centers_half_span_m,
            post_half_width_m,
            round_posts,
            runs_along_x,
            length_m,
            width_m,
            top_y_m: underside * r.random_range(0.26..0.56),
            underside_y_m: underside,
            thickness_m: if deck == ShelfDeck::FoldedTray {
                r.random_range(0.0015..0.0028)
            } else {
                r.random_range(0.012_f32..0.024).max(length_m * 0.019)
            },
            rail_height_m: r.random_range(0.020_f32..0.044).min(underside * 0.13),
            rim_height_m: if deck == ShelfDeck::FoldedTray {
                r.random_range(0.013..0.032)
            } else {
                0.
            },
            boards,
            gap_m,
        })
    }

    /// Clear shelf rectangle in the table's local XZ plane. A folded rim
    /// occupies the edge even when the stored object is taller than the rim.
    pub fn storage_bounds(&self) -> (Vec2, Vec2) {
        let inset = if self.deck == ShelfDeck::FoldedTray {
            self.thickness_m * 1.5
        } else {
            0.
        };
        let half = Vec2::new(self.length_m, self.width_m) * 0.5 - Vec2::splat(inset);
        let half = if self.runs_along_x { half } else { half.yx() };
        (self.center_xz_m - half, self.center_xz_m + half)
    }

    pub(crate) fn clear_height(&self, margin: f32) -> f32 {
        self.underside_y_m - self.top_y_m - margin - 0.006
    }

    fn deck_bevel(&self) -> f32 {
        (self.thickness_m * 0.22).min(0.003)
    }

    pub(crate) fn carries(&self, kind: ObjectKind, size: Vec3, margin: f32) -> bool {
        if !matches!(
            kind,
            ObjectKind::Books | ObjectKind::Notepad | ObjectKind::StorageBox
        ) {
            return false;
        }
        // Even the tapered box bottom and smallest inset book cover contain
        // this central disk. Its radius must bridge a whole gap so there is
        // bearing on either side of the centre at every admitted heading.
        size.y <= self.clear_height(margin)
            && (self.deck != ShelfDeck::SpacedSlats
                || size.xz().min_element() * 0.20 >= self.gap_m + 2. * self.deck_bevel() + 0.001)
    }

    pub(super) fn build(self, a: &mut impl StructureSink, o: &IndoorObject, underside: f32) {
        let (u, v, grain, cross_grain) = if self.runs_along_x {
            (Vec3::X, Vec3::Z, GrainAxis::X, GrainAxis::Z)
        } else {
            (Vec3::Z, Vec3::X, GrainAxis::Z, GrainAxis::X)
        };
        let xyz = |length, height, width| u * length + Vec3::Y * height + v * width;
        let center = Vec3::new(self.center_xz_m.x, 0., self.center_xz_m.y);
        let span = Vec3::new(
            self.post_centers_half_span_m.x,
            0.,
            self.post_centers_half_span_m.y,
        );
        let (along, across) = (span.dot(u), span.dot(v));
        let label = o.kind.class_name();
        let frame = (self.frame, label);
        let r = self.post_half_width_m;
        let floor = self.top_y_m - self.thickness_m;
        let mut cuts = TimberCuts::new(o.seed, 0x5449_4552);
        for end in [-along, along] {
            for side in [-across, across] {
                let foot = center + xyz(end, 0., side);
                let glide = if self.frame == Surface::Wood {
                    0.
                } else {
                    0.004
                };
                if glide > 0. {
                    a.part(Surface::Rubber, label).cylinder(
                        r * 0.89,
                        glide + 0.0006,
                        Transform::from_translation(foot.with_y((glide + 0.0006) * 0.5)),
                    );
                }
                let height = underside - glide + 0.0008;
                let at = foot.with_y(glide + height * 0.5);
                if self.round_posts {
                    a.part(self.frame, label)
                        .cylinder(r, height, Transform::from_translation(at));
                } else {
                    a.cut_board(
                        &mut cuts,
                        frame,
                        at,
                        Vec3::new(2. * r, height, 2. * r),
                        r * 0.13,
                        GrainAxis::Y,
                    );
                }
            }
            // Seat the rail ends into the posts. The deck bears on both rails.
            a.cut_board(
                &mut cuts,
                frame,
                center + xyz(end, floor - self.rail_height_m * 0.5 + 0.0005, 0.),
                xyz(r * 1.35, self.rail_height_m, 2. * (across - r * 0.64)),
                0.0015,
                cross_grain,
            );
        }
        let surface = if self.deck == ShelfDeck::FoldedTray {
            Surface::Metal
        } else {
            Surface::Wood
        };
        let board_w = (self.width_m - self.gap_m * (self.boards - 1) as f32) / self.boards as f32;
        for i in 0..self.boards {
            let cross = (i as f32 + 0.5) * board_w + i as f32 * self.gap_m - self.width_m * 0.5;
            a.cut_board(
                &mut cuts,
                (surface, label),
                center + xyz(0., floor + self.thickness_m * 0.5, cross),
                xyz(self.length_m, self.thickness_m, board_w),
                self.deck_bevel(),
                grain,
            );
        }
        if self.deck == ShelfDeck::FoldedTray {
            let t = self.thickness_m;
            let h = self.rim_height_m;
            let y = self.top_y_m + h * 0.5 - 0.0004;
            for sign in [-1., 1.] {
                // Inset folds and narrow corner relief keep the exposed sheet
                // faces separate. The raised perimeter stiffens the thin tray.
                a.box_part(
                    surface,
                    label,
                    center + xyz(0., y, sign * (self.width_m * 0.5 - t)),
                    xyz(self.length_m - t, h, t),
                    t * 0.22,
                );
                a.box_part(
                    surface,
                    label,
                    center + xyz(sign * (self.length_m * 0.5 - t), y, 0.),
                    xyz(t, h, self.width_m - 3. * t - 0.001),
                    t * 0.22,
                );
            }
        }
    }
}
