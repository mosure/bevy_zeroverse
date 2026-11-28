use super::*;
use crate::sample::failure::{CaptureError, SceneRejectionKind};
impl IndoorManifest {
    pub fn validate_public_programs(&self) -> Result<(), String> {
        self.validate_public_programs_typed()
            .map_err(|e| e.to_string())
    }

    fn validate_service_routes(&self, p: &ProgramAssembly) -> Result<(), &'static str> {
        let [spine, cross, fronts @ ..] = p.reservations.as_slice() else {
            return Err("missing service cross-aisles");
        };
        if spine.role != ReservationRole::Circulation
            || cross.role != ReservationRole::Circulation
            || spine.min.y != p.min.y
            || spine.max.y != p.max.y
            || cross.min.x != p.min.x
            || cross.max.x != p.max.x
        {
            return Err("disconnected service cross-aisles");
        }
        if p.activity.is_retail() {
            self.validate_retail_fronts(p, fronts)?;
        }
        let structure = self.columns();
        for r in &p.reservations {
            if r.min.cmplt(p.min - 0.001).any()
                || r.max.cmpgt(p.max + 0.001).any()
                || structure
                    .iter()
                    .any(|&(l, h)| l.y < 1.90 && h.y > 0.001 && r.overlaps(l, h))
            {
                return Err("service aisle obstructed by architecture");
            }
            let touches = |s: &Reservation| {
                r.min.cmple(s.max + 0.001).all() && r.max.cmpge(s.min - 0.001).all()
            };
            if matches!(
                r.role,
                ReservationRole::Browsing
                    | ReservationRole::FittingAccess
                    | ReservationRole::EquipmentAccess
            ) && !touches(spine)
                && !touches(cross)
            {
                return Err("working front is isolated from the cross-aisles");
            }
        }
        Ok(())
    }

    fn validate_retail_fronts(
        &self,
        p: &ProgramAssembly,
        fronts: &[Reservation],
    ) -> Result<(), &'static str> {
        use crate::scene::procedural_indoor::objects::public::{retail::checkout, CheckoutForm};
        let tills = p
            .members
            .iter()
            .map(|&i| &self.objects[i])
            .filter(|o| o.kind == ObjectKind::CheckoutCounter);
        let (mut staffed, mut count) = (0, 0);
        for o in tills {
            count += 1;
            let form = CheckoutForm::for_object(o);
            let has_staff = form != Some(CheckoutForm::SelfCheckout);
            staffed += usize::from(has_staff);
            if self.generator_version >= checkout::VERSION {
                if CheckoutForm::apparel_assortment(o)
                    != (p.activity == IndoorLayout::ClothingStore)
                    || !matches!(
                        (p.activity, form),
                        (_, Some(CheckoutForm::SelfCheckout))
                            | (IndoorLayout::ClothingStore, Some(CheckoutForm::ApparelWrap))
                            | (
                                IndoorLayout::GroceryStore,
                                Some(CheckoutForm::BasketRegister | CheckoutForm::ConveyorLane)
                            )
                    )
                {
                    return Err("checkout form does not serve its shop");
                }
                for staff in [false, true].into_iter().take(1 + usize::from(has_staff)) {
                    let role = if staff {
                        ReservationRole::StaffAccess
                    } else {
                        ReservationRole::Browsing
                    };
                    if !fronts.iter().any(|r| {
                        r.role == role && furnishing::retail::checkout::serves(r, o, staff)
                    }) {
                        return Err("missing sales front or cashier access");
                    }
                }
            }
        }
        if (self.generator_version >= checkout::VERSION && count > 0 && staffed != 1)
            || fronts
                .iter()
                .filter(|r| r.role == ReservationRole::StaffAccess)
                .count()
                != staffed
            || fronts
                .iter()
                .filter(|r| r.role == ReservationRole::Browsing)
                .count()
                != p.accepted_capacity + count
        {
            return Err("missing sales front or cashier access");
        }
        let approaches = fronts
            .iter()
            .filter(|r| r.role == ReservationRole::FittingAccess);
        let booths = p
            .members
            .iter()
            .map(|&id| &self.objects[id])
            .filter(|o| o.kind == ObjectKind::FittingRoom);
        if approaches.clone().count() != booths.clone().count()
            || booths.into_iter().any(|o| {
                !approaches
                    .clone()
                    .any(|r| furnishing::fitting::serves(r, o))
            })
        {
            return Err("missing fitting room approach");
        }
        Ok(())
    }

    pub(crate) fn validate_public_programs_typed(&self) -> Result<(), CaptureError> {
        if !self.layout.is_public() {
            if !self.assembly_diagnostics.programs.is_empty() {
                return Err("public program receipt on an unrelated scene".into());
            }
            return Ok(());
        }
        let zones = &self
            .program
            .as_ref()
            .ok_or("missing public spatial program")?
            .zones;
        if self.assembly_diagnostics.programs.len() != zones.len() {
            return Err("missing functional zone receipt".into());
        }
        let mut rejection = None;
        for p in &self.assembly_diagnostics.programs {
            let reason =
                |s: &str| format!("seed {} {:?} zone {}: {s}", self.seed, p.activity, p.zone);
            let fail = |s: &str| CaptureError::Failed(reason(s));
            if zones.get(p.zone).is_none_or(|z| z.activity != p.activity) {
                return Err(fail("zone identity differs from functional receipt"));
            }
            if p.parameters != ProgramParameters::sample(self.seed, p.zone, self.density) {
                return Err(fail("functional parameters cannot replay from seed"));
            }
            if p.accepted_capacity != p.realized_capacity(&self.objects) {
                return Err(fail("capacity receipt differs from accepted objects"));
            }
            for kind in &p.required {
                if !p
                    .members
                    .iter()
                    .any(|&id| self.objects.get(id).is_some_and(|o| o.kind == *kind))
                {
                    rejection.get_or_insert_with(|| {
                        CaptureError::rejected(
                            self.seed,
                            SceneRejectionKind::MissingProgramAnchor,
                            reason(&format!("missing required {kind:?}")),
                        )
                    });
                }
            }
            if matches!(
                p.activity,
                IndoorLayout::Classroom
                    | IndoorLayout::RehearsalRoom
                    | IndoorLayout::LectureHall
                    | IndoorLayout::Cafe
                    | IndoorLayout::Cafeteria
                    | IndoorLayout::Restaurant
            ) && p.accepted_capacity < 2
            {
                rejection.get_or_insert_with(|| {
                    CaptureError::rejected(
                        self.seed,
                        SceneRejectionKind::InsufficientProgramCapacity,
                        reason("fewer than two usable seats"),
                    )
                });
            }
            if matches!(
                p.activity,
                IndoorLayout::CyclingStudio | IndoorLayout::YogaStudio
            ) {
                if p.accepted_capacity < 2 {
                    rejection.get_or_insert_with(|| {
                        CaptureError::rejected(
                            self.seed,
                            SceneRejectionKind::InsufficientProgramCapacity,
                            reason("class needs an instructor and a participant"),
                        )
                    });
                } else {
                    let instructor = p
                        .class_instructor(&self.objects)
                        .ok_or_else(|| fail("class focus misses its instructor station"))?;
                    for participant in p
                        .members
                        .iter()
                        .filter_map(|&id| self.objects.get(id))
                        .filter(|o| o.kind == instructor.kind && o.id != instructor.id)
                    {
                        if !class_stations_face(p.activity, instructor, participant) {
                            return Err(fail("class stations face away from each other"));
                        }
                    }
                }
            }
            if p.activity.is_retail() && p.accepted_capacity < 2 {
                rejection.get_or_insert_with(|| {
                    CaptureError::rejected(
                        self.seed,
                        SceneRejectionKind::InsufficientProgramCapacity,
                        reason("fewer than two accessible stocked fixtures"),
                    )
                });
            }
            if p.activity == IndoorLayout::LockerRoom {
                furnishing::changing::validate(self, p).map_err(fail)?;
            }
            if p.activity == IndoorLayout::RehearsalRoom {
                furnishing::rehearsal::percussion::validate(self, p).map_err(fail)?;
            }
            for &id in &p.members {
                let o = self
                    .objects
                    .get(id)
                    .ok_or_else(|| fail("dangling assembly member"))?;
                if o.kind == ObjectKind::Lectern
                    && self.generator_version >= crate::scene::procedural_indoor::objects::public::furnishing::lectern::VERSION
                    && !furnishing::presenter::faces_class(o, p)
                {
                    return Err(fail("presenter at the reading edge faces away from the class"));
                }
                if matches!(o.kind, ObjectKind::MusicStand | ObjectKind::StagePiano)
                    && !o.interaction_target.and_then(|id| self.objects.get(id)).is_some_and(|chair| {
                        p.activity == IndoorLayout::RehearsalRoom
                            && p.members.contains(&chair.id)
                            && crate::scene::procedural_indoor::objects::public::music::faces_player(o, chair)
                    })
                {
                    return Err(fail("ensemble equipment lacks a facing player station"));
                }
                let (lo, hi) = o.bounds();
                if matches!(
                    o.kind,
                    ObjectKind::WashingMachine
                        | ObjectKind::TumbleDryer
                        | ObjectKind::LaundryFoldingTable
                ) && (p.activity != IndoorLayout::Laundromat
                    || !furnishing::laundry::front_is_served(p, o))
                {
                    return Err(fail(
                        "laundry station lacks a connected loading or folding aisle",
                    ));
                }
                if o.kind == ObjectKind::LaundryCart
                    && !furnishing::laundry_carts::valid_site(self, p, o)
                {
                    return Err(fail(
                        "laundry cart lacks its station or clear aisle approach",
                    ));
                }
                if o.kind == ObjectKind::BasketStack
                    && (!matches!(
                        p.activity,
                        IndoorLayout::GroceryStore | IndoorLayout::ClothingStore
                    ) || !furnishing::retail::baskets::valid_site(self, p, o))
                {
                    return Err(fail(
                        "shopping baskets lack a reachable entrance or checkout aisle",
                    ));
                }
                if o.kind == ObjectKind::EspressoMachine
                    && (p.activity != IndoorLayout::Cafe
                        || !p
                            .members
                            .iter()
                            .filter_map(|&i| self.objects.get(i))
                            .any(|g| {
                                g.kind == ObjectKind::CoffeeGrinder
                                    && g.interaction_target == Some(id)
                                    && g.support == o.support
                            })
                        || !o.support.is_some_and(|id| {
                            p.members.contains(&id)
                                && self
                                    .objects
                                    .get(id)
                                    .is_some_and(|c| c.kind == ObjectKind::ServiceCounter)
                        }))
                {
                    return Err(fail(
                        "espresso machine lacks its grinder or service counter",
                    ));
                }
                if o.kind == ObjectKind::HotFoodWell
                    && (p.activity != IndoorLayout::Cafeteria
                        || !o.support.is_some_and(|id| {
                            p.members.contains(&id)
                                && self
                                    .objects
                                    .get(id)
                                    .is_some_and(|counter| self.hot_food_access(o, counter))
                        }))
                {
                    return Err(fail("hot-food well has no accessible service counter"));
                }
                if o.kind == ObjectKind::ExhibitPlinth
                    && !self.objects.iter().any(|child| child.support == Some(id))
                {
                    return Err(fail("exhibit plinth has no supported exhibit"));
                }
                if matches!(o.kind, ObjectKind::YogaBlock | ObjectKind::YogaBolster)
                    && (!matches!(
                        p.activity,
                        IndoorLayout::YogaStudio | IndoorLayout::FitnessRoom
                    ) || !o.support.is_some_and(|id| {
                        p.members.contains(&id)
                            && self
                                .objects
                                .get(id)
                                .is_some_and(|mat| mat.kind == ObjectKind::ExerciseMat)
                    }))
                {
                    return Err(fail("practice prop lacks a mat in its exercise zone"));
                }
                if o.kind == ObjectKind::PowerRack
                    && !furnishing::fitness::rack_access_present(self, p, o)
                {
                    return Err(fail(
                        "lifting rack lacks sleeve access, working floor or overhead clearance",
                    ));
                }
                let (min, max) = p.member_region(self, o);
                if lo.x < min.x - 0.001
                    || lo.z < min.y - 0.001
                    || hi.x > max.x + 0.001
                    || hi.z > max.y + 0.001
                {
                    return Err(fail("assembly crosses its functional region"));
                }
            }
            for route in &p.reservations {
                if !route.min.is_finite()
                    || !route.max.is_finite()
                    || route.min.cmpge(route.max).any()
                {
                    return Err(fail("invalid reserved route"));
                }
            }
            // Each rotated footprint is shared by all routes in this zone.
            for o in self.objects.iter().filter(|o| o.solid && !o.neighbor) {
                let (a, b) = o.bounds();
                if !p.space_clear_for(self, o, a, b) {
                    return Err(fail("reserved circulation obstructed"));
                }
            }
            if p.activity == IndoorLayout::Gymnasium {
                let court = self
                    .program
                    .as_ref()
                    .and_then(|p| p.volleyball)
                    .ok_or_else(|| fail("missing gymnasium court"))?;
                let (lo, hi) = court.protected_bounds();
                let clear_height = court.clear_height();
                if p.reservations.first()
                    != Some(&Reservation {
                        min: lo,
                        max: hi,
                        role: ReservationRole::PlayingArea,
                    })
                    || p.capacity_unit != CapacityUnit::PlayingCourts
                    || p.accepted_capacity != 1
                    || !p
                        .members
                        .iter()
                        .any(|&id| furnishing::gymnasium::matches_net(&self.objects[id], court))
                {
                    return Err(fail(
                        "court run-off or net installation differs from its plan",
                    ));
                }
                let mut accesses = p.reservations[1..].iter();
                let columns = self.columns();
                for bank in self
                    .objects
                    .iter()
                    .filter(|o| !o.neighbor && o.kind == ObjectKind::BleacherBank)
                {
                    let access = furnishing::gymnasium::bank_access(bank);
                    if !furnishing::gymnasium::valid_bank(bank, court)
                        || accesses.next() != Some(&access)
                        || columns
                            .iter()
                            .any(|&(lo, hi)| lo.y < 2.1 && hi.y > 0. && access.overlaps(lo, hi))
                    {
                        return Err(fail(
                            "spectator bank misses its sideline, access or tier bounds",
                        ));
                    }
                }
                for goal in self
                    .objects
                    .iter()
                    .filter(|o| !o.neighbor && o.kind == ObjectKind::BasketballGoal)
                {
                    if !p.members.contains(&goal.id)
                        || !furnishing::gymnasium::valid_goal(self, goal, court)
                        || accesses.next() != Some(&furnishing::gymnasium::goal_access(goal))
                    {
                        return Err(fail(
                            "basketball goal misses its backing, run-off or shooting approach",
                        ));
                    }
                }
                if accesses.next().is_some() {
                    return Err(fail("unexpected gymnasium access route"));
                }
                if self.minimum_ceiling_height(lo, hi) < clear_height
                    || columns.iter().any(|&(a, b)| {
                        b.y > 0.001 && a.y < clear_height && p.reservations[0].overlaps(a, b)
                    })
                {
                    return Err(fail("structure obstructs required overhead playing space"));
                }
            }
            if p.activity == IndoorLayout::Courtyard {
                let e = self
                    .envelope
                    .as_ref()
                    .ok_or_else(|| fail("missing courtyard envelope"))?;
                let roof = e
                    .courtyard
                    .as_ref()
                    .ok_or_else(|| fail("missing courtyard opening"))?;
                if p.role != SpaceRole::Waiting
                    || p.zone != 0
                    || !matches!(
                        p.required.as_slice(),
                        [ObjectKind::Plant | ObjectKind::PlantingBed]
                    )
                    || p.reservations != furnishing::courtyard::routes(self, roof)
                    || e.pillars.iter().any(|col| {
                        let pad = Vec2::splat(col.radius);
                        p.reservations.iter().any(|r| {
                            (col.center + pad).cmpgt(r.min).all()
                                && (col.center - pad).cmplt(r.max).all()
                        })
                    })
                {
                    return Err(fail(
                        "courtyard entrance route or column clearance is invalid",
                    ));
                }
            }
            if p.activity == IndoorLayout::StairHall {
                let e = self
                    .envelope
                    .as_ref()
                    .ok_or_else(|| fail("missing stair hall envelope"))?;
                let m = e
                    .mezzanine
                    .as_ref()
                    .ok_or_else(|| fail("missing stair hall flight"))?;
                m.validate(e, self.room_size).map_err(|s| fail(&s))?;
                let circulation = furnishing::stair_hall::routes(self);
                if m.construction
                    .as_ref()
                    .is_none_or(|c| c.wall_abutment.is_none())
                    || p.role != SpaceRole::Transit
                    || p.zone != 0
                    || !p.reservations.starts_with(&circulation)
                    || p.reservations
                        .iter()
                        .skip(circulation.len())
                        .any(|r| r.role != ReservationRole::Viewing)
                {
                    return Err(fail(
                        "stair hall must connect the entrance to the upper gallery",
                    ));
                }
                for route in &p.reservations {
                    if !super::super::super::envelope::polygon::box_inside(
                        &e.footprint,
                        route.min,
                        route.max,
                        0.,
                    ) || e.pillars.iter().any(|pillar| {
                        let pad = Vec2::splat(pillar.radius);
                        (pillar.center + pad).cmpgt(route.min).all()
                            && (pillar.center - pad).cmplt(route.max).all()
                    }) {
                        return Err(fail("stair hall route obstructed by architecture"));
                    }
                }
            }
            if self.generator_version >= super::super::super::objects::public::beam_seating::VERSION
                && matches!(
                    p.activity,
                    IndoorLayout::Hallway | IndoorLayout::ElevatorLanding
                )
            {
                furnishing::waiting::validate(self, p).map_err(fail)?;
            }
            if p.activity == IndoorLayout::Hallway {
                let zone = &zones[p.zone];
                if p.zone != 0
                    || p.role != SpaceRole::Transit
                    || p.min != zone.min
                    || p.max != zone.max
                    || !p
                        .reservations
                        .iter()
                        .filter(|r| r.role != ReservationRole::SeatingAccess)
                        .eq(furnishing::hallway::routes(self, zone, &p.parameters).iter())
                {
                    return Err(fail(
                        "hallway circulation must connect the entrance and every suite frontage",
                    ));
                }
            }
            if p.activity.is_retail() || p.activity == IndoorLayout::Laundromat {
                self.validate_service_routes(p).map_err(fail)?;
            }
            for service in &p.services {
                if self.objects.get(service.source).is_none()
                    || self.objects.get(service.target).is_none()
                    || !service.source_point.is_finite()
                    || !service.target_point.is_finite()
                {
                    return Err(fail("invalid service relationship"));
                }
            }
        }
        // A recoverable capacity/placement rejection must never hide a corrupt
        // receipt, service relationship or reservation elsewhere in the room.
        rejection.map_or(Ok(()), Err)
    }
}
