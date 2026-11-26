//! Load paths between an upholstered seat pan and the backrest's actual sockets.
use super::*;

mod strip;
#[cfg(test)]
mod tests;

const INTRODUCED: u32 = 274;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackMountStyle {
    TwinTubes,
    BentStraps,
    BentPlywood,
    ForkedSpine,
}

impl BackMountStyle {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::TwinTubes => "twin_tubes",
            Self::BentStraps => "bent_straps",
            Self::BentPlywood => "bent_plywood",
            Self::ForkedSpine => "forked_spine",
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct BackMount {
    pub style: BackMountStyle,
    /// Outside width of a carrier; for round stock this is the diameter.
    pub width_m: f32,
    /// Full section depth, normal to the bend.
    pub section_depth_m: f32,
    pub bend_radius_m: f32,
    pub rear_clearance_m: f32,
    pub pan_station_fraction: f32,
    pub foot_span_fraction: f32,
    pub return_height_fraction: f32,
}

impl ChairProgram {
    pub fn back_mount(&self, o: &IndoorObject, version: u32) -> Option<BackMount> {
        if version < INTRODUCED
            || o.kind != ObjectKind::Chair
            || !matches!(o.variant, 0..=5 | 9)
            || self.formed_seat.is_some()
        {
            return None;
        }
        // Chairs bought together can share tooling. Independent chairs and
        // replacement seats retain instance variation, even in a matched set.
        let mut item = stream(o.seed, u64::from_le_bytes(*b"BackStay"));
        let order = o
            .finish_seed
            .zip(o.interaction_target)
            .filter(|_| item.random_bool(0.72));
        let mut lot = order.map(|(seed, table)| {
            stream(
                seed,
                u64::from_le_bytes(*b"StayTool")
                    ^ (table as u64).rotate_left(23)
                    ^ u64::from(o.variant),
            )
        });
        let r = lot.as_mut().unwrap_or(&mut item);
        let wood = self.frame_surface == Surface::WoodEdge;
        let style = if wood {
            BackMountStyle::BentPlywood
        } else if self.base_style < 4 && r.random_bool(0.54) {
            BackMountStyle::ForkedSpine
        } else if r.random_bool(0.53) {
            BackMountStyle::TwinTubes
        } else {
            BackMountStyle::BentStraps
        };
        let (width, depth) = match style {
            BackMountStyle::TwinTubes => {
                let diameter = r.random_range(0.014..0.025);
                (diameter, diameter)
            }
            BackMountStyle::BentPlywood => {
                (r.random_range(0.027..0.048), r.random_range(0.009..0.015))
            }
            BackMountStyle::BentStraps => {
                (r.random_range(0.022..0.043), r.random_range(0.0045..0.009))
            }
            BackMountStyle::ForkedSpine => {
                (r.random_range(0.049..0.084), r.random_range(0.008..0.016))
            }
        };
        Some(BackMount {
            style,
            width_m: width,
            section_depth_m: depth,
            bend_radius_m: r.random_range(if wood { 0.026..0.040 } else { 0.016..0.035 }),
            rear_clearance_m: r.random_range(0.004..0.017),
            pan_station_fraction: r.random_range(0.16..0.31),
            foot_span_fraction: if wood {
                1.0
            } else {
                r.random_range(0.77..1.06)
            },
            return_height_fraction: r.random_range(0.44..0.73),
        })
    }
}

#[derive(Clone, Copy)]
struct Pan {
    root_z: f32,
    root_y: f32,
    rear_z: f32,
    bend: f32,
    return_fraction: f32,
}

impl BackMount {
    fn fit(&self, reserve_depth: f32, seat_depth: f32, pan_bottom: f32) -> Pan {
        // The pan is centered; the cushion sits 15 mm forward of it.
        let rear_edge = (seat_depth * 0.465).max(seat_depth * 0.5 - 0.015);
        let half = self.section_depth_m * 0.5;
        let available = (reserve_depth * 0.5 - rear_edge - 2. * half - 0.002).max(0.);
        let clearance = self.rear_clearance_m.min(available * 0.40);
        let bend = self.bend_radius_m.min(available - clearance);
        let saddle = if self.style == BackMountStyle::TwinTubes {
            0.004
        } else {
            0.
        };
        Pan {
            root_z: seat_depth * self.pan_station_fraction,
            root_y: pan_bottom - saddle - half + 0.0015,
            rear_z: rear_edge + half + clearance + bend,
            bend,
            return_fraction: self.return_height_fraction,
        }
    }

    pub(super) fn build(
        &self,
        a: &mut Assembly,
        o: &IndoorObject,
        frame: Surface,
        depth: f32,
        pan_bottom: f32,
        sockets: [Vec3; 2],
    ) {
        let pan = self.fit(o.size.z, depth, pan_bottom);
        match self.style {
            BackMountStyle::BentPlywood => {
                let mut cuts = TimberCuts::new(o.seed, u64::from_le_bytes(*b"StayPlys"));
                for socket in sockets {
                    laminate::build(
                        a,
                        &pan.path(socket.x, socket),
                        self.width_m,
                        self.section_depth_m,
                        cuts.origin(),
                    );
                }
            }
            BackMountStyle::ForkedSpine => {
                let g = a.part(frame, "chair");
                let mid = (sockets[0] + sockets[1]) * 0.5;
                let joint = Vec3::new(
                    mid.x,
                    UPRIGHT_SEAT_HEIGHT
                        + (mid.y - UPRIGHT_SEAT_HEIGHT)
                            * (0.40 + 0.42 * self.return_height_fraction),
                    pan.rear_z + (mid.z - pan.rear_z) * 0.52,
                );
                let path = pan.path(mid.x, joint);
                strip::build(g, &path, self.width_m, self.section_depth_m);
                let join = joint - (joint - path[6]).normalize() * 0.004;
                for (sign, socket) in [-1., 1.].into_iter().zip(sockets) {
                    let root =
                        join + Vec3::X * (sign * self.width_m * 0.26 * self.foot_span_fraction);
                    let elbow = root.lerp(socket, 0.48) + Vec3::Z * 0.009;
                    g.tapered_tube(
                        &[root, elbow, socket],
                        self.section_depth_m * 0.85,
                        self.section_depth_m * 0.62,
                        8,
                    );
                }
            }
            BackMountStyle::TwinTubes | BackMountStyle::BentStraps => {
                let g = a.part(frame, "chair");
                for socket in sockets {
                    let x = socket.x * self.foot_span_fraction;
                    let path = pan.path(x, socket);
                    if self.style == BackMountStyle::TwinTubes {
                        // A bearing pad fixes the round tube to the flat pan.
                        g.cuboid(
                            Vec3::new(self.width_m * 1.65, 0.005, depth * 0.14),
                            0.,
                            Transform::from_xyz(x, pan_bottom - 0.0015, pan.root_z + depth * 0.034),
                        );
                        g.tapered_tube(&path, self.width_m * 0.5, self.width_m * 0.42, 8);
                    } else {
                        strip::build(g, &path, self.width_m, self.section_depth_m);
                    }
                }
            }
        }
    }
}

impl Pan {
    /// Both bends have tangent straight runs. The lower bend starts aft of the
    /// seat edge and the upper one returns towards the back above the cushion.
    fn path(self, x: f32, socket: Vec3) -> [Vec3; 8] {
        let root = Vec3::new(x, self.root_y, self.root_z);
        let lower = Vec3::new(x, self.root_y, self.rear_z);
        let upper = lower
            .with_y(UPRIGHT_SEAT_HEIGHT + self.return_fraction * (socket.y - UPRIGHT_SEAT_HEIGHT));
        let exit = socket - upper;
        let r = self
            .bend
            .min(0.40 * (upper.y - lower.y))
            .min(0.40 * (lower.z - root.z));
        let r2 = self
            .bend
            .min(exit.length() * 0.35)
            .min((upper.y - lower.y) * 0.32);
        let a = lower - Vec3::Z * r;
        let b = lower + Vec3::Y * r;
        let c = upper - Vec3::Y * r2;
        let d = upper + exit.normalize() * r2;
        [
            root,
            a,
            (a + 2. * lower + b) * 0.25,
            b,
            c,
            (c + 2. * upper + d) * 0.25,
            d,
            socket,
        ]
    }
}
