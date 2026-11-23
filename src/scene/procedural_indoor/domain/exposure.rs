//! A single exposure for the rig, estimated from the installed lights and facade.
//! The estimates describe room-average admission, not measured lux at a surface.
use super::*;
use crate::scene::procedural_indoor::{
    architecture::{
        self,
        facade::{Facade, Shade, WindowOpening},
    },
    envelope::polygon,
};
mod borrowed;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IndoorExposureMeter {
    pub electric_lux: f32,
    pub direct_daylight_lux: f32,
    pub diffuse_daylight_lux: f32,
    pub bias_ev: f32,
}

impl IndoorExposureMeter {
    pub(crate) fn incident_lux(self) -> f32 {
        self.electric_lux + self.direct_daylight_lux + self.diffuse_daylight_lux
    }

    pub(super) fn ev100(self) -> f32 {
        let illuminance = self.incident_lux();
        // Partial adaptation retains deliberate low-light/highlight cases across
        // rooms. All views share exposure; RGB responses never feed back into it.
        (6.25 + 0.62 * (illuminance.max(0.1) / 400.).log2() + self.bias_ev).clamp(2., 12.)
    }

    pub(super) fn valid(self) -> bool {
        [
            self.electric_lux,
            self.direct_daylight_lux,
            self.diffuse_daylight_lux,
        ]
        .into_iter()
        .all(|v| v.is_finite() && v >= 0.)
            && self.bias_ev.is_finite()
    }
}

/// Effective areas for diffuse sky and the collimated solar beam, in square
/// metres. Shade deployment and opaque frit affect both channels.
fn transmitted_apertures(facade: &Facade, sun_cosine: f32) -> Vec2 {
    facade
        .openings
        .iter()
        .map(|o| transmitted_opening(facade, o, sun_cosine))
        .sum()
}

fn transmitted_opening(facade: &Facade, opening: &WindowOpening, sun_cosine: f32) -> Vec2 {
    let coverage = facade.shade_coverage_at(opening);
    let area = architecture::facade::treatments::drapery::visible_area(facade, opening, coverage)
        .unwrap_or_else(|| other_covering_area(facade, opening, coverage));
    let transmission = opening
        .glazing
        .as_ref()
        .map_or(Vec2::new(0.85, 0.92), |pane| {
            let recipe = pane.glass_recipe();
            Vec2::new(
                recipe.sky_transmittance(),
                recipe.luminous_transmittance(sun_cosine),
            )
        });
    area * transmission * Vec2::new(1., sun_cosine)
}

fn other_covering_area(facade: &Facade, opening: &WindowOpening, coverage: f32) -> f32 {
    let obstructed = match facade.shade {
        Shade::None => 0.,
        Shade::Venetian => {
            coverage * (0.15 + 0.85 * (facade.shade_tilt * coverage.sqrt()).sin().abs())
        }
        Shade::Vertical => {
            let angle = facade.shade_tilt.abs().max(0.12) * coverage + 1.50 * (1. - coverage);
            angle.cos().abs()
        }
        _ => coverage,
    };
    let clear_fraction = 1. - 0.95 * obstructed.clamp(0., 1.);
    let span = opening.max - opening.min;
    let clear = facade
        .treatment
        .as_ref()
        .and_then(|p| p.track)
        .filter(|_| facade.shade == Shade::PanelTrack)
        .map_or(clear_fraction, |track| {
            track.clear_fraction(span.x, coverage)
        });
    let clear = architecture::facade::treatments::slats::clear_fraction(facade, span, coverage)
        .unwrap_or(clear);
    architecture::facade::visible_pane_area(facade, opening) * clear
}

/// Lower solar elevations travel through a longer atmospheric path. Retain
/// atmosphere-dependent spread at a given elevation without orange midday beams.
fn solar_kelvin(elevation: f32, seed: u64) -> f32 {
    let scatter = stream(seed, 0x534f4c4152434354).random::<f32>();
    let t = ((elevation.sin() - 0.07) / 0.62).clamp(0., 1.);
    let t = t * t * (3. - 2. * t);
    (2450. + 1350. * scatter).lerp(4900. + 1450. * scatter, t)
}

impl IndoorManifest {
    /// Meter after the envelope and fixtures exist, before applying explicit
    /// illumination/exposure interventions. Archived manifests retain their EV.
    pub(crate) fn meter_indoor_lighting(&mut self) {
        let Some(photometry) = self.domain().map(|d| &d.photometry) else {
            return;
        };
        let Some(mut meter) = photometry.meter else {
            return;
        };
        let Some(envelope) = &self.envelope else {
            return;
        };
        let floor = polygon::area(&envelope.footprint);
        let positions = architecture::fixture_positions(self);
        let install = self.generator_version >= 166 && photometry.ceiling_circuits.is_none();
        let mut windows = Vec::new();
        let diffuse_lux = photometry.diffuse_sky_lux();
        let sun = architecture::sun_direction(self);
        let mut apertures = borrowed::apertures(self, sun);
        let borrowed_lux =
            (0.28 * diffuse_lux * apertures.x + 0.38 * photometry.sun_lux * apertures.y) / floor;
        for wall in &envelope.walls {
            if let Some(facade) = &wall.facade {
                let frame = envelope.wall_transform(wall.edge);
                let incidence = (frame.rotation * Vec3::NEG_Z).dot(sun).clamp(0., 1.);
                if install {
                    let mut wall_area = Vec2::ZERO;
                    for o in &facade.openings {
                        let clear = transmitted_opening(facade, o, incidence);
                        wall_area += clear;
                        windows.push(circuits::DaylightOpening {
                            a: frame.transform_point(Vec3::X * o.min.x).xz(),
                            b: frame.transform_point(Vec3::X * o.max.x).xz(),
                            reach_m: (o.max.y - 0.75).max(0.25) * 1.6,
                            admitted_lux: (0.28 * diffuse_lux * clear.x
                                + 0.38 * photometry.sun_lux * clear.y)
                                / floor,
                        });
                    }
                    apertures += wall_area;
                } else {
                    apertures += transmitted_apertures(facade, incidence);
                }
            }
        }
        // A vertical opening receives half an isotropic sky hemisphere. These
        // utilization factors estimate distribution from the pane into the room.
        meter.diffuse_daylight_lux = diffuse_lux * (0.5 * 0.56) * apertures.x / floor;
        meter.direct_daylight_lux = photometry.sun_lux * 0.38 * apertures.y / floor;
        let kelvin = solar_kelvin(
            self.sun_elevation,
            super::super::photometry::lighting_seed(self),
        );
        if install {
            let circuits = CeilingCircuits::sample(self, &positions, &windows, borrowed_lux);
            self.program
                .as_mut()
                .unwrap()
                .domain
                .as_mut()
                .unwrap()
                .photometry
                .ceiling_circuits = Some(circuits);
        }
        let flux: f32 = (0..architecture::primary_fixture_count(self, &positions))
            .map(|i| architecture::fixture_photometry_at(self, i, &positions).1)
            .sum();
        meter.electric_lux = flux * 0.70 / architecture::fixture_floor_area(self);
        let photometry = &mut self
            .program
            .as_mut()
            .unwrap()
            .domain
            .as_mut()
            .unwrap()
            .photometry;
        photometry.sun_kelvin = kelvin;
        photometry.ev100 = meter.ev100();
        photometry.meter = Some(meter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        appearance::AppearanceSettings, layout::IndoorLayout, photometry,
    };

    fn room() -> IndoorManifest {
        IndoorManifest::generate_with_humans(42003, IndoorLayout::Classroom, 0.65, 2, 0.0).unwrap()
    }

    #[test]
    fn metered_clear_area_excludes_exactly_the_opaque_frit_visible_in_the_mesh() {
        let scene = room();
        let mut facade = scene
            .envelope
            .as_ref()
            .unwrap()
            .walls
            .iter()
            .find_map(|w| w.facade.clone())
            .unwrap();
        facade.shade = Shade::None;
        for o in &mut facade.openings {
            let p = o.glazing.as_mut().unwrap();
            p.opening = 0.;
            p.treatment = 0;
            p.band_m = Vec2::new(o.min.y, o.max.y);
        }
        let area = |f: &Facade| {
            f.openings
                .iter()
                .map(|o| architecture::facade::visible_pane_area(f, o))
                .sum::<f32>()
        };
        let clear_area = area(&facade);
        let clear_light = transmitted_apertures(&facade, 0.7);
        for o in &mut facade.openings {
            o.glazing.as_mut().unwrap().treatment = 6;
        }
        let mut mesh = crate::scene::procedural_indoor::objects::Assembly::default();
        architecture::facade::glazing_at(&mut mesh, Transform::IDENTITY, &facade);
        let g = &mesh.parts[&(
            crate::scene::procedural_indoor::materials::Surface::Paint,
            "window#frit".into(),
        )];
        let projected: f32 = g
            .indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| Vec3::from_array(g.positions[i as usize]));
                0.5 * (b - a).cross(c - a).z.max(0.)
            })
            .sum();
        assert!(projected > 0.05);
        assert!((clear_area - area(&facade) - projected).abs() < 1e-4);
        assert!(transmitted_apertures(&facade, 0.7).cmplt(clear_light).all());
        assert_eq!(transmitted_apertures(&facade, 0.).y, 0.);
    }

    #[test]
    fn solar_spectrum_varies_with_atmosphere_and_elevation() {
        for seed in 0..256 {
            let low = solar_kelvin(0.08, seed);
            let high = solar_kelvin(0.9, seed);
            assert!(low < 3900. && high > 4800. && high < 6500.);
            assert!(solar_kelvin(0.35, seed) > low);
            assert_eq!(high, solar_kelvin(0.9, seed));
        }
    }

    #[test]
    fn room_geometry_and_shades_control_daylight_without_affecting_electric_flux() {
        let mut scene = room();
        // Isolate the selected facade so the 5% shade transmission below can
        // be compared against its own unshaded illuminance.
        scene.adjoining_elevation = None;
        let wall = scene
            .envelope
            .as_ref()
            .unwrap()
            .walls
            .iter()
            .position(|w| w.facade.is_some())
            .unwrap();
        let envelope = scene.envelope.as_mut().unwrap();
        let outward = envelope.wall_transform(envelope.walls[wall].edge).rotation * Vec3::NEG_Z;
        for (i, w) in envelope.walls.iter_mut().enumerate() {
            if i != wall {
                w.facade = None;
            } else {
                w.facade.as_mut().unwrap().shade = Shade::None;
            }
        }
        scene.sun_azimuth = outward.z.atan2(-outward.x);
        scene.sun_elevation = 0.35;
        let lighting = &mut scene
            .program
            .as_mut()
            .unwrap()
            .domain
            .as_mut()
            .unwrap()
            .photometry;
        lighting.sun_lux = 40_000.;
        lighting.sky_radiance = Vec3::splat(600.);
        scene.meter_indoor_lighting();
        let open = scene.domain().unwrap().photometry.meter.unwrap();
        let original = scene.ev100();
        assert!(open.direct_daylight_lux > 0. && open.diffuse_daylight_lux > 0.);
        let expected_electric = {
            let positions = architecture::fixture_positions(&scene);
            positions
                .iter()
                .enumerate()
                .take_while(|(_, p)| p.z <= scene.room_size.z * 0.5)
                .map(|(i, _)| architecture::fixture_photometry_at(&scene, i, &positions).1)
                .sum::<f32>()
                * 0.70
                / architecture::fixture_floor_area(&scene)
        };
        assert_eq!(open.electric_lux, expected_electric);
        scene.sun_azimuth += std::f32::consts::PI;
        scene.meter_indoor_lighting();
        let away = scene.domain().unwrap().photometry.meter.unwrap();
        assert_eq!(away.direct_daylight_lux, 0.);
        assert_eq!(away.diffuse_daylight_lux, open.diffuse_daylight_lux);
        assert!(scene.ev100() < original);
        let facade = scene.envelope.as_mut().unwrap().walls[wall]
            .facade
            .as_mut()
            .unwrap();
        facade.shade = Shade::Roller;
        facade.shade_coverage = 1.;
        scene.meter_indoor_lighting();
        let closed = scene.domain().unwrap().photometry.meter.unwrap();
        assert!((closed.diffuse_daylight_lux / open.diffuse_daylight_lux - 0.05).abs() < 0.0001);
        assert_eq!(closed.electric_lux, open.electric_lux);
    }

    #[test]
    fn illuminance_and_exposure_interventions_remain_independent() {
        let mut control = room();
        let mut variant = control.clone();
        let settings = AppearanceSettings {
            lighting_seed: Some(42),
            ..default()
        };
        control.apply_appearance(settings.clone()).unwrap();
        variant
            .apply_appearance(AppearanceSettings {
                illumination_scale: 0.2,
                exposure_ev100_offset: 1.,
                ..settings
            })
            .unwrap();
        assert_eq!(
            control.domain().unwrap().photometry.meter,
            variant.domain().unwrap().photometry.meter
        );
        assert_eq!(control.cameras, variant.cameras);
        assert_eq!(variant.ev100(), control.ev100() + 1.);
        assert_eq!(variant.daylight_lux, control.daylight_lux * 0.2);
        for temperature in [2200., 4000., 6500., 9600.] {
            let chroma = photometry::tinted_white(temperature, 0.03);
            let c = photometry::source_color(&control, chroma).to_linear();
            assert!((c.red * 0.2126 + c.green * 0.7152 + c.blue * 0.0722 - 1.).abs() < 2e-6);
        }
    }

    #[test]
    fn archived_exposure_and_source_colors_replay() {
        let mut scene = room();
        let mut archived = serde_json::to_value(&scene).unwrap();
        archived["generator_version"] = 121.into();
        archived["program"]["domain"]["photometry"]
            .as_object_mut()
            .unwrap()
            .remove("meter");
        scene = serde_json::from_value(archived).unwrap();
        let ev = scene.ev100();
        scene.meter_indoor_lighting();
        assert_eq!(ev, scene.ev100());
        let c = Vec3::new(1., 0.6, 0.25);
        assert_eq!(
            photometry::source_color(&scene, c),
            Color::srgb(c.x, c.y, c.z)
        );
        assert_eq!(architecture::energized_fixture(&scene, 12), 0);
        let mounts = architecture::fixture_positions(&scene);
        scene
            .apply_appearance(AppearanceSettings {
                lighting_seed: Some(42),
                ..default()
            })
            .unwrap();
        assert_eq!(architecture::fixture_positions(&scene), mounts);
    }
}
