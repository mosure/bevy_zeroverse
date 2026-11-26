use super::*;
use crate::camera::{update_camera_trajectory, Playback};

/// One component per active room. The material handle is private to the animated
/// screen or aperture and remains valid while an annotation replaces its mesh material.
#[derive(Component, Default)]
pub(crate) struct AppearancePlayback {
    screens: Vec<Screen>,
    apertures: Vec<Aperture>,
}
struct Screen {
    at: f32,
    material: Handle<StandardMaterial>,
    pages: [Handle<Image>; 2],
    showing_next: bool,
}
struct Aperture {
    source: DimmedLight,
    material: Handle<StandardMaterial>,
    emission: LinearRgba,
}

#[derive(Component)]
pub(crate) struct DimmedLight {
    curve: DimmerCurve,
    rest: f32,
    last: f32,
}
impl DimmedLight {
    pub(crate) fn new(curve: DimmerCurve, rest: f32) -> Self {
        Self {
            curve,
            rest,
            last: 1.,
        }
    }
    fn change(&mut self, t: f32) -> Option<f32> {
        let factor = self.curve.multiplier_at(t);
        if factor == self.last {
            return None;
        }
        self.last = factor;
        Some(self.rest * factor)
    }
}

impl AppearancePlayback {
    pub(crate) fn screen(
        &mut self,
        at: f32,
        material: Handle<StandardMaterial>,
        pages: [Handle<Image>; 2],
    ) {
        self.screens.push(Screen {
            at,
            material,
            pages,
            showing_next: false,
        });
    }
    pub(crate) fn aperture(
        &mut self,
        curve: DimmerCurve,
        material: Handle<StandardMaterial>,
        emission: LinearRgba,
    ) {
        self.apertures.push(Aperture {
            source: DimmedLight::new(curve, 1.),
            material,
            emission,
        });
    }
}
pub(crate) fn install(app: &mut App) {
    app.add_systems(
        Update,
        (
            materials.run_if(any_with_component::<AppearancePlayback>),
            lights.run_if(any_with_component::<DimmedLight>),
        )
            .after(update_camera_trajectory),
    );
}
fn materials(
    playback: Res<Playback>,
    mut assets: ResMut<Assets<StandardMaterial>>,
    mut rooms: Query<&mut AppearancePlayback>,
) {
    let time = playback.trajectory_progress();
    for mut room in &mut rooms {
        for screen in &mut room.screens {
            let next = time >= screen.at;
            if next == screen.showing_next {
                continue;
            }
            let Some(mut mat) = assets.get_mut(&screen.material) else {
                continue;
            };
            mat.emissive_texture = Some(screen.pages[usize::from(next)].clone());
            screen.showing_next = next;
        }
        for aperture in &mut room.apertures {
            let gain = aperture.source.curve.multiplier_at(time);
            if gain == aperture.source.last {
                continue;
            }
            let Some(mut material) = assets.get_mut(&aperture.material) else {
                continue;
            };
            let c = aperture.emission;
            material.emissive =
                LinearRgba::new(c.red * gain, c.green * gain, c.blue * gain, c.alpha);
            aperture.source.last = gain;
        }
    }
}
fn lights(
    playback: Res<Playback>,
    mut lights: Query<(
        &mut DimmedLight,
        Option<&mut PointLight>,
        Option<&mut SpotLight>,
    )>,
) {
    let t = playback.trajectory_progress();
    for (mut dimmer, point, spot) in &mut lights {
        if let Some(power) = dimmer.change(t) {
            if let Some(mut light) = point {
                light.intensity = power;
            }
            if let Some(mut light) = spot {
                light.intensity = power;
            }
        }
    }
}
