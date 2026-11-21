use super::*;
use crate::scene::procedural_indoor::layout::IndoorManifest;

impl StringProgram {
    /// Linear RGB luminance of a physical diffuser, shared by material,
    /// reflection and light transport. Limit saturated LEDs' peak output.
    pub(crate) fn emission(&self, channel: usize, scale: f32) -> Option<LinearRgba> {
        let [r, g, b] = *self.colors_srgb.get(channel)?;
        let c = Color::srgb(r, g, b).to_linear();
        let y = c.red * 0.2126 + c.green * 0.7152 + c.blue * 0.0722;
        let gain = (self.luminance * self.channel_gain[channel] / y.max(0.001)).min(2500.) * scale;
        Some(LinearRgba::new(
            c.red * gain,
            c.green * gain,
            c.blue * gain,
            1.,
        ))
    }
}
pub(crate) fn illumination_scale(scene: &IndoorManifest) -> f32 {
    scene
        .appearance
        .as_ref()
        .map_or(1., |a| a.illumination_scale)
}

pub(crate) struct BulbLight {
    pub position: Vec3,
    pub axis: Vec3,
    pub radius: f32,
    pub color: Color,
    pub candela: f32,
    pub range: f32,
    pub cone: Option<[f32; 2]>,
}

impl BulbLight {
    pub(crate) fn bevy_intensity(&self) -> f32 {
        // Bevy divides both point and spot intensity by 4π in the renderer.
        4. * PI * self.candela
    }
}

impl bulb::Bulb {
    fn light(&self, tf: Transform, emission: LinearRgba) -> BulbLight {
        let c = Vec3::new(emission.red, emission.green, emission.blue);
        let y = c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
        let chroma = c / y.max(f32::MIN_POSITIVE);
        let candela = self.projected_area() * y;
        BulbLight {
            position: tf.transform_point(self.center()),
            axis: tf.rotation * self.axis,
            radius: self.radius * self.reflector.map_or(1., |r| r.lens_radius),
            color: Color::linear_rgb(chroma.x, chroma.y, chroma.z),
            candela,
            range: (candela / 0.30).sqrt().clamp(0.25, 5.),
            cone: self.reflector.map(Reflector::beam),
        }
    }
}

/// Without an irradiance volume, expose the near-field glow of miniature LEDs
/// to clustered PBR. They require no shadow views. A diode's influence stops
/// inside its backing or at the photometric cutoff, whichever is closer.
pub(crate) fn spawn_miniature_lights(
    scene: &IndoorManifest,
    root: Entity,
    commands: &mut Commands,
) {
    let scale = illumination_scale(scene);
    let wall_thickness = scene.envelope.as_ref().map_or(0.18, |e| e.wall_thickness());
    for o in scene
        .objects
        .iter()
        .filter(|o| o.kind == ObjectKind::StringLights && !practical(o))
    {
        let p = program(o);
        for (slot, bulb) in p.sockets(o).enumerate() {
            let center = bulb.center();
            let light = bulb.light(
                o.transform(),
                p.emission(bulb.channel, scale)
                    .expect("sampled bulb channel"),
            );
            let backing_limit = if p.suspended {
                // The generated roof slab is at least 50 mm thick.
                o.size.y - center.y + 0.045
            } else {
                center.z + 0.5 * o.size.z + wall_thickness - 0.003
            };
            // A miniature LED's 2-lux radius is at most a few decimetres.
            let range = (light.candela / 2.).sqrt().min(0.28).min(backing_limit);
            if range <= light.radius {
                continue;
            }
            commands.spawn((
                Name::new(format!("indoor_string_micro_{}_{slot}", o.id)),
                PointLight {
                    color: light.color,
                    intensity: light.bevy_intensity(),
                    range,
                    radius: light.radius,
                    shadow_maps_enabled: false,
                    ..default()
                },
                Transform::from_translation(light.position),
                ChildOf(root),
            ));
        }
    }
}

pub(crate) fn practical_lights(
    scene: &IndoorManifest,
) -> impl Iterator<Item = (usize, usize, BulbLight)> + '_ {
    let scale = illumination_scale(scene);
    scene
        .objects
        .iter()
        .filter(|o| practical(o))
        .flat_map(move |o| {
            let p = program(o);
            let tf = o.transform();
            p.sockets(o)
                .enumerate()
                .map(|(i, b)| {
                    let e = p.emission(b.channel, scale).expect("sampled bulb channel");
                    (o.id, i, b.light(tf, e))
                })
                .collect::<Vec<_>>()
        })
}

/// Mesh reflectors and path-traced references see the real emitting surface.
/// Only the small practical strands need a second BVH entry, since fine LED
/// strings already participate in the emissive transport geometry.
pub(crate) fn reflection_apertures(
    scene: &IndoorManifest,
    mut append: impl FnMut(&Geometry, Transform, LinearRgba),
) {
    let scale = illumination_scale(scene);
    for o in scene.objects.iter().filter(|o| practical(o)) {
        let p = program(o);
        for b in p.sockets(o) {
            let mut geometry = Geometry::default();
            b.aperture(&mut geometry);
            append(
                &geometry,
                o.transform(),
                p.emission(b.channel, scale).expect("sampled bulb channel"),
            );
        }
    }
}
