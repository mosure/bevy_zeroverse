//! Supported rigid movement. This catalog describes handling, not physics:
//! props slide/turn on their existing surface, chairs slide or roll on the floor.
use super::super::{
    layout::{IndoorManifest, IndoorObject, ObjectKind},
    objects,
    preparation::SceneGeometry,
};
use super::*;

pub(super) fn construct(a: &mut Assembly, o: &IndoorObject) {
    if a.motion_policy.is_none() || !a.joints.is_empty() {
        return;
    }
    use ObjectKind::*;
    let kind = match o.kind {
        Chair if o.variant < 11 && !objects::seating::is_recliner(o) => MotionKind::Chair,
        MobileCart => MotionKind::Cart,
        // Fixed/mounted versions are excluded by requiring a surface relationship.
        Laptop | Monitor | Phone | Keyboard | Mouse | Notebook | Notepad | Pencil | Books
        | WaterBottle | Mug | CoffeeCup | SodaCan | PenHolder | Microphone | Headphones
        | Stapler | TapeDispenser | DocumentTray | DeskOrganizer | TissueBox | Bowl | DeskLamp
        | DeskFan | Speaker | Kettle | Printer | CoffeeMaker | Projector | DryingRack
        | SoapDispenser | Sponge | KnifeBlock | UtensilHolder | PaperTowel | KitchenTowel
        | Dishware | CuttingBoard | FruitBowl | Bread | Vase
            if o.support.is_some() =>
        {
            MotionKind::Prop
        }
        Bag | StorageBox | TrashCan => MotionKind::Prop,
        _ => return,
    };
    if !a.motion_selected(kind, o.seed) {
        return;
    }
    let p = a.motion_policy.as_ref().unwrap();
    let (travel, turn) = match kind {
        MotionKind::Chair => (p.chair_travel_m, p.chair_swivel_degrees),
        MotionKind::Cart => (p.cart_travel_m, 60.),
        _ => (p.prop_travel_m, p.prop_rotation_degrees),
    };
    let mut r = stream(o.seed, 0x4d4f42494c45);
    let yaw = r.random_range(0.0..std::f32::consts::TAU);
    let translation = Vec3::new(yaw.cos(), 0., yaw.sin()) * travel * r.random_range(0.35..1.);
    let angle =
        turn.to_radians() * r.random_range(0.35..1.) * if r.random_bool(0.5) { -1. } else { 1. };
    let mut joint = construction::rigid(kind, o.seed, Vec3::ZERO, Vec3::Y, angle, translation);
    joint.contact = if o.support.is_some() {
        Contact::Surface
    } else {
        Contact::Floor
    };
    let body = Assembly {
        parts: std::mem::take(&mut a.parts),
        ..default()
    };
    a.articulated(body, joint);
}

/// A cart and everything recursively supported by it form one admission unit.
/// Cargo keeps its own mesh, semantic and annotation identity at playback.
pub(super) fn cargo(scene: &IndoorManifest, g: &SceneGeometry) -> Vec<Vec<usize>> {
    let mut result = vec![vec![]; scene.objects.len()];
    for (id, a) in g.objects.iter().enumerate() {
        if scene.objects[id].kind != ObjectKind::MobileCart
            || !a.joints.iter().any(|j| j.kind == MotionKind::Cart)
        {
            continue;
        }
        for o in &scene.objects {
            let mut parent = o.support;
            for _ in 0..scene.objects.len() {
                let Some(p) = parent else { break };
                if p == id {
                    result[id].push(o.id);
                    break;
                }
                parent = scene.objects.get(p).and_then(|o| o.support);
            }
        }
    }
    result
}

/// Remove motion tags before retagging cargo on repeated preparation. No vertex,
/// normal, UV, topology or finish is changed.
fn rest(a: &mut Assembly) {
    a.joints.clear();
    let parts = std::mem::take(&mut a.parts);
    for ((s, label), geometry) in parts {
        let label = label
            .split('#')
            .filter(|s| !s.starts_with("joint"))
            .collect::<Vec<_>>()
            .join("#");
        a.part(s, &label).append(geometry);
    }
}

pub(super) fn carry(scene: &IndoorManifest, g: &mut SceneGeometry, cargo: &[Vec<usize>]) {
    for (id, children) in cargo.iter().enumerate().filter(|(_, c)| !c.is_empty()) {
        let Some(parent) = g.objects[id].joints.first().cloned() else {
            continue;
        };
        let parent_tf = scene.objects[id].transform();
        for &child in children {
            let a = &mut g.objects[child];
            rest(a);
            if !parent.accepted {
                continue;
            }
            let tf = scene.objects[child].transform();
            let inverse = tf.compute_affine().inverse();
            let mut joint = construction::rigid(
                MotionKind::Cart,
                scene.objects[id].seed,
                inverse.transform_point3(parent_tf.transform_point(parent.pivot)),
                tf.rotation.inverse() * parent_tf.rotation * parent.axis,
                parent.effective_angle(),
                tf.rotation.inverse() * parent_tf.rotation * parent.effective_translation(),
            );
            joint.interval = parent.interval;
            joint.accepted = true;
            let body = Assembly {
                parts: std::mem::take(&mut a.parts),
                ..default()
            };
            a.articulated(body, joint);
        }
    }
}
